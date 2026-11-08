"""FastAPI entrypoint."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

_log = logging.getLogger(__name__)

from app.api import feedback as feedback_store
from app.api.factcheck import search_evidence
from app.ingest import MediaError, is_available as media_available, transcribe_upload
from app.ingest import unavailable_reason as media_unavailable_reason
from app.explanation.cue_stats import coverage
from app.explanation.cues import extract_cues
from app.explanation.templater import build_explanation
from app.schemas.analyse import (
    AnalyseRequest,
    AnalyseResponse,
    Corroboration,
    Evidence,
    Explanation,
    SourceInfo,
    TokenWeight,
)
from app.schemas.feedback import FeedbackRequest
from app.security import (
    allowed_origins,
    blocked_ips,
    current_feedback_rate_limit,
    current_media_rate_limit,
    current_rate_limit,
    get_client_ip,
)


def get_engine():
    """Select the serving engine by MODEL_KIND (default 'baseline')."""
    if os.environ.get("MODEL_KIND", "baseline").lower() == "transformer":
        from app.models import transformer_classifier as engine
    else:
        from app.models import classifier as engine
    return engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine = get_engine()
    if engine.load_model():
        print(f"[startup] model loaded from {engine.model_path()}")
    else:
        print(f"[startup] model NOT loaded ({engine.last_error()}). /analyse will return 503.")
    yield


def create_app() -> FastAPI:
    """Build the app."""
    app = FastAPI(title="XAI Misinformation Detector (PT-BR)", lifespan=lifespan)

    limiter = Limiter(key_func=get_client_ip)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def _denylist(request: Request, call_next):
        if get_client_ip(request) in blocked_ips():
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_ready": get_engine().is_ready()}

    @app.post("/analyse", response_model=AnalyseResponse)
    @limiter.limit(current_rate_limit)
    def analyse(request: Request, payload: AnalyseRequest) -> AnalyseResponse:
        engine = get_engine()
        if not engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Model artefact not available. Run training first (see README).",
            )

        return _assemble(engine, payload.text)

    @app.post("/analyse/media", response_model=AnalyseResponse)
    @limiter.limit(current_media_rate_limit)
    async def analyse_media(request: Request, file: UploadFile = File(...)) -> AnalyseResponse:
        """Audio or video in, the same contract out."""
        engine = get_engine()
        if not engine.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Model artefact not available. Run training first (see README).",
            )
        if not media_available():
            raise HTTPException(
                status_code=503,
                detail=("Ingestão de áudio/vídeo não está disponível neste servidor. "
                        f"({media_unavailable_reason()})"),
            )

        data = await file.read()
        try:
            result = transcribe_upload(data, file.content_type or "", file.filename or "")
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - deliberate, see below
            _log.exception("media transcription failed")
            raise HTTPException(
                status_code=503,
                detail=("Não foi possível processar este arquivo agora. "
                        "A transcrição está indisponível no servidor."),
            ) from exc

        if not result.text.strip():
            raise HTTPException(
                status_code=422,
                detail=("Nenhuma fala foi reconhecida neste arquivo, portanto não há texto para "
                        "verificar."),
            )
        return _assemble(
            engine,
            result.text,
            SourceInfo(kind=result.kind, extracted_text=result.text,
                       duration_seconds=result.duration_seconds, model=result.model,
                       notes=result.notes),
        )

    @app.post("/feedback", status_code=204)
    @limiter.limit(current_feedback_rate_limit)
    def feedback(request: Request, payload: FeedbackRequest) -> Response:
        """Record a reader's objection to a result."""
        if not feedback_store.record(payload):
            raise HTTPException(
                status_code=503,
                detail="Não foi possível registrar sua resposta agora.",
            )
        return Response(status_code=204)

    def _assemble(engine, text: str, source: SourceInfo | None = None) -> AnalyseResponse:
        """Build the response for a piece of text, whatever produced it."""
        prediction = engine.predict(text)

        fact_checks, checked = search_evidence(text)

        return AnalyseResponse(
            score=prediction.prob,
            label=prediction.label,
            uncertainty_note="Avaliação probabilística, não um veredito.",
            explanation=Explanation(
                text=build_explanation(prediction.prob, prediction.top_tokens),
                top_tokens=[TokenWeight(**t) for t in prediction.top_tokens],
            ),
            evidence=Evidence(
                fact_checks=fact_checks,
                note="Checagens públicas relacionadas. Quando ativado, a primeira frase do texto (até 200 caracteres) é enviada a um serviço externo de consulta; o texto completo nunca é enviado nem armazenado. A ausência de checagem não implica veracidade.",
            ),
            corroboration=Corroboration(
                found_in_trusted_sources=any(c.found for c in checked),
                note="Apenas contexto. A ausência de checagem não é prova de falsidade.",
                checked=checked,
            ),
            indicators=extract_cues(text),
            indicator_coverage=coverage(),
            source=source or SourceInfo(kind="text"),
        )

    return app


app = create_app()
