"""Serve a fine-tuned transformer (BERTimbau/BERTugues) with calibrated probabilities + SHAP."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.classifier import Prediction

DEFAULT_MODEL_DIR = "/app/artifacts/bertugues"
DECISION_THRESHOLD = 0.5
TOP_K_TOKENS = 8


class _State:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.calibrator = None
        self.meta: dict = {}
        self.explainer = None
        self.ready = False
        self.error: str | None = None


_state = _State()


def model_path() -> Path:
    return Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))


def load_model(path: Path | None = None) -> bool:
    target = path or model_path()
    try:
        _state.tokenizer = AutoTokenizer.from_pretrained(target)
        _state.model = AutoModelForSequenceClassification.from_pretrained(target)
        _state.model.eval()
        cal = Path(target) / "calibrator.joblib"
        _state.calibrator = joblib.load(cal) if cal.exists() else None
        meta = Path(target) / "meta.json"
        _state.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        _state.explainer = None
        _state.ready = True
        _state.error = None
    except Exception as exc:  # noqa: BLE001 - surfaced via is_ready()/last_error()
        _state.ready = False
        _state.error = f"{type(exc).__name__}: {exc}"
    return _state.ready


def is_ready() -> bool:
    return _state.ready


def last_error() -> str | None:
    return _state.error


def _max_length() -> int:
    return int(_state.meta.get("max_length", 256))


def _fake_prob(texts: list[str]) -> np.ndarray:
    enc = _state.tokenizer(
        texts, truncation=True, max_length=_max_length(), padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _state.model(**enc).logits.cpu().numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def _calibrate(raw: float) -> float:
    if _state.calibrator is None:
        return float(raw)
    return float(_state.calibrator.predict([raw])[0])


def _shap_top_tokens(text: str, k: int = TOP_K_TOKENS) -> list[dict]:
    """SHAP token attributions for the transformer, bounded for latency; [] on any failure."""
    try:
        import shap

        if _state.explainer is None:
            masker = shap.maskers.Text(_state.tokenizer)
            _state.explainer = shap.Explainer(lambda xs: _fake_prob(list(xs)), masker)
        max_evals = int(os.environ.get("SHAP_MAX_EVALS", "120"))
        snippet = text[:400]
        values = _state.explainer([snippet], max_evals=max_evals, silent=True)
        toks = list(values.data[0])
        scores = list(values.values[0])
        pairs = [
            (str(t).strip(), float(s))
            for t, s in zip(toks, scores)
            if str(t).strip()
        ]
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return [{"token": t, "weight": round(w, 4)} for t, w in pairs[:k]]
    except Exception:  # noqa: BLE001 - attribution is best-effort; explanation text still renders
        return []


def predict(text: str) -> Prediction:
    if not _state.ready:
        raise RuntimeError("Transformer model not loaded.")
    raw = float(_fake_prob([text])[0])
    prob = _calibrate(raw)
    label_map = _state.meta.get("label_map", {0: "likely_legitimate", 1: "likely_misinformation"})
    predicted = 1 if prob >= DECISION_THRESHOLD else 0
    label = label_map.get(predicted, label_map.get(str(predicted), str(predicted)))
    return Prediction(prob=prob, label=label, top_tokens=_shap_top_tokens(text))
