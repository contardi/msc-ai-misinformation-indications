"""Audio and video to text, via ffmpeg and faster-whisper."""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_log = logging.getLogger(__name__)

AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a", "audio/aac", "audio/ogg",
    "audio/opus", "audio/wav", "audio/x-wav", "audio/webm", "audio/flac", "audio/amr",
}
VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/quicktime", "video/webm", "video/x-matroska", "video/3gpp"}
ACCEPTED_TYPES = AUDIO_TYPES | VIDEO_TYPES


def max_upload_bytes() -> int:
    """Default 25 MB."""
    return int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


def transcribe_window_seconds() -> int:
    """How much of a long file to actually transcribe."""
    return int(os.environ.get("TRANSCRIBE_WINDOW_SECONDS", "300"))


def max_media_seconds() -> int:
    """Absolute refusal ceiling, purely a denial-of-service guard."""
    return int(os.environ.get("MAX_MEDIA_SECONDS", "3600"))


def ffmpeg_timeout() -> int:
    return int(os.environ.get("FFMPEG_TIMEOUT", "120"))


def whisper_model_name() -> str:
    return os.environ.get("WHISPER_MODEL", "small")


class MediaError(Exception):
    """A refusal the user should see, phrased for them rather than for a log."""


@dataclass
class TranscriptionResult:
    text: str
    kind: str
    duration_seconds: float = 0.0
    model: str = ""
    language: str = ""
    segments: int = 0
    notes: list[str] = field(default_factory=list)


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def _silent_wav() -> bytes:
    """A ~0.1 s silent mono WAV, built with the stdlib so the probe needs no ffmpeg and no fixture."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


@lru_cache(maxsize=1)
def _decoder_fault() -> str:
    """"" if faster-whisper can actually decode audio, else a description of why it cannot."""
    try:
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        return f"faster-whisper not installed ({exc})"
    try:
        decode_audio(io.BytesIO(_silent_wav()), sampling_rate=16000)
    except Exception as exc:  # noqa: BLE001 - see docstring
        return (f"faster-whisper cannot decode audio: {type(exc).__name__}: {exc}. "
                "Usually a dependency version skew; check the `av` pin in "
                "backend/requirements-media.txt.")
    return ""


def is_available() -> bool:
    if not _has("ffmpeg") or not _has("ffprobe"):
        return False
    return not _decoder_fault()


def unavailable_reason() -> str:
    """Why the feature is off, specifically enough to act on."""
    missing = [b for b in ("ffmpeg", "ffprobe") if not _has(b)]
    if missing:
        return (f"missing: {', '.join(missing)}. Rebuild the backend with "
                "--build-arg INSTALL_MEDIA=1 (see compose.yml).")
    fault = _decoder_fault()
    if fault.startswith("faster-whisper not installed"):
        return ("missing: faster-whisper. Rebuild the backend with "
                "--build-arg INSTALL_MEDIA=1 (see compose.yml).")
    return fault


def _probe_duration(path: Path) -> float:
    """Duration in seconds via ffprobe, or 0.0 if it cannot be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return float((out.stdout or "0").strip() or 0.0)
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def _extract_audio(src: Path, dst: Path, window: int | None = None) -> None:
    """Video (or any container) to 16 kHz mono WAV, which is what Whisper wants anyway."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src)]
    if window:
        cmd += ["-t", str(window)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
    proc = subprocess.run(
        cmd, capture_output=True, timeout=ffmpeg_timeout(), check=False,
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-500:].strip()
        _log.warning("ffmpeg failed (rc=%s): %s", proc.returncode, tail)
        raise MediaError(
            "Não foi possível extrair áudio deste arquivo. Ele pode estar corrompido ou não "
            "conter faixa de áudio."
        )


_model = None


def _get_model():
    """Load the Whisper model once per process."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(whisper_model_name(), device="cpu", compute_type="int8")
    return _model


def transcribe_upload(data: bytes, content_type: str, filename: str = "") -> TranscriptionResult:
    """Bytes in, transcript out."""
    if not is_available():
        raise MediaError(f"Transcrição não está disponível neste servidor. ({unavailable_reason()})")

    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype not in ACCEPTED_TYPES:
        raise MediaError(
            f"Tipo de arquivo não suportado: {ctype or 'desconhecido'}. "
            "Envie áudio (mp3, m4a, ogg, wav) ou vídeo (mp4, webm, mov)."
        )
    if not data:
        raise MediaError("O arquivo está vazio.")
    if len(data) > max_upload_bytes():
        raise MediaError(
            f"Arquivo grande demais: {len(data) / 1e6:.1f} MB. "
            f"O limite é {max_upload_bytes() / 1e6:.0f} MB."
        )

    kind = "video" if ctype in VIDEO_TYPES else "audio"
    tmpdir = Path(tempfile.mkdtemp(prefix="ingest_"))
    try:
        src = tmpdir / ("upload.bin")
        src.write_bytes(data)

        duration = _probe_duration(src)
        if duration > max_media_seconds():
            raise MediaError(
                f"Arquivo longo demais: {duration / 60:.1f} minutos. "
                f"O limite é {max_media_seconds() / 60:.0f} minutos."
            )

        window = transcribe_window_seconds()
        truncated = bool(duration and duration > window)

        wav = tmpdir / "audio.wav"
        _extract_audio(src, wav, window=window if truncated else None)

        model = _get_model()
        segments, info = model.transcribe(str(wav), language="pt", vad_filter=True)
        parts = [seg.text.strip() for seg in segments]
        text = " ".join(p for p in parts if p).strip()

        notes: list[str] = []
        if truncated:
            notes.append(
                f"Arquivo com {duration / 60:.0f} minutos: apenas os primeiros "
                f"{window // 60} minutos foram transcritos e verificados."
            )
        if not text:
            notes.append("Nenhuma fala foi detectada no arquivo.")
        return TranscriptionResult(
            text=text,
            kind=kind,
            duration_seconds=round(duration or getattr(info, "duration", 0.0) or 0.0, 1),
            model=whisper_model_name(),
            language=getattr(info, "language", "pt") or "pt",
            segments=len(parts),
            notes=notes,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            "O processamento do arquivo demorou tempo demais e foi interrompido."
        ) from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
