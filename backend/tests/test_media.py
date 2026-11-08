"""Tests for audio/video ingestion and the /analyse/media endpoint."""

import glob
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingest import media as media_mod
from app.main import create_app


def _upload(client, data=b"x" * 1000, ctype="audio/mpeg", name="a.mp3"):
    return client.post("/analyse/media", files={"file": (name, io.BytesIO(data), ctype)})


def test_media_endpoint_exists_even_without_the_optional_deps(monkeypatch):
    """The light image must answer with a clear 503, not a 404."""
    monkeypatch.setattr("app.main.media_available", lambda: False)
    monkeypatch.setattr("app.main.media_unavailable_reason", lambda: "missing: faster-whisper")
    with TestClient(create_app()) as client:
        r = _upload(client)
    assert r.status_code == 503
    assert "faster-whisper" in r.json()["detail"]


@pytest.mark.parametrize(
    "data,ctype,fragment",
    [
        (b"", "audio/mpeg", "vazio"),
        (b"x" * 100, "application/pdf", "não suportado"),
        (b"x" * 100, "", "não suportado"),
    ],
)
def test_media_refusals_are_422_with_a_usable_message(monkeypatch, data, ctype, fragment):
    """A refusal the user can act on is a 422, not a 500."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    with TestClient(create_app()) as client:
        r = _upload(client, data=data, ctype=ctype)
    assert r.status_code == 422
    assert fragment in r.json()["detail"].lower()


def test_oversize_upload_is_refused_before_decoding(monkeypatch):
    """The size cap must apply before ffmpeg ever sees the bytes."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000")
    called = {"n": 0}
    real_probe = media_mod._probe_duration
    monkeypatch.setattr(media_mod, "_probe_duration",
                        lambda p: (called.__setitem__("n", called["n"] + 1), real_probe(p))[1])
    with TestClient(create_app()) as client:
        r = _upload(client, data=b"x" * 5000)
    assert r.status_code == 422
    assert "grande demais" in r.json()["detail"]
    assert called["n"] == 0, "the file reached a decoder despite exceeding the size cap"


def test_transcript_is_returned_so_a_mishearing_is_visible(monkeypatch):
    """The extracted text must come back to the user."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    monkeypatch.setattr(
        "app.main.transcribe_upload",
        lambda data, ctype, name="": media_mod.TranscriptionResult(
            text="urgente compartilhe essa mentira", kind="audio",
            duration_seconds=12.5, model="small", segments=2,
        ),
    )
    with TestClient(create_app()) as client:
        r = _upload(client)
    assert r.status_code == 200
    body = r.json()
    assert body["source"]["kind"] == "audio"
    assert body["source"]["extracted_text"] == "urgente compartilhe essa mentira"
    assert body["source"]["duration_seconds"] == 12.5


def test_media_path_obeys_the_scoring_guardrail(monkeypatch):
    """Score is the classifier's probability ONLY, on this path exactly as on the text path."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    monkeypatch.setattr(
        "app.main.transcribe_upload",
        lambda data, ctype, name="": media_mod.TranscriptionResult(
            text="o instituto divulgou os dados oficiais", kind="video", model="small",
        ),
    )
    with TestClient(create_app()) as client:
        r = _upload(client, ctype="video/mp4", name="v.mp4")
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["evidence"]["fact_checks"] == []
    assert body["corroboration"]["found_in_trusted_sources"] is False
    assert body["uncertainty_note"]


def test_silent_media_is_refused_rather_than_analysed_as_empty(monkeypatch):
    """No speech means nothing to verify."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    monkeypatch.setattr(
        "app.main.transcribe_upload",
        lambda data, ctype, name="": media_mod.TranscriptionResult(text="   ", kind="audio"),
    )
    with TestClient(create_app()) as client:
        r = _upload(client)
    assert r.status_code == 422
    assert "fala" in r.json()["detail"].lower()


def test_uploads_leave_nothing_on_disk(monkeypatch):
    """The backend is stateless and persists no request content - the no-ethics-approval position depends on it, and media must not become the exception."""
    monkeypatch.setattr("app.main.media_available", lambda: True)
    before = set(glob.glob("/tmp/ingest_*"))
    with TestClient(create_app()) as client:
        _upload(client, data=b"not really audio", ctype="audio/mpeg")
    assert set(glob.glob("/tmp/ingest_*")) == before


def test_text_endpoint_reports_its_source_as_text():
    """The contract gained a field; the existing path must fill it sensibly rather than omit it."""
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json={"text": "urgente compartilhe essa mentira"})
    assert r.json()["source"]["kind"] == "text"


def test_ffmpeg_stderr_never_reaches_the_user(monkeypatch, caplog):
    """A failed extraction must explain itself without quoting the decoder."""
    import logging
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout=b"",
            stderr=(
                b"[libx264 @ 0x5b09892fa800] encoder : Lavc61.19.101 libx264\n"
                b"Output #0, wav, to '/tmp/ingest_yb7eah6d/audio.wav':\n"
            ),
        )

    monkeypatch.setattr(media_mod.subprocess, "run", fake_run)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(media_mod.MediaError) as exc:
            media_mod._extract_audio(Path("in.mp4"), Path("out.wav"))

    message = str(exc.value)
    assert "áudio" in message
    assert "/tmp/" not in message
    assert "libx264" not in message
    assert "Output #0" not in message

    assert "libx264" in caplog.text


def test_the_real_audio_decoder_actually_runs():
    """faster-whisper's decoder is exercised for real, because importing it is not enough."""
    pytest.importorskip("faster_whisper")
    from faster_whisper.audio import decode_audio

    samples = decode_audio(io.BytesIO(media_mod._silent_wav()), sampling_rate=16000)

    assert len(samples) > 0, "the decoder returned no samples for a valid WAV"


def test_a_decoder_that_cannot_decode_is_reported_as_unavailable(monkeypatch):
    """A broken-but-importable dependency chain must take the 503 path, not raise."""
    pytest.importorskip("faster_whisper")
    import faster_whisper.audio as fa

    def broken(*_a, **_k):
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    monkeypatch.setattr(fa, "decode_audio", broken)
    media_mod._decoder_fault.cache_clear()
    monkeypatch.setattr(media_mod._decoder_fault, "cache_clear", lambda: None, raising=False)
    try:
        assert media_mod.is_available() is False
        reason = media_mod.unavailable_reason()
        assert "cannot decode" in reason
        assert "metadata_errors" in reason, "the actual fault must reach the operator"
        assert "requirements-media.txt" in reason, "and it must say where to look"
    finally:
        media_mod._decoder_fault.cache_clear()


def test_an_unexpected_transcription_fault_is_a_503_the_browser_can_read(monkeypatch):
    """An unexpected exception must not leave the route unhandled."""
    monkeypatch.setattr("app.main.media_available", lambda: True)

    def explode(*_a, **_k):
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    monkeypatch.setattr("app.main.transcribe_upload", explode)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        r = _upload(client, ctype="audio/mpeg")

    assert r.status_code == 503, "an unhandled exception would have been a bare 500"
    detail = r.json()["detail"]
    assert "transcrição está indisponível" in detail
    assert "metadata_errors" not in detail, "library internals must not reach the user"
    assert "TypeError" not in detail
