"""Contestability."""

import pytest
from fastapi.testclient import TestClient

from app.api import feedback as feedback_store
from app.main import create_app


@pytest.fixture(autouse=True)
def _reset():
    feedback_store.reset()
    yield
    feedback_store.reset()


@pytest.fixture()
def client():
    return TestClient(create_app())


VALID = {"reason": "evidence_irrelevant", "band": "high", "had_evidence": True, "locale": "pt-BR"}


def test_rejects_free_text(client, monkeypatch):
    """THE guard."""
    monkeypatch.setattr(feedback_store, "record", lambda _f: True)
    res = client.post("/feedback", json={**VALID, "comment": "meu nome é ... e eu moro em ..."})
    assert res.status_code == 422


def test_rejects_an_unknown_reason(client, monkeypatch):
    """The reason set is closed."""
    monkeypatch.setattr(feedback_store, "record", lambda _f: True)
    assert client.post("/feedback", json={**VALID, "reason": "qualquer coisa"}).status_code == 422


def test_rejects_a_score_masquerading_as_a_band(client, monkeypatch):
    """The band is three buckets."""
    monkeypatch.setattr(feedback_store, "record", lambda _f: True)
    assert client.post("/feedback", json={**VALID, "band": "0.7213"}).status_code == 422


def test_records_only_the_declared_fields(client, monkeypatch):
    """What reaches the store is the payload and nothing else - no IP, no header, no timestamp."""
    seen = {}
    monkeypatch.setattr(feedback_store, "record", lambda f: seen.update(f.model_dump()) or True)

    res = client.post(
        "/feedback",
        json=VALID,
        headers={"CF-Connecting-IP": "203.0.113.7", "User-Agent": "something-identifying"},
    )

    assert res.status_code == 204
    assert set(seen) == {"reason", "band", "had_evidence", "locale"}
    assert "203.0.113.7" not in str(seen)


def test_disabled_without_database_url(client, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert feedback_store.is_enabled() is False
    assert "DATABASE_URL not set" in feedback_store.status()
    assert client.post("/feedback", json=VALID).status_code == 503


def test_unreachable_database_does_not_raise(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    assert client.post("/feedback", json=VALID).status_code == 503
    assert feedback_store.is_enabled() is False
    assert feedback_store.status().startswith("disabled")


def test_a_broken_feedback_store_does_not_break_analyse(monkeypatch):
    """The whole point of the degradation contract."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    with TestClient(create_app()) as live:
        assert live.post("/feedback", json=VALID).status_code == 503
        analysed = live.post("/analyse", json={"text": "urgente compartilhe agora essa mentira"})
    assert analysed.status_code == 200


def test_record_returns_false_rather_than_raising(monkeypatch):
    """`record` is the boundary where exceptions stop."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    from app.schemas.feedback import FeedbackRequest

    assert feedback_store.record(FeedbackRequest(**VALID)) is False


def test_band_is_optional_for_the_public_build(client, monkeypatch):
    """VITE_PUBLIC_BUILD=1 shows no score, so there is no band the reader could be objecting to."""
    monkeypatch.setattr(feedback_store, "record", lambda _f: True)
    payload = {"reason": "evidence_missing", "had_evidence": False, "locale": "en-GB"}
    assert client.post("/feedback", json=payload).status_code == 204


@pytest.mark.parametrize(
    "reason",
    ["evidence_irrelevant", "evidence_missing", "score_too_high", "score_too_low",
     "explanation_unclear", "other"],
)
def test_every_offered_reason_is_accepted(client, monkeypatch, reason):
    """Each value the front-end can send must round-trip; a mismatch would fail only in production."""
    monkeypatch.setattr(feedback_store, "record", lambda _f: True)
    assert client.post("/feedback", json={**VALID, "reason": reason}).status_code == 204
