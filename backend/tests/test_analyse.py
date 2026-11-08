"""API tests for /analyse, backed by the toy model artefact (see conftest.py)."""

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_analyse_contract_shape():
    with TestClient(app) as client:
        r = client.post("/analyse", json={"text": "urgente compartilhe essa mentira"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    for key in (
        "score", "label", "uncertainty_note", "explanation", "evidence", "corroboration",
        "indicators",
    ):
        assert key in body
    assert isinstance(body["explanation"]["top_tokens"], list)
    assert isinstance(body["indicators"], list)


def test_evidence_empty_and_no_external_call_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    with TestClient(app) as client:
        r = client.post("/analyse", json={"text": "URGENTE!! compartilhe https://x.co essa bomba"})
    body = r.json()
    assert body["evidence"]["fact_checks"] == []
    assert len(body["indicators"]) >= 1


def test_score_is_classifier_only_and_evidence_separate():
    """Guardrail."""
    with TestClient(app) as client:
        r = client.post("/analyse", json={"text": "o instituto divulgou os dados oficiais"})
    body = r.json()
    assert body["evidence"]["fact_checks"] == []
    assert body["corroboration"]["found_in_trusted_sources"] is False
    assert 0.0 <= body["score"] <= 1.0


def test_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/tmp/definitely-not-a-model.joblib")
    with TestClient(app) as client:
        r = client.post("/analyse", json={"text": "qualquer texto"})
    assert r.status_code == 503
    assert "training" in r.json()["detail"].lower()


def test_score_is_the_classifier_probability_and_cues_do_not_touch_it():
    """Guardrail."""
    from app.models import classifier

    loud = "URGENTE!! compartilhe essa bomba agora!!"
    quiet = "O instituto divulgou os dados oficiais da pesquisa nesta terça-feira."

    with TestClient(app) as client:
        for text in (loud, quiet):
            body = client.post("/analyse", json={"text": text}).json()
            assert body["score"] == classifier.predict(text).prob, text

    with TestClient(app) as client:
        assert len(client.post("/analyse", json={"text": loud}).json()["indicators"]) >= 1
        assert client.post("/analyse", json={"text": quiet}).json()["indicators"] == []
