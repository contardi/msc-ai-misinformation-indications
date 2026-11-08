"""Tests for the public-launch hardening."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.security import get_client_ip

ALLOWED_TEXT = {"text": "urgente compartilhe essa mentira"}


def _fake_request(headers: dict, client_host: str | None):
    client = SimpleNamespace(host=client_host) if client_host is not None else None
    return SimpleNamespace(headers=headers, client=client)


def test_client_ip_prefers_cf_connecting_ip():
    req = _fake_request({"cf-connecting-ip": "1.2.3.4", "x-forwarded-for": "5.6.7.8"}, "9.9.9.9")
    assert get_client_ip(req) == "1.2.3.4"


def test_client_ip_falls_back_to_first_forwarded_hop():
    req = _fake_request({"x-forwarded-for": "5.6.7.8, 1.1.1.1"}, "9.9.9.9")
    assert get_client_ip(req) == "5.6.7.8"


def test_client_ip_falls_back_to_socket_peer():
    req = _fake_request({}, "9.9.9.9")
    assert get_client_ip(req) == "9.9.9.9"


def test_client_ip_ignores_headers_when_proxy_untrusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "0")
    req = _fake_request({"cf-connecting-ip": "1.2.3.4"}, "9.9.9.9")
    assert get_client_ip(req) == "9.9.9.9"


def test_empty_text_rejected():
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json={"text": ""})
    assert r.status_code == 422


def test_oversize_text_rejected():
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json={"text": "a" * 20_001})
    assert r.status_code == 422


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json=ALLOWED_TEXT, headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_omits_header_for_disallowed_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json=ALLOWED_TEXT, headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_blocked_ip_gets_403(monkeypatch):
    monkeypatch.setenv("BLOCKED_IPS", "203.0.113.9")
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json=ALLOWED_TEXT, headers={"CF-Connecting-IP": "203.0.113.9"})
    assert r.status_code == 403


def test_non_blocked_ip_passes(monkeypatch):
    monkeypatch.setenv("BLOCKED_IPS", "203.0.113.9")
    with TestClient(create_app()) as client:
        r = client.post("/analyse", json=ALLOWED_TEXT, headers={"CF-Connecting-IP": "198.51.100.4"})
    assert r.status_code == 200


def test_rate_limit_returns_429_after_threshold(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    headers = {"CF-Connecting-IP": "203.0.113.55"}
    with TestClient(create_app()) as client:
        statuses = [client.post("/analyse", json=ALLOWED_TEXT, headers=headers).status_code for _ in range(4)]
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429
