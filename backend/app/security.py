"""Public-launch hardening."""

from __future__ import annotations

import os

from starlette.requests import Request

DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
DEFAULT_RATE_LIMIT = "30/minute"
DEFAULT_MAX_TEXT_CHARS = 20_000


def _csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into a list of non-empty, stripped values."""
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def trust_proxy_headers() -> bool:
    return os.environ.get("TRUST_PROXY_HEADERS", "1").strip().lower() not in ("0", "false", "no", "")


def get_client_ip(request: Request) -> str:
    """Resolve the client IP, preferring CloudFlare's CF-Connecting-IP header."""
    if trust_proxy_headers():
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def current_rate_limit(*_args) -> str:
    """The active per-IP rate limit (read per request so it is env-tunable without a code change)."""
    return os.environ.get("RATE_LIMIT", DEFAULT_RATE_LIMIT)


def current_media_rate_limit() -> str:
    """Per-IP limit for /analyse/media, far stricter than the text endpoint."""
    return os.environ.get("MEDIA_RATE_LIMIT", "5/minute")


def current_feedback_rate_limit() -> str:
    """Per-IP limit for POST /feedback."""
    return os.environ.get("FEEDBACK_RATE_LIMIT", "10/minute")


def blocked_ips() -> set[str]:
    """The IP denylist, read per request so a deploy can update BLOCKED_IPS without code changes."""
    return set(_csv_env("BLOCKED_IPS"))


def allowed_origins() -> list[str]:
    """CORS allowlist; defaults to local dev origins when ALLOWED_ORIGINS is unset."""
    return _csv_env("ALLOWED_ORIGINS") or DEFAULT_ALLOWED_ORIGINS


def max_text_chars() -> int:
    try:
        return int(os.environ.get("MAX_TEXT_CHARS", DEFAULT_MAX_TEXT_CHARS))
    except ValueError:
        return DEFAULT_MAX_TEXT_CHARS


MAX_TEXT_CHARS = max_text_chars()

