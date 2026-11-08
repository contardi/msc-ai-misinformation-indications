"""Recording a reader's objection to a result."""

from __future__ import annotations

import os

from app.schemas.feedback import FeedbackRequest

_error: str | None = None
_checked = False
_available = False


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def _connect():
    """Open a short-timeout connection."""
    import psycopg

    return psycopg.connect(database_url(), connect_timeout=5)


def is_enabled() -> bool:
    """Whether an objection could be stored right now."""
    global _checked, _available, _error
    if _checked:
        return _available
    _checked = True
    if not database_url():
        _error = "DATABASE_URL not set"
        return False
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM result_feedback LIMIT 1")
                cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - any failure means the feature is off, never a 500
        _error = f"{type(exc).__name__}: {exc}"
        return False
    _available = True
    return True


def status() -> str:
    """One line for the deploy runbook."""
    return "ok" if is_enabled() else f"disabled ({_error or 'unknown'})"


def record(feedback: FeedbackRequest) -> bool:
    """Store one objection."""
    global _error
    if not is_enabled():
        return False
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO result_feedback (reason, band, had_evidence, locale, day)
                    VALUES (%s, %s, %s, %s, CURRENT_DATE)
                    """,
                    (feedback.reason, feedback.band, feedback.had_evidence, feedback.locale),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - a failed write is a 503, never a 500
        _error = f"{type(exc).__name__}: {exc}"
        return False
    return True


def reset() -> None:
    """Drop the cached availability probe so the next call re-checks."""
    global _error, _checked, _available
    _error, _checked, _available = None, False, False
