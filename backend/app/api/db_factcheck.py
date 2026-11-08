"""Fact-check retrieval backed by Postgres - the same matcher, a different row source."""

from __future__ import annotations

import os
import time

from app.api.local_factcheck import LocalClaimIndex
from app.schemas.analyse import FactCheck

MAX_ROWS = int(os.environ.get("DB_FACTCHECK_MAX_ROWS", "200000"))

DEFAULT_TTL_SECONDS = 3600

_index: LocalClaimIndex | None = None
_error: str | None = None
_loaded = False
_loaded_at = 0.0


def ttl_seconds() -> int:
    return int(os.environ.get("DB_FACTCHECK_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))


def _fresh() -> bool:
    """Whether the cached result (index OR error) is still within its TTL."""
    ttl = ttl_seconds()
    return ttl <= 0 or (time.monotonic() - _loaded_at) < ttl


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def _fetch_rows() -> list[dict]:
    """Every usable fact-check, shaped exactly like a row of the bundled CSV."""
    import psycopg

    with psycopg.connect(database_url(), connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT url, claim, rating, verdict, publisher, summary
                FROM fact_checks
                WHERE claim <> ''
                ORDER BY published_at DESC NULLS LAST
                LIMIT %s
                """,
                (MAX_ROWS,),
            )
            return [
                {"url": r[0], "claim": r[1], "rating": r[2], "verdict": r[3], "publisher": r[4],
                 "summary": r[5]}
                for r in cur.fetchall()
            ]


def get_index() -> LocalClaimIndex | None:
    """Process-wide singleton, built on first use."""
    global _index, _error, _loaded, _loaded_at
    if _loaded and _fresh():
        return _index
    _loaded = True
    _loaded_at = time.monotonic()
    _index, _error = None, None
    if not database_url():
        _error = "DATABASE_URL not set"
        return None
    try:
        rows = _fetch_rows()
    except Exception as exc:  # noqa: BLE001 - an unreachable database is a disabled source
        _error = f"{type(exc).__name__}: {exc}"
        return None
    if not rows:
        _error = "no fact_checks rows"
        return None
    _index = LocalClaimIndex(rows=rows)
    return _index


def is_enabled() -> bool:
    index = get_index()
    return bool(index and index.ready)


def status() -> str:
    """One line for /health and the deploy runbook."""
    index = get_index()
    if index and index.ready:
        return f"ok ({len(index.rows)} checks)"
    return f"disabled ({_error or 'unknown'})"


def search_db(query: str, limit: int = 5) -> list[FactCheck]:
    index = get_index()
    if not index or not index.ready:
        return []
    return index.search(query, limit)


def reset() -> None:
    """Drop the cached index so the next call rebuilds."""
    global _index, _error, _loaded, _loaded_at
    _index, _error, _loaded, _loaded_at = None, None, False, 0.0
