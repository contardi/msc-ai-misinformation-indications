"""Apply the database schema, idempotently, before the API starts serving."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("app.migrate")

DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "infra" / "postgres" / "schema.sql"


def schema_path() -> Path:
    return Path(os.environ.get("SCHEMA_PATH", str(DEFAULT_SCHEMA)))


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def apply_schema(url: str, path: Path) -> str:
    """Run the schema file in one transaction."""
    import psycopg

    sql = path.read_text(encoding="utf-8")
    if not sql.strip():
        raise ValueError(f"{path} is empty")

    with psycopg.connect(url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM publishers")
            publishers = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM fact_checks")
            checks = cur.fetchone()[0]
    return f"{publishers} publishers, {checks} fact_checks"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[migrate] %(message)s")
    url = database_url()
    if not url:
        _log.info("DATABASE_URL not set; nothing to migrate (the harvested cache is optional)")
        return 0

    path = schema_path()
    if not path.exists():
        _log.error("schema file not found at %s", path)
        return 1

    try:
        summary = apply_schema(url, path)
    except Exception as exc:  # noqa: BLE001 - every failure here must fail the deploy
        _log.error("schema failed to apply: %s: %s", type(exc).__name__, exc)
        return 1

    _log.info("schema applied from %s (%s)", path.name, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
