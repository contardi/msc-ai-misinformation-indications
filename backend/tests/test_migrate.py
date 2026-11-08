"""The startup migration's contract, which is mostly about when it must NOT fail."""

import pytest

from app import migrate


def test_no_database_is_not_a_failure(monkeypatch, tmp_path):
    """The harvested cache is OPTIONAL, so a deployment without one must still start."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCHEMA_PATH", str(tmp_path / "absent.sql"))
    assert migrate.main([]) == 0


def test_a_configured_database_that_cannot_be_migrated_fails_the_deploy(monkeypatch, tmp_path):
    """Loud at deploy time beats subtle at request time."""
    schema = tmp_path / "schema.sql"
    schema.write_text("SELECT 1;", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/none")
    monkeypatch.setenv("SCHEMA_PATH", str(schema))
    assert migrate.main([]) == 1


def test_a_missing_schema_file_fails_rather_than_silently_skipping(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/none")
    monkeypatch.setenv("SCHEMA_PATH", str(tmp_path / "not-here.sql"))
    assert migrate.main([]) == 1


def test_the_connection_string_never_reaches_the_log(monkeypatch, tmp_path, caplog):
    """A DATABASE_URL carries a password, and this module logs on every failure path."""
    schema = tmp_path / "schema.sql"
    schema.write_text("SELECT 1;", encoding="utf-8")
    secret = "sup3rs3cr3t-should-not-appear"
    monkeypatch.setenv("DATABASE_URL", f"postgresql://user:{secret}@127.0.0.1:1/none")
    monkeypatch.setenv("SCHEMA_PATH", str(schema))

    with caplog.at_level("INFO"):
        assert migrate.main([]) == 1

    assert secret not in caplog.text, "the password reached the log"


def test_an_empty_schema_file_is_an_error_not_a_no_op(tmp_path):
    """Guards the case where a build copies the file but the file is truncated."""
    schema = tmp_path / "schema.sql"
    schema.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        migrate.apply_schema("postgresql://unused", schema)


def test_the_default_schema_path_matches_the_image_layout():
    """`/app/app/migrate.py` must resolve to `/app/infra/postgres/schema.sql`."""
    assert migrate.DEFAULT_SCHEMA.parts[-3:] == ("infra", "postgres", "schema.sql")
