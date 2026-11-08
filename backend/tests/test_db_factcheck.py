"""The database-backed source must degrade to silence, never take a request down with it."""

import pytest

from app.api import db_factcheck, factcheck


@pytest.fixture(autouse=True)
def _reset():
    db_factcheck.reset()
    factcheck._cache.clear()
    yield
    db_factcheck.reset()
    factcheck._cache.clear()


def test_disabled_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db_factcheck.is_enabled() is False
    assert db_factcheck.search_db("qualquer texto sobre vacinas") == []
    assert "DATABASE_URL not set" in db_factcheck.status()


def test_unreachable_database_disables_the_source_without_raising(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")
    assert db_factcheck.search_db("vacina causa autismo") == []
    assert db_factcheck.is_enabled() is False
    assert db_factcheck.status().startswith("disabled")


def test_search_fact_checks_survives_a_broken_database(monkeypatch):
    """The whole point."""
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nothing")

    def boom(*_a, **_k):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(db_factcheck, "search_db", boom)
    assert isinstance(factcheck.search_fact_checks("urnas eletrônicas foram fraudadas"), list)


def test_index_is_the_same_matcher_not_a_new_one():
    """Storage changed; the matcher did not."""
    from app.api.local_factcheck import LocalClaimIndex

    target = "As urnas eletrônicas foram fraudadas na eleição presidencial brasileira"
    rows = [{"url": "https://example.org/a", "claim": target, "rating": "Falso",
             "verdict": "false", "publisher": "Exemplo"}]
    rows += [{"url": f"https://example.org/f{i}",
              "claim": f"Notícia comum número {i} sobre economia agricultura clima transporte",
              "rating": "", "verdict": "unknown", "publisher": "Exemplo"} for i in range(60)]
    rows += [{"url": "https://example.org/short", "claim": "curto", "rating": "",
              "verdict": "unknown", "publisher": "Exemplo"}]

    index = LocalClaimIndex(rows=rows)
    assert index.ready
    assert index.skipped == 1, "rows below MIN_CLAIM_TOKENS must be skipped, exactly as with a CSV"
    hits = index.search("urnas eletrônicas foram fraudadas na eleição de 2022", limit=5)
    assert hits and hits[0].url == "https://example.org/a"
    assert hits[0].rating == "Falso"


def test_a_tiny_index_retrieves_nothing_and_that_is_arithmetic_not_a_bug():
    """A nearly-empty harvest is silently useless, which the deploy runbook has to know."""
    from app.api.local_factcheck import LocalClaimIndex

    claim = "As urnas eletrônicas foram fraudadas na eleição presidencial"
    single = LocalClaimIndex(rows=[{"url": "https://example.org/a", "claim": claim,
                                    "rating": "Falso", "verdict": "false", "publisher": "X"}])
    assert single.ready, "the index loads"
    assert single.search(claim, 5) == [], "but it cannot rank a single document"


def test_row_source_and_csv_source_agree():
    """Same rows, two construction paths, identical retrieval - or the storage swap changed results."""
    import csv as _csv
    import os

    path = os.environ.get("CLAIMREVIEW_INDEX", "/app/data/processed/claimreview_br.csv")
    if not os.path.exists(path):
        pytest.skip("bundled corpus not mounted")

    from app.api.local_factcheck import LocalClaimIndex

    with open(path, encoding="utf-8") as fh:
        rows = [r for _, r in zip(range(2000), _csv.DictReader(fh))]

    from_rows = LocalClaimIndex(rows=rows)
    query = next(r["claim"] for r in rows if len(r.get("claim", "").split()) > 8)
    assert [fc.url for fc in from_rows.search(query, 5)], "expected the index to find its own claim"

def test_retrieval_is_accent_insensitive():
    """The same claim must be found whether or not the reader typed the accents."""
    from app.api.local_factcheck import _tokens, LocalClaimIndex

    target = "Eleição terá fiscalização automática na votação eletrônica"
    rows = [{"url": "https://example.org/a", "claim": target, "rating": "Falso",
             "verdict": "false", "publisher": "Exemplo"}]
    rows += [{"url": f"https://example.org/f{i}",
              "claim": f"Boletim comum número {i} sobre economia agricultura clima transporte",
              "rating": "", "verdict": "unknown", "publisher": "Exemplo"} for i in range(60)]
    index = LocalClaimIndex(rows=rows)

    accented = index.search("Eleição terá fiscalização automática na votação eletrônica", limit=5)
    plain = index.search("Eleicao tera fiscalizacao automatica na votacao eletronica", limit=5)
    assert [fc.url for fc in accented] == ["https://example.org/a"]
    assert [fc.url for fc in plain] == [fc.url for fc in accented], (
        "dropping the accents changed the results, so the index is accent-sensitive again"
    )

    assert _tokens("Bolsa Família após João ação") == _tokens("Bolsa Familia apos Joao acao")
    assert "apos" not in _tokens("texto apos alguma coisa relevante")
    assert "tambem" not in _tokens("isso tambem aconteceu naquele lugar")


def test_the_index_is_rebuilt_once_the_ttl_elapses_and_not_before(monkeypatch):
    """A scheduled harvest must become visible WITHOUT a redeploy, and not on every request either."""
    from app.api import db_factcheck as db

    calls = {"n": 0}
    rows = [{"url": "https://example.org/a", "claim": "urnas eletrônicas foram fraudadas na eleição",
             "rating": "Falso", "verdict": "false", "publisher": "X", "summary": None}]
    rows += [{"url": f"https://example.org/f{i}",
              "claim": f"Notícia comum número {i} sobre economia agricultura clima transporte",
              "rating": "", "verdict": "unknown", "publisher": "X", "summary": None}
             for i in range(60)]

    def counting_fetch():
        calls["n"] += 1
        return list(rows)

    now = {"t": 1000.0}
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("DB_FACTCHECK_TTL_SECONDS", "60")
    monkeypatch.setattr(db, "_fetch_rows", counting_fetch)
    monkeypatch.setattr(db.time, "monotonic", lambda: now["t"])
    db.reset()
    try:
        assert db.get_index() is not None
        assert calls["n"] == 1

        now["t"] += 59
        db.get_index()
        db.get_index()
        assert calls["n"] == 1, "the index was rebuilt inside its TTL"

        rows.append({"url": "https://example.org/new",
                     "claim": "vacina contra gripe causa paralisia segundo boato antigo",
                     "rating": "Falso", "verdict": "false", "publisher": "Y", "summary": None})
        now["t"] += 2
        index = db.get_index()
        assert calls["n"] == 2, "the index was not rebuilt after the TTL elapsed"
        assert any("paralisia" in r["claim"] for r in index.rows), "the refresh is not visible"
    finally:
        db.reset()


def test_a_ttl_of_zero_pins_the_index_for_the_life_of_the_process(monkeypatch):
    """The escape hatch, so a deployment can opt out of rebuilding entirely."""
    from app.api import db_factcheck as db

    calls = {"n": 0}

    def counting_fetch():
        calls["n"] += 1
        return [{"url": f"https://example.org/{i}",
                 "claim": f"Notícia comum número {i} sobre economia agricultura clima transporte",
                 "rating": "", "verdict": "unknown", "publisher": "X", "summary": None}
                for i in range(60)]

    now = {"t": 1000.0}
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("DB_FACTCHECK_TTL_SECONDS", "0")
    monkeypatch.setattr(db, "_fetch_rows", counting_fetch)
    monkeypatch.setattr(db.time, "monotonic", lambda: now["t"])
    db.reset()
    try:
        db.get_index()
        now["t"] += 100_000
        db.get_index()
        assert calls["n"] == 1
    finally:
        db.reset()
