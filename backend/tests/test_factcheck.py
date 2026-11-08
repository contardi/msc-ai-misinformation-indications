"""Tests for the online evidence layer (Google Fact Check Tools client)."""

import pytest

from app.api import factcheck

SAMPLE_PAYLOAD = {
    "claims": [
        {
            "text": "Vacina X causa efeito Y",
            "claimReview": [
                {
                    "publisher": {"name": "Aos Fatos"},
                    "url": "https://aosfatos.org/checagem-x",
                    "textualRating": "Falso",
                    "title": "É falso que...",
                }
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def _clear_cache():
    factcheck._cache.clear()
    yield
    factcheck._cache.clear()


def test_build_query_sends_only_the_first_sentence():
    """The query is one leading sentence - switched from a keyword bag on 2026-08-24 because Claim Search matches claim text (25.6% retrieval against 13.8%)."""
    q = factcheck.build_query(
        "Vacina causa autismo e a mídia esconde. A denúncia foi apagada. Compartilhe agora."
    )
    assert q == "Vacina causa autismo e a mídia esconde."
    assert "apagada" not in q and "Compartilhe" not in q


def test_build_query_caps_length():
    """The 200-character cap is the measured configuration AND the privacy bound - raising it sends more of the user's prose to a third party, so it is asserted rather than assumed."""
    q = factcheck.build_query("a" * 500 + ". segunda frase")
    assert len(q) == factcheck.MAX_QUERY_CHARS == 200


def test_build_query_falls_back_to_keywords_when_no_usable_sentence():
    """Text opening with a very short fragment would otherwise send a useless query."""
    q = factcheck.build_query("Veja. Governo e mídia esconderam denúncia sobre vacinas no país.")
    terms = q.split()
    assert q != "Veja."
    assert "governo" in terms and "vacinas" in terms
    assert len(terms) <= 8


def test_build_query_keeps_a_long_headline_without_punctuation():
    """A headline with no full stop is still natural language and should be sent as-is, not reduced to keywords - it is exactly the form that retrieved best in the 2026-08-24 measurement."""
    headline = "Governo confirma suspensão do Bolsa Família após as eleições de 2026"
    assert factcheck.build_query(headline) == headline


def test_monitored_checker_patterns_do_not_collide():
    """No single URL may match two checkers."""
    urls = [
        "https://noticias.uol.com.br/confere/ultimas-noticias/2026/08/19/x.htm",
        "https://noticias.uol.com.br/comprova/ultimas-noticias/2026/08/19/y.htm",
        "https://lupa.uol.com.br/jornalismo/2026/08/19/z",
        "https://www.estadao.com.br/estadao-verifica/w/",
        "https://oglobo.globo.com/fato-ou-fake/noticia/2026/08/v.ghtml",
        "https://www.aosfatos.org/noticias/u/",
        "https://www.agenciatatu.com.br/checagem/t/",
    ]
    for url in urls:
        hits = [
            name for name, domains in factcheck.MONITORED_CHECKERS
            if any(d in url.lower() for d in domains)
        ]
        assert len(hits) == 1, f"{url} matched {hits}"


def test_observador_is_not_a_monitored_brazilian_checker():
    """Observador is Portuguese."""
    url = "https://observador.pt/factcheck/2026/08/19/algo/"
    hits = [
        name for name, domains in factcheck.MONITORED_CHECKERS
        if any(d in url.lower() for d in domains)
    ]
    assert hits == []


def test_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    assert factcheck.is_enabled() is False
    monkeypatch.setattr(factcheck, "_fetch", lambda q: (_ for _ in ()).throw(AssertionError("called")))
    assert factcheck.search_fact_checks("qualquer texto urgente") == []


def test_maps_claimreview_to_factcheck(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.setattr(factcheck, "_fetch", lambda q: SAMPLE_PAYLOAD)
    results = factcheck.search_fact_checks("vacina causa efeito grave urgente")
    assert len(results) == 1
    fc = results[0]
    assert fc.claim == "Vacina X causa efeito Y"
    assert fc.rating == "Falso"
    assert fc.publisher == "Aos Fatos"
    assert fc.url == "https://aosfatos.org/checagem-x"


def test_graceful_empty_on_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")

    def boom(_q):
        raise RuntimeError("network down")

    monkeypatch.setattr(factcheck, "_fetch", boom)
    assert factcheck.search_fact_checks("texto qualquer sobre eleicoes") == []


def test_checker_status_marks_found_and_not_found(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    fcs = factcheck._map_claims(SAMPLE_PAYLOAD, 5)
    statuses = {s.name: s for s in factcheck.build_checker_status(fcs)}
    assert statuses["Aos Fatos"].found is True
    assert statuses["Aos Fatos"].url == "https://aosfatos.org/checagem-x"
    assert statuses["Boatos.org"].found is False


def test_checker_status_empty_only_when_nothing_was_searched(monkeypatch):
    """[] means "we checked nothing", so it must depend on BOTH sources, not just the API key."""
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.setattr(factcheck, "_local_enabled", lambda: False)
    assert factcheck.build_checker_status([]) == []


def test_checker_status_lists_only_checkers_a_running_source_could_return(monkeypatch):
    """With no API key the local corpus still runs, so claiming nothing was checked is false."""
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
    monkeypatch.setattr(factcheck, "_local_enabled", lambda: True)
    names = {c.name for c in factcheck.build_checker_status([])}
    assert names == {"Boatos.org", "Agência Lupa", "UOL Comprova", "G1 Fato ou Fake"}
    assert "Aos Fatos" not in names


def test_checker_status_lists_every_checker_when_the_live_source_runs(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    names = {c.name for c in factcheck.build_checker_status([])}
    assert names == {name for name, _ in factcheck.MONITORED_CHECKERS}


def test_cache_avoids_second_fetch(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    calls = {"n": 0}

    def counting_fetch(_q):
        calls["n"] += 1
        return SAMPLE_PAYLOAD

    monkeypatch.setattr(factcheck, "_fetch", counting_fetch)
    text = "vacina causa efeito grave urgente compartilhe"
    factcheck.search_fact_checks(text)
    factcheck.search_fact_checks(text)
    assert calls["n"] == 1

def test_checker_status_sees_matches_ranked_below_the_display_cap(monkeypatch):
    """A publisher whose check ranks below FACTCHECK_MAX_RESULTS is still reported as found."""
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    monkeypatch.setenv("FACTCHECK_MAX_RESULTS", "2")
    monkeypatch.setattr(factcheck, "_cache_get", lambda _q: None)
    monkeypatch.setattr(factcheck, "_cache_put", lambda _q, _v: None)
    monkeypatch.setattr(factcheck, "_fetch", lambda _q: {})

    ranked = [
        factcheck.FactCheck(claim="c1", rating="Falso", publisher="UOL",
                            url="https://noticias.uol.com.br/confere/a", verdict="false"),
        factcheck.FactCheck(claim="c2", rating="Falso", publisher="AFP",
                            url="https://checamos.afp.com/b", verdict="false"),
        factcheck.FactCheck(claim="c3", rating="Boato", publisher="Boatos.org",
                            url="https://www.boatos.org/c.html", verdict="false"),
    ]
    monkeypatch.setattr(factcheck, "_map_claims", lambda _p, _l: list(ranked))
    monkeypatch.setattr(factcheck, "_local_enabled", lambda: True)

    shown, statuses = factcheck.search_evidence("uma alegação qualquer sobre o governo federal")

    assert len(shown) == 2
    assert all("boatos.org" not in fc.url for fc in shown)

    by_name = {s.name: s for s in statuses}
    assert by_name["Boatos.org"].found is True, "a match below the cap was reported as not found"
    assert by_name["Boatos.org"].url == "https://www.boatos.org/c.html"


def test_monitored_checker_domains_cover_the_publisher_hosts_actually_indexed(monkeypatch):
    """A checker that moved domain must still be recognised, or the panel invents an absence."""
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "test-key")
    hosts = {
        "https://www.agencialupa.org/checagem/x": "Agência Lupa",
        "https://lupa.uol.com.br/jornalismo/2021/x": "Agência Lupa",
        "https://www.boatos.org/politica/x.html": "Boatos.org",
        "https://projetocomprova.com.br/publicações/x": "UOL Comprova",
        "https://checamos.afp.com/doc.afp.com.X": "AFP Checamos",
    }
    for url, expected in hosts.items():
        fc = factcheck.FactCheck(claim="c", rating="Falso", publisher=expected, url=url,
                                 verdict="false")
        found = [s.name for s in factcheck.build_checker_status([fc]) if s.found]
        assert found == [expected], f"{url} was not attributed to {expected}, got {found}"
