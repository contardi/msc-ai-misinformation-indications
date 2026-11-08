"""Online evidence layer."""

from __future__ import annotations

import os
import re
import time

import httpx

from app.api.verdicts import canonical_verdict
from app.schemas.analyse import CheckerStatus, FactCheck

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

MONITORED_CHECKERS: list[tuple[str, list[str]]] = [
    ("Boatos.org", ["boatos.org"]),
    ("Aos Fatos", ["aosfatos.org"]),
    ("Agência Lupa", ["agencialupa.org", "lupa.uol.com.br", "lupa.news"]),
    ("Estadão Verifica", ["estadao.com.br/estadao-verifica"]),
    ("UOL Comprova", ["uol.com.br/comprova", "projetocomprova.com.br", "/comprova"]),
    ("UOL Confere", ["uol.com.br/confere", "/confere"]),
    ("G1 Fato ou Fake", ["g1.globo.com/fato-ou-fake", "globo.com/fato-ou-fake"]),
    ("e-farsas", ["e-farsas.com"]),
    ("AFP Checamos", ["checamos.afp.com"]),
    ("Agência Tatu", ["agenciatatu.com.br"]),
]

_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos", "em", "no",
    "na", "nos", "nas", "por", "para", "com", "sem", "sob", "sobre", "e", "ou", "mas", "que", "se",
    "como", "quando", "porque", "ao", "aos", "à", "às", "é", "ser", "está", "este", "esta", "esse",
    "essa", "isso", "isto", "ele", "ela", "eles", "elas", "seu", "sua", "seus", "suas", "já", "não",
    "sim", "mais", "menos", "muito", "também", "foi", "ser", "tem", "há", "the", "of", "and",
}


def _api_key() -> str:
    return os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(_api_key())


def _timeout() -> float:
    try:
        return float(os.environ.get("FACTCHECK_TIMEOUT", "4.0"))
    except ValueError:
        return 4.0


def _max_results() -> int:
    try:
        return int(os.environ.get("FACTCHECK_MAX_RESULTS", "5"))
    except ValueError:
        return 5


MAX_QUERY_CHARS = 200
_MIN_SENTENCE_CHARS = 15
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _keyword_query(text: str, max_terms: int = 8, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Fallback."""
    tokens = re.findall(r"[0-9A-Za-zÀ-ÿ]+", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in tokens:
        if len(tok) < 4 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
        if len(keywords) >= max_terms:
            break
    return " ".join(keywords)[:max_chars]


def build_query(text: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Derive the search query."""
    first = _SENTENCE_END.split(text.strip(), maxsplit=1)[0]
    sentence = " ".join(first.split())
    if len(sentence) >= _MIN_SENTENCE_CHARS:
        return sentence[:max_chars]
    return _keyword_query(text, max_chars=max_chars)


_CACHE_TTL_SECONDS = 3600.0
_cache: dict[str, tuple[float, list[FactCheck]]] = {}


def _cache_get(query: str) -> list[FactCheck] | None:
    hit = _cache.get(query)
    if hit and (time.monotonic() - hit[0]) < _CACHE_TTL_SECONDS:
        return hit[1]
    return None


def _cache_put(query: str, results: list[FactCheck]) -> None:
    _cache[query] = (time.monotonic(), results)


def _fetch(query: str) -> dict:
    """Perform the HTTP request and return parsed JSON."""
    params = {
        "query": query,
        "languageCode": os.environ.get("FACTCHECK_LANGUAGE", "pt"),
        "pageSize": _max_results(),
        "key": _api_key(),
    }
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.get(FACTCHECK_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _map_claims(payload: dict, limit: int) -> list[FactCheck]:
    results: list[FactCheck] = []
    for claim in payload.get("claims", []):
        reviews = claim.get("claimReview") or []
        if not reviews:
            continue
        review = reviews[0]
        publisher = (review.get("publisher") or {}).get("name", "")
        url = review.get("url", "")
        if not url:
            continue
        rating = review.get("textualRating", "")
        results.append(
            FactCheck(
                claim=claim.get("text") or review.get("title", ""),
                rating=rating,
                publisher=publisher,
                url=url,
                verdict=canonical_verdict(rating),
            )
        )
        if len(results) >= limit:
            break
    return results


def _dedupe(results: list[FactCheck]) -> list[FactCheck]:
    """Union of several sources, de-duplicated by URL."""
    seen, out = set(), []
    for fc in results:
        key = (fc.url or fc.claim or "").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(fc)
    return out


LOCAL_CORPUS_CHECKERS = frozenset({"Boatos.org", "Agência Lupa", "UOL Comprova", "G1 Fato ou Fake"})


def _local_enabled() -> bool:
    """Whether any offline source is available - the bundled corpus or the harvested database."""
    for module, name in (("app.api.local_factcheck", "is_enabled"),
                         ("app.api.db_factcheck", "is_enabled")):
        try:
            mod = __import__(module, fromlist=[name])
            if getattr(mod, name)():
                return True
        except Exception:  # noqa: BLE001 - an unavailable source is simply not a source
            continue
    return False


def _search_all(text: str) -> list[FactCheck]:
    """Every deduped match from all three sources, UNCAPPED, or [] (disabled / no match / error)."""
    query = build_query(text)
    if not query:
        return []

    cached = _cache_get(query)
    if cached is not None:
        return cached

    results: list[FactCheck] = []

    if is_enabled():
        try:
            results.extend(_map_claims(_fetch(query), _max_results()))
        except Exception:  # noqa: BLE001 - external failure degrades to "no online evidence"
            pass

    try:
        from app.api.local_factcheck import search_local

        results.extend(search_local(query, _max_results()))
    except Exception:  # noqa: BLE001 - a missing or unreadable corpus is simply no local evidence
        pass

    try:
        from app.api.db_factcheck import search_db

        results.extend(search_db(query, _max_results()))
    except Exception:  # noqa: BLE001 - an unreachable database is simply no harvested evidence
        pass

    merged = _dedupe(results)
    _cache_put(query, merged)
    return merged


def search_fact_checks(text: str) -> list[FactCheck]:
    """The fact-checks to DISPLAY."""
    return _search_all(text)[: _max_results()]


def search_evidence(text: str) -> tuple[list[FactCheck], list[CheckerStatus]]:
    """The evidence list AND the per-checker status, from ONE retrieval."""
    all_results = _search_all(text)
    return all_results[: _max_results()], build_checker_status(all_results)


def build_checker_status(fact_checks: list[FactCheck]) -> list[CheckerStatus]:
    """Per-monitored-checker 'found / not found' status, derived from the returned fact-checks."""
    live, local = is_enabled(), _local_enabled()
    if not live and not local:
        return []
    statuses: list[CheckerStatus] = []
    for name, domains in MONITORED_CHECKERS:
        if not live and not any(d in LOCAL_CORPUS_CHECKERS for d in (name,)):
            continue
        match = next(
            (fc for fc in fact_checks if any(d in fc.url.lower() for d in domains)), None
        )
        statuses.append(
            CheckerStatus(name=name, found=match is not None, url=match.url if match else "")
        )
    return statuses
