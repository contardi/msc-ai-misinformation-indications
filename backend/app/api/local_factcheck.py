"""Local ClaimReview index."""

from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from app.api.verdicts import canonical_verdict
from app.schemas.analyse import FactCheck

DEFAULT_PATH = "/app/data/processed/claimreview_br.csv"
MIN_SHARED = int(os.environ.get("LOCAL_MIN_SHARED", "2"))
MIN_CLAIM_TOKENS = int(os.environ.get("LOCAL_MIN_CLAIM_TOKENS", "4"))


def index_summary() -> bool:
    """Whether a row's `summary` is indexed alongside its `claim`."""
    return os.environ.get("INDEX_SUMMARY", "1").strip().lower() in ("1", "true", "yes", "on")
MIN_RATIO = float(os.environ.get("LOCAL_MIN_RATIO", "0.65"))

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+")


def _fold(text: str) -> str:
    """Strip diacritics, so `familia` matches `família`."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c)
    )


_STOPWORDS_ACCENTED = {
    "a", "o", "as", "os", "um", "uma", "de", "da", "do", "das", "dos", "em", "no", "na", "nos",
    "nas", "por", "para", "com", "sem", "sobre", "e", "ou", "mas", "que", "se", "como", "quando",
    "porque", "ao", "aos", "à", "às", "é", "ser", "está", "este", "esta", "esse", "essa", "isso",
    "isto", "ele", "ela", "eles", "elas", "seu", "sua", "seus", "suas", "já", "não", "sim", "mais",
    "menos", "muito", "também", "foi", "tem", "há", "the", "of", "and", "após", "entre", "pelo",
    "pela", "seja", "sao", "são", "ter", "vai", "faz", "diz",
}
_STOPWORDS = {_fold(w) for w in _STOPWORDS_ACCENTED}


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in _WORD_RE.findall(_fold(text).lower())
        if len(t) > 3 and t not in _STOPWORDS
    }


def index_path() -> Path:
    return Path(os.environ.get("CLAIMREVIEW_INDEX", DEFAULT_PATH))


class LocalClaimIndex:
    """In-memory inverted index over the local ClaimReview corpus."""

    def __init__(self, path: Path | None = None, rows: Iterable[dict] | None = None) -> None:
        """Build from a CSV path, or from any iterable of rows."""
        self.rows: list[dict] = []
        self.tokens: list[set[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.skipped = 0
        self.error: str | None = None
        if rows is not None:
            self._ingest(rows)
            self._build_idf()
        else:
            self._load(path or index_path())

    def _ingest(self, rows: Iterable[dict]) -> None:
        """Index rows carrying at least `claim`; `rating`, `publisher` and `url` are passed through."""
        with_summary = index_summary()
        for row in rows:
            claim = (row.get("claim") or "").strip()
            if not claim:
                continue
            indexed = f"{claim} {row.get('summary') or ''}" if with_summary else claim
            toks = _tokens(indexed)
            if len(toks) < MIN_CLAIM_TOKENS:
                self.skipped += 1
                continue
            self.postings_add(len(self.rows), toks)
            self.rows.append(row)
            self.tokens.append(toks)

    def _load(self, path: Path) -> None:
        if not path.exists():
            self.error = f"{path} not found"
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self._ingest(csv.DictReader(fh))
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self._build_idf()

    def postings_add(self, idx: int, toks: set[str]) -> None:
        for t in toks:
            self.postings[t].append(idx)

    def _build_idf(self) -> None:
        """Inverse document frequency over the claim corpus."""
        n = max(1, len(self.rows))
        self.idf = {t: math.log(n / len(set(ids))) for t, ids in self.postings.items()}

    def _weight(self, toks: set[str]) -> float:
        return sum(self.idf.get(t, 0.0) for t in toks)

    @property
    def ready(self) -> bool:
        return bool(self.rows)

    def search(self, query: str, limit: int = 5) -> list[FactCheck]:
        """Best-matching published checks for a query, most similar first."""
        ref = _tokens(query)
        if not ref or not self.rows:
            return []

        candidates: set[int] = set()
        for tok in ref:
            candidates.update(self.postings.get(tok, ()))

        scored: list[tuple[float, int]] = []
        for i in candidates:
            other = self.tokens[i]
            shared = ref & other
            if len(shared) < MIN_SHARED:
                continue
            w_shared = self._weight(shared)
            w_claim, w_query = self._weight(other), self._weight(ref)
            if w_claim <= 0 or w_query <= 0:
                continue
            score = max(w_shared / w_claim, w_shared / w_query)
            if score >= MIN_RATIO:
                scored.append((score, i))

        scored.sort(reverse=True)
        out: list[FactCheck] = []
        for _, i in scored[:limit]:
            row = self.rows[i]
            rating = row.get("rating", "")
            out.append(FactCheck(
                claim=row.get("claim", ""),
                rating=rating,
                publisher=row.get("publisher", ""),
                url=row.get("url", ""),
                verdict=canonical_verdict(rating),
            ))
        return out


_index: LocalClaimIndex | None = None


def get_index() -> LocalClaimIndex:
    """Process-wide singleton."""
    global _index
    if _index is None:
        _index = LocalClaimIndex()
    return _index


def is_enabled() -> bool:
    return get_index().ready


def search_local(query: str, limit: int = 5) -> list[FactCheck]:
    """Return matching checks from the local corpus, or [] when it is not present."""
    return get_index().search(query, limit)
