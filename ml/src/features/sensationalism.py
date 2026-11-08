"""Candidate features for detecting exaggerated framing, none of them shipped yet."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass


SUPERLATIVE = re.compile(r"\b\w+(?:íssim|érrim)[oa]s?\b", re.IGNORECASE)

_IMPERATIVES = (
    "compartilhe", "compartilha", "compartilhem", "compartilhar",
    "repassem",
    "divulguem", "espalhe", "espalhem", "encaminhe", "encaminhem",
    "reenvie", "reenviem", "propaguem",
)
IMPERATIVE = re.compile(r"\b(?:" + "|".join(_IMPERATIVES) + r")\b", re.IGNORECASE)

SECOND_PERSON = re.compile(r"\b(?:você|vocês|vc|vcs|teu|tua|teus|tuas|contigo)\b", re.IGNORECASE)

SUPPRESSION = re.compile(
    r"(?:"
    r"não querem que (?:você|voce|vc)"
    r"|não (?:vão|vao) te (?:contar|mostrar)"
    r"|a (?:mídia|midia|imprensa) (?:não |nao )?(?:esconde|escondeu|omite|não mostra|nao mostra)"
    r"|(?:estão|estao) escondendo"
    r"|ninguém (?:está|esta) falando"
    r"|antes que (?:apaguem|tirem do ar|removam|censurem)"
    r"|a verdade que (?:ninguém|ninguem|não|nao)"
    r")",
    re.IGNORECASE,
)

PATTERNS = {
    "superlative": SUPERLATIVE,
    "imperative": IMPERATIVE,
    "second_person": SECOND_PERSON,
    "suppression": SUPPRESSION,
}

ATTRIBUTION = re.compile(
    r"\b(?:segundo|de acordo com|conforme|afirmou|afirma|disse|declarou|informou|"
    r"segundo o|segundo a|em nota|em comunicado)\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"[0-9A-Za-zÀ-ÿ]+")
_SENTENCE = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class Hit:
    """One pattern match, with the span so the interface can show it in place."""

    kind: str
    start: int
    end: int
    match: str


def hits(text: str) -> list[Hit]:
    """Every pattern match, in document order."""
    found = [
        Hit(kind=kind, start=m.start(), end=m.end(), match=m.group(0))
        for kind, pattern in PATTERNS.items()
        for m in pattern.finditer(text or "")
    ]
    return sorted(found, key=lambda h: h.start)


def structural(text: str) -> dict[str, float]:
    """Features with no vocabulary at all, so synonym choice cannot evade them."""
    text = text or ""
    words = _WORD.findall(text)
    n = len(words) or 1
    per100 = 100.0 / n

    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    lengths = [len(_WORD.findall(s)) for s in sentences] or [0]

    return {
        "exclam_rate": text.count("!") * per100,
        "question_rate": text.count("?") * per100,
        "ellipsis_rate": len(re.findall(r"\.{3,}|…", text)) * per100,
        "multi_punct_rate": len(re.findall(r"!!|\?\?|\?!|!\?", text)) * per100,
        "quote_rate": len(re.findall(r"[\"“”']", text)) * per100,
        "digit_rate": len(re.findall(r"\d", text)) * per100,
        "mean_sentence_len": float(statistics.fmean(lengths)),
        "sentence_len_sd": float(statistics.pstdev(lengths)) if len(lengths) > 1 else 0.0,
        "attribution_rate": len(ATTRIBUTION.findall(text)) * per100,
    }


def features(text: str) -> dict[str, float]:
    """The full feature vector."""
    words = _WORD.findall(text or "")
    per100 = 100.0 / (len(words) or 1)
    counts = {f"{kind}_rate": 0.0 for kind in PATTERNS}
    for hit in hits(text):
        counts[f"{hit.kind}_rate"] += per100
    return {**counts, **structural(text)}


FEATURE_NAMES: tuple[str, ...] = tuple(features("").keys())
