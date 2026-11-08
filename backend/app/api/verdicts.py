"""Map a fact-checker's free-text rating onto a canonical verdict."""

from __future__ import annotations

import re

_PARTIAL = re.compile(
    r"(engano|exager|distorc|contexto|contextualiz|insustent|parcial|misleading|altered"
    r"|subestim|contradit|s[áa]tira|verdadeiro[,\s]+mas|\bexplicad|\bexplica\b"
    r"|n[ãa]o\s*[ée]\s*bem\s*assim)",
    re.IGNORECASE,
)

_TRUE = re.compile(
    r"^\W*(verdadeiro|comprovad\w*|confirmad\w*|aut[êe]ntic\w*|correto|procede"
    r"|fato(?!\s*(ou|/|\s)\s*fake))\b",
    re.IGNORECASE,
)

_FALSE = re.compile(
    r"(fals[ao]|boato|bulo|rumor|hoax|mentir|fake|inver[íi]d|golpe|n[ãa]o\s+[ée]\s+verdade)",
    re.IGNORECASE,
)

VERDICTS = ("true", "partial", "false", "unknown")


def canonical_verdict(rating: str) -> str:
    """One of `true`, `partial`, `false`, `unknown`."""
    text = (rating or "").strip()
    if not text:
        return "unknown"
    if _PARTIAL.search(text):
        return "partial"
    if _TRUE.match(text):
        return "true"
    if _FALSE.search(text):
        return "false"
    return "unknown"
