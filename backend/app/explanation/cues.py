"""Descriptive linguistic cues - extra *indications* for the reader, not score inputs."""

from __future__ import annotations

import re

from app.explanation.cue_stats import evidence_for
from app.schemas.analyse import CueHit, Indicator

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+")

MAX_HITS_PER_CUE = 3
CONTEXT_CHARS = 60


def _excerpt(text: str, start: int, end: int) -> str:
    """The matched span with a little text either side, trimmed to whole words."""
    left = max(0, start - CONTEXT_CHARS)
    right = min(len(text), end + CONTEXT_CHARS)
    chunk = text[left:right].strip()
    if left > 0 and " " in chunk:
        chunk = chunk.split(" ", 1)[1]
    if right < len(text) and " " in chunk:
        chunk = chunk.rsplit(" ", 1)[0]
    chunk = " ".join(chunk.split())
    return f"{'...' if left > 0 else ''}{chunk}{'...' if right < len(text) else ''}"


def _hit(text: str, start: int, end: int) -> CueHit:
    return CueHit(start=start, end=end, match=text[start:end], excerpt=_excerpt(text, start, end))


def _hits(text: str, spans: list[tuple[int, int]], limit: int) -> list[CueHit]:
    """Up to `limit` hits, none of them showing the reader the same passage twice."""
    out: list[CueHit] = []
    last_start: int | None = None
    for start, end in spans:
        if last_start is not None and start - last_start < CONTEXT_CHARS:
            continue
        out.append(_hit(text, start, end))
        last_start = start
        if len(out) == limit:
            break
    return out


_SENSATIONAL = {
    "urgente", "urgentíssimo", "compartilhe", "compartilha", "compartilhem", "repassem",
    "chocante", "bomba", "atenção", "imperdível",
}


def extract_cues(text: str) -> list[Indicator]:
    cues: list[Indicator] = []

    exclamations = text.count("!")
    if exclamations >= 2:
        marks = [(m.start(), m.start() + 1) for m in re.finditer(r"!", text)]
        cues.append(
            Indicator(
                kind="exclamations",
                label="Pontuação sensacionalista",
                detail=f"{exclamations} pontos de exclamação.",
                count=exclamations,
                evidence=evidence_for("exclamations"),
                hits=_hits(text, marks, 1),
            )
        )

    seen: set[str] = set()
    positions: list[tuple[int, int]] = []
    for match in _WORD_RE.finditer(text):
        if match.group(0).lower() in _SENSATIONAL:
            seen.add(match.group(0).lower())
            positions.append((match.start(), match.end()))
    if seen:
        terms = sorted(seen)
        cues.append(
            Indicator(
                kind="sensational_terms",
                label="Termos de apelo",
                detail="Termos como: " + ", ".join(terms) + ".",
                terms=terms,
                evidence=evidence_for("sensational_terms"),
                hits=_hits(text, positions, MAX_HITS_PER_CUE),
            )
        )

    return cues
