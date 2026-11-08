"""The contestability contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Reason = Literal[
    "evidence_irrelevant",
    "evidence_missing",
    "score_too_high",
    "score_too_low",
    "explanation_unclear",
    "other",
]

Band = Literal["low", "inconclusive", "high"]

Locale = Literal["pt-BR", "en-GB"]


class FeedbackRequest(BaseModel):
    """One reader's objection to one result."""

    model_config = ConfigDict(extra="forbid")

    reason: Reason
    band: Band | None = None
    had_evidence: bool
    locale: Locale = "pt-BR"
