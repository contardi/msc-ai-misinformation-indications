"""Pydantic models for the /analyse contract (see architecture.md)."""

from pydantic import BaseModel, Field

from app.security import MAX_TEXT_CHARS


class AnalyseRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Brazilian Portuguese text to assess.",
    )


class TokenWeight(BaseModel):
    token: str
    weight: float


class Explanation(BaseModel):
    text: str
    top_tokens: list[TokenWeight] = []


class FactCheck(BaseModel):
    claim: str
    rating: str
    publisher: str
    url: str
    verdict: str = "unknown"


class Evidence(BaseModel):
    fact_checks: list[FactCheck] = []
    note: str = ""


class CheckerStatus(BaseModel):
    name: str
    found: bool
    url: str = ""


class Corroboration(BaseModel):
    found_in_trusted_sources: bool = False
    note: str = ""
    checked: list[CheckerStatus] = []


class CueEvidence(BaseModel):
    """How often this cue actually fired on labelled data, so the reader can weigh it."""

    fake_rate: float
    real_rate: float
    precision: float
    n_fired: int
    corpus: str


class CueHit(BaseModel):
    """One place a cue fired, with enough context to show it without the original text."""

    start: int
    end: int
    match: str
    excerpt: str


class Indicator(BaseModel):
    """A descriptive stylistic cue (links, ALL-CAPS, sensational wording)."""

    kind: str = ""
    label: str
    detail: str
    count: int | None = None
    terms: list[str] = Field(default_factory=list)
    evidence: CueEvidence | None = None
    hits: list[CueHit] = Field(default_factory=list)


class IndicatorCoverage(BaseModel):
    """What the cue panel can and cannot see."""

    detection_rate: float
    false_alarm_rate: float
    one_in: int
    corpus: str


class SourceInfo(BaseModel):
    """What was actually analysed, when the input was not plain text."""
    kind: str = "text"
    extracted_text: str = ""
    duration_seconds: float = 0.0
    model: str = ""
    notes: list[str] = []


class AnalyseResponse(BaseModel):
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Classifier calibrated probability - the only number. Nothing else blended in.",
    )
    label: str
    uncertainty_note: str
    explanation: Explanation
    evidence: Evidence
    corroboration: Corroboration
    indicators: list[Indicator] = []
    indicator_coverage: IndicatorCoverage | None = None
    source: SourceInfo = SourceInfo()
