"""Measured evidence for each style cue, as shown to the reader."""

from __future__ import annotations

from app.schemas.analyse import CueEvidence, IndicatorCoverage

SOURCE = "ml/results/cue_precision.csv"
CORPUS = "fakerecogna2"
CORPUS_LABEL = "FakeRecogna 2.0"
MEASURED_ON = "2026-09-10"

CUE_EVIDENCE: dict[str, CueEvidence] = {
    "exclamations": CueEvidence(
        fake_rate=10.45, real_rate=0.33, precision=96.94, n_fired=2846,
        corpus=CORPUS_LABEL,
    ),
    "sensational_terms": CueEvidence(
        fake_rate=6.44, real_rate=2.07, precision=75.68, n_fired=2245,
        corpus=CORPUS_LABEL,
    ),
}

DETECTION_RATE = CueEvidence(
    fake_rate=14.12, real_rate=2.39, precision=85.52, n_fired=4358,
    corpus=CORPUS_LABEL,
)
DETECTION_RATE_POLITICS = CueEvidence(
    fake_rate=29.54, real_rate=1.71, precision=90.25, n_fired=1251,
    corpus=f"{CORPUS_LABEL} (política)",
)

DETECTION_ONE_IN = round(100 / DETECTION_RATE.fake_rate)


def coverage() -> IndicatorCoverage:
    """What the panel can see, for the reader, derived from the measured union above."""
    return IndicatorCoverage(
        detection_rate=DETECTION_RATE.fake_rate,
        false_alarm_rate=DETECTION_RATE.real_rate,
        one_in=DETECTION_ONE_IN,
        corpus=DETECTION_RATE.corpus,
    )


def evidence_for(kind: str) -> CueEvidence | None:
    """The measured evidence for a cue kind, or None if it has never been measured."""
    return CUE_EVIDENCE.get(kind)
