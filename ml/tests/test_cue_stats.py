"""The numbers the interface prints must be the numbers that were measured."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/app")

from app.explanation.cue_stats import (  # noqa: E402
    CORPUS,
    CUE_EVIDENCE,
    DETECTION_ONE_IN,
    DETECTION_RATE,
    DETECTION_RATE_POLITICS,
)

_CANDIDATES = [
    Path("/mlresults/cue_precision.csv"),
    Path(__file__).resolve().parents[1] / "results" / "cue_precision.csv",
]

TOL = 0.005


def _table() -> dict[tuple[str, str], dict]:
    """Cue-level rows only (blank `term`), keyed by (corpus, kind)."""
    for path in _CANDIDATES:
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as fh:
                return {
                    (r["corpus"], r["kind"]): r
                    for r in csv.DictReader(fh)
                    if not r["term"]
                }
    raise RuntimeError(  # pragma: no cover - configuration error, and it must be loud
        f"cue_precision.csv not found in any of {[str(p) for p in _CANDIDATES]}. This test exists "
        "to check the shipped numbers against it; skipping quietly would defeat the point."
    )


@pytest.mark.parametrize("kind", sorted(CUE_EVIDENCE))
def test_cue_evidence_matches_the_measurement(kind):
    row = _table()[(CORPUS, kind)]
    shipped = CUE_EVIDENCE[kind]
    assert shipped.fake_rate == pytest.approx(float(row["fake_rate"]), abs=TOL)
    assert shipped.real_rate == pytest.approx(float(row["real_rate"]), abs=TOL)
    assert shipped.precision == pytest.approx(float(row["precision"]), abs=TOL)
    assert shipped.n_fired == int(row["n_fired"])


def test_detection_rate_matches_the_measured_union():
    """The union row, not the sum of the per-cue rows."""
    row = _table()[(CORPUS, "any_cue")]
    assert DETECTION_RATE.fake_rate == pytest.approx(float(row["fake_rate"]), abs=TOL)
    assert DETECTION_RATE.real_rate == pytest.approx(float(row["real_rate"]), abs=TOL)
    assert DETECTION_RATE.n_fired == int(row["n_fired"])


def test_politics_detection_rate_matches():
    row = _table()[(f"{CORPUS} [politica]", "any_cue")]
    assert DETECTION_RATE_POLITICS.fake_rate == pytest.approx(float(row["fake_rate"]), abs=TOL)
    assert DETECTION_RATE_POLITICS.real_rate == pytest.approx(float(row["real_rate"]), abs=TOL)


def test_one_in_n_is_derived_from_the_union_not_typed_by_hand():
    """The ratio in the user-facing sentence, recomputed."""
    assert DETECTION_ONE_IN == round(100 / DETECTION_RATE.fake_rate)
    assert DETECTION_ONE_IN == 7


def test_every_shipped_cue_has_a_measurement():
    """No cue may reach a user without a measured fire rate behind it."""
    from app.explanation.cues import extract_cues

    firing = {"exclamations", "sensational_terms"}
    text = "URGENTE!! compartilhe agora!"
    assert {ind.kind for ind in extract_cues(text)} == firing
    assert firing <= set(CUE_EVIDENCE), "a cue can fire with no entry in CUE_EVIDENCE"
    for ind in extract_cues(text):
        assert ind.evidence is not None, f"{ind.kind} shipped without its measurement"
