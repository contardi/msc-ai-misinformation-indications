#!/usr/bin/env python3
"""Rate the EXAGERO (exaggeration) intensity of 300 news openings, one keystroke each."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labeller import Scheme, assert_blind, run  # noqa: E402

EXAGERO = Scheme(
    name="exagero",
    sheet=Path(os.environ.get("EXAGERO_SHEET", "ml/results/exagero_sheet.csv")),
    label_field="exagero",
    display=("title", "text"),
    keys={"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
    hints={
        "0": "nada    - plain reporting, no inflation",
        "1": "pouco   - a trace: one loaded adjective, mild puffery",
        "2": "médio   - clearly inflated, but the story is still recognisable",
        "3": "muito   - framing dominates: outrage, shock or scandal is the point",
        "4": "todo    - pure exaggeration, nothing left of sober reporting",
    },
    doc="documents/sensationalism-coding.md",
    regenerate=(
        "docker compose --profile tools run --rm analysis "
        "python ml_src/evaluation/make_sensationalism_sheet.py"
    ),
    next_step=(
        "docker compose --profile tools run --rm analysis "
        "python ml_src/evaluation/eval_sensationalism.py"
    ),
    forbidden=(("exagero", "_key"),),
)

assert_blind(EXAGERO, Path(__file__), Path(__file__).with_name("labeller.py"))


if __name__ == "__main__":
    raise SystemExit(run(EXAGERO))
