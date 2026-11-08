#!/usr/bin/env python3
"""Label the political direction of 400 headlines, one keystroke each."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labeller import Scheme, assert_blind, run  # noqa: E402

DIRECTION = Scheme(
    name="direction",
    sheet=Path(os.environ.get("DIRECTION_SHEET", "ml/results/direction_sheet.csv")),
    label_field="direction",
    display=("title", "text"),
    keys={
        "e": "esquerda",
        "d": "direita",
        "a": "ambos",
        "n": "neutro",
        "i": "indefinido",
    },
    hints={
        "esquerda": "PT, Lula, PSOL - a left-leaning actor is the SUBJECT",
        "direita": "Bolsonaro, PL, PP - a right-leaning actor is the SUBJECT",
        "ambos": "both sides, in genuine conflict",
        "neutro": "STF/TSE/OAB acting institutionally, or not political at all",
        "indefinido": "cannot tell from the headline",
    },
    doc="documents/political-direction-coding.md",
    regenerate=(
        "docker compose --profile tools run --rm analysis "
        "python ml_src/evaluation/make_direction_sheet.py"
    ),
    next_step=(
        "docker compose --profile tools run --rm analysis "
        "python ml_src/evaluation/eval_direction_bias.py"
    ),
    forbidden=(("direction", "_key"),),
)

assert_blind(DIRECTION, Path(__file__), Path(__file__).with_name("labeller.py"))


if __name__ == "__main__":
    raise SystemExit(run(DIRECTION))
