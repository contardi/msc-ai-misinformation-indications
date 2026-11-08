"""Build a BLIND coding sheet for political-direction labelling of false positives."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_crosscorpus import stratified_sample  # noqa: E402
from sheet_common import assert_alignment, body  # noqa: E402

SEED = 42
SHEET_SEED = 20260827
EVAL_SAMPLE = 5000
POLITICS = "política"


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def build(corpus_csv: Path, eval_csv: Path, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (blind sheet, private key)."""
    df = pd.read_csv(corpus_csv)
    for column in ("text", "label", "title", "category"):
        if column not in df.columns:
            raise SystemExit(f"[sheet] {corpus_csv} has no '{column}' column")

    sample = stratified_sample(df, EVAL_SAMPLE, SEED)
    saved = pd.read_csv(eval_csv)
    assert_alignment(sample, saved)

    sample = sample.reset_index(drop=True)
    sample["prob"] = saved["prob"].to_numpy()
    sample["id"] = sample.index

    pool = sample[(sample["category"] == POLITICS) & (sample["label"] == 0)]
    flagged = int((pool["prob"] >= 0.5).sum())
    print(
        f"[sheet] real '{POLITICS}' articles: {len(pool)} | flagged at 0.5: {flagged} "
        f"({flagged / len(pool):.1%})"
    )
    if len(pool) < n:
        print(f"[sheet] pool smaller than requested sample; coding all {len(pool)}")
        n = len(pool)

    chosen = pool.sample(n=n, random_state=SHEET_SEED)

    sheet = pd.DataFrame({
        "id": chosen["id"].to_numpy(),
        "title": chosen["title"].fillna("").to_numpy(),
        "text": [body(t) for t in chosen["text"].fillna("")],
        "direction": "",
    }).sample(frac=1, random_state=SHEET_SEED).reset_index(drop=True)

    leaked = [c for c in sheet.columns if c not in ("id", "title", "text", "direction")]
    if leaked:  # pragma: no cover - guard against a careless future edit
        raise SystemExit(f"[sheet] blinding violated: unexpected columns {leaked}")

    key = pd.DataFrame({
        "id": chosen["id"].to_numpy(),
        "prob": chosen["prob"].to_numpy(),
        "words": chosen["text"].fillna("").str.split().str.len().to_numpy(),
    })
    return sheet, key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/fakerecogna2.csv")
    ap.add_argument("--corpus", default="fakerecogna2", help="name used in ml/results/eval filenames")
    ap.add_argument("--model", default="tfidf", help="the SERVED model; that is the one users meet")
    ap.add_argument("--sample", type=int, default=400,
                    help="headlines to code. 400 gives roughly 150 per side after 'neutro' and "
                         "'indefinido' are set aside, which detects a ~15-point difference at 80%% power")
    args = ap.parse_args(argv)

    results = _results_dir()
    eval_csv = results / "eval" / f"{args.corpus}__{args.model}.csv"
    if not eval_csv.exists():
        raise SystemExit(f"[sheet] no saved probabilities at {eval_csv}")

    sheet, key = build(Path(args.data), eval_csv, args.sample)

    sheet_path = results / "direction_sheet.csv"
    key_path = results / "direction_key.csv"
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)

    print(f"[sheet] {len(sheet)} headlines to code -> {sheet_path}")
    print(f"[sheet] probabilities held separately   -> {key_path}")
    print("[sheet] fill the 'direction' column following documents/political-direction-coding.md,")
    print("[sheet] then run: python ml/src/evaluation/eval_direction_bias.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
