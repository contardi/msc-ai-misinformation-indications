"""Build a BLIND coding sheet for sensationalist framing, using a published instrument."""

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
SHEET_SEED = 20260910
EVAL_SAMPLE = 5000


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def build(corpus_csv: Path, eval_csv: Path, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    per_class = n // 2
    parts = []
    for label in (0, 1):
        pool = sample[sample["label"] == label]
        if len(pool) < per_class:
            raise SystemExit(f"[sheet] only {len(pool)} rows with label={label}, need {per_class}")
        parts.append(pool.sample(n=per_class, random_state=SHEET_SEED))
    chosen = pd.concat(parts)

    sheet = pd.DataFrame({
        "id": chosen["id"].to_numpy(),
        "title": chosen["title"].fillna("").to_numpy(),
        "text": [body(t) for t in chosen["text"].fillna("")],
        "exagero": "",
    }).sample(frac=1, random_state=SHEET_SEED).reset_index(drop=True)

    allowed = {"id", "title", "text", "exagero"}
    leaked = [c for c in sheet.columns if c not in allowed]
    if leaked:  # pragma: no cover - guard against a careless future edit
        raise SystemExit(f"[sheet] blinding violated: unexpected columns {leaked}")

    key = pd.DataFrame({
        "id": chosen["id"].to_numpy(),
        "label": chosen["label"].to_numpy(),
        "prob": chosen["prob"].to_numpy(),
        "category": chosen["category"].fillna("").to_numpy(),
        "words": chosen["text"].fillna("").str.split().str.len().to_numpy(),
    })
    return sheet, key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/fakerecogna2.csv")
    ap.add_argument("--corpus", default="fakerecogna2", help="name used in ml/results/eval filenames")
    ap.add_argument("--model", default="tfidf", help="the SERVED model; that is the one users meet")
    ap.add_argument("--sample", type=int, default=300,
                    help="items to code. 300 is about 1.5 to 2.5 hours with scripts/label_exagero.py")
    args = ap.parse_args(argv)

    results = _results_dir()
    eval_csv = results / "eval" / f"{args.corpus}__{args.model}.csv"
    if not eval_csv.exists():
        raise SystemExit(f"[sheet] no saved probabilities at {eval_csv}")

    sheet, key = build(Path(args.data), eval_csv, args.sample)

    sheet_path = results / "exagero_sheet.csv"
    key_path = results / "exagero_key.csv"
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)

    print(f"[sheet] {len(sheet)} items to code           -> {sheet_path}")
    print(f"[sheet] labels, probabilities held separately -> {key_path}")
    print("[sheet] code the 'exagero' column following documents/sensationalism-coding.md,")
    print("[sheet] then run: python ml/src/evaluation/eval_sensationalism.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
