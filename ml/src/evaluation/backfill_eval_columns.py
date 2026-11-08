"""Add derived columns to saved probability files without re-running inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_crosscorpus import stratified_sample  # noqa: E402

COLUMN_ORDER = ("label", "prob", "raw_prob", "category", "words")

SEED = 42


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/app/data")) / "processed"


def _eval_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results")) / "eval"


def load_corpus(name: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if name not in cache:
        path = _data_dir() / f"{name}.csv"
        if not path.exists():
            cache[name] = None
        else:
            df = pd.read_csv(path).dropna(subset=["text", "label"])
            df["label"] = df["label"].astype(int)
            cache[name] = df
    return cache[name]


def resolve_sample(saved: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame | None:
    """The corpus rows this file was scored on, or None if they cannot be identified."""
    for n in dict.fromkeys((len(saved), 5000, len(corpus))):
        candidate = stratified_sample(corpus, n, SEED)
        if len(candidate) != len(saved):
            continue
        if not (saved["label"].astype(int).to_numpy() == candidate["label"].to_numpy()).all():
            continue
        if "category" in saved.columns and "category" in candidate.columns:
            left = saved["category"].fillna("").astype(str).to_numpy()
            right = candidate["category"].fillna("").astype(str).to_numpy()
            if not (left == right).all():
                continue
        return candidate
    return None


def derive(sample: pd.DataFrame) -> dict[str, pd.Series]:
    """The model-independent columns."""
    out = {"words": sample["text"].astype(str).str.split().str.len()}
    if "category" in sample.columns:
        out["category"] = sample["category"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill derived columns into saved probabilities.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    parser.add_argument("--eval-dir", default=None, help="Override the directory of probability files.")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir) if args.eval_dir else _eval_dir()
    files = sorted(eval_dir.glob("*__*.csv"))
    if not files:
        print(f"[backfill] no probability files under {eval_dir}")
        return 1

    cache: dict[str, pd.DataFrame] = {}
    updated = unchanged = failed = 0

    for path in files:
        corpus_name = path.stem.split("__", 1)[0]
        corpus = load_corpus(corpus_name, cache)
        if corpus is None:
            print(f"  SKIP    {path.name}: corpus {corpus_name}.csv not found")
            failed += 1
            continue

        saved = pd.read_csv(path, dtype=str)
        sample = resolve_sample(saved, corpus)
        if sample is None:
            print(f"  FAILED  {path.name}: rows do not match {corpus_name}.csv - corpus changed?")
            failed += 1
            continue

        derived = derive(sample)
        adding = [c for c in derived if c not in saved.columns]

        for column, values in derived.items():
            if column in saved.columns:
                left = saved[column].fillna("").astype(str).to_numpy()
                right = values.reset_index(drop=True).fillna("").astype(str).to_numpy()
                assert (left == right).all(), (
                    f"{path.name}: existing {column!r} disagrees with the re-derived value. The "
                    f"backfill and the evaluator must agree exactly; investigate before writing."
                )
            else:
                saved[column] = values.reset_index(drop=True).astype(str).to_numpy()

        if not adding:
            print(f"  ok      {path.name}: already complete, {len(saved)} rows re-verified")
            unchanged += 1
            continue

        ordered = [c for c in COLUMN_ORDER if c in saved.columns]
        ordered += [c for c in saved.columns if c not in ordered]
        saved = saved[ordered]

        if args.dry_run:
            print(f"  would   {path.name}: add {', '.join(adding)} ({len(saved)} rows)")
        else:
            saved.to_csv(path, index=False)
            print(f"  wrote   {path.name}: added {', '.join(adding)} ({len(saved)} rows)")
        updated += 1

    verb = "would update" if args.dry_run else "updated"
    print(f"\n[backfill] {verb} {updated}, already complete {unchanged}, unresolved {failed}")
    if failed:
        print("[backfill] unresolved files were left untouched - do not paper over this.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
