"""Temporal generalisation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "training"))

from eval_crosscorpus import best_f1_threshold  # noqa: E402
from train_baseline import build_pipeline, load_config  # noqa: E402

TRIVIAL_F1 = 2 / 3


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def load_dated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    if "date" not in df.columns:
        raise SystemExit(
            f"{path} has no 'date' column. Regenerate it with the current prepare_faketruebr.py, "
            "which carries publication dates through."
        )
    df = df.dropna(subset=["date"]).copy()
    df["label"] = df["label"].astype(int)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def show_years(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["year", "label"]).size().unstack(fill_value=0).rename(columns={0: "true", 1: "fake"})
    )
    for col in ("true", "fake"):
        if col not in table.columns:
            table[col] = 0
    table = table[["true", "fake"]]
    table["total"] = table.sum(axis=1)
    print("\nRows per year (this is what the cutoff must be chosen from, not assumed):")
    print(table.to_string())
    single = table[(table["true"] == 0) | (table["fake"] == 0)]
    if len(single):
        rows = int(single["total"].sum())
        print(f"\n  WARNING: {len(single)} of {len(table)} years contain only ONE class "
              f"({rows} rows, {rows / len(df):.1%} of the corpus).")
        only_fake = single[single["true"] == 0]
        only_true = single[single["fake"] == 0]
        if len(only_fake):
            print(f"    fake-only years: {only_fake.index.min()}-{only_fake.index.max()} "
                  f"({int(only_fake['total'].sum())} rows)")
        if len(only_true):
            print(f"    true-only years: {only_true.index.min()}-{only_true.index.max()} "
                  f"({int(only_true['total'].sum())} rows)")

    for cls in ("true", "fake"):
        total = int(table[cls].sum())
        if not total:
            continue
        top = table[cls].sort_values(ascending=False).head(2)
        print(f"  {cls:<5} articles: {top.sum() / total:.1%} fall in {sorted(top.index.tolist())}")
    print("\n  A cutoff is only usable where BOTH classes exist on BOTH sides in workable numbers.")
    return table


def _fit_score(train: pd.DataFrame, test: pd.DataFrame, seed: int, cfg: dict) -> dict:
    pipe = build_pipeline(seed, "logreg", cfg)
    pipe.fit(train["text"].astype(str).tolist(), train["label"].to_numpy())
    y = test["label"].to_numpy()
    idx = list(pipe.named_steps["clf"].classes_).index(1)
    prob = pipe.predict_proba(test["text"].astype(str).tolist())[:, idx]
    pred = (prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    auc = float(roc_auc_score(y, prob)) if len(set(y.tolist())) > 1 else float("nan")
    _, _, _, best_f1 = best_f1_threshold(y, prob)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "best_f1": float(best_f1),
    }


def _matched_random(df: pd.DataFrame, train_counts: dict, test_counts: dict, seed: int):
    """A random split with the SAME per-class train/test sizes as the chronological one."""
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for label, group in df.groupby("label"):
        need_train, need_test = train_counts.get(label, 0), test_counts.get(label, 0)
        if need_train + need_test > len(group):
            return None, None
        picked = rng.permutation(group.index.to_numpy())
        train_parts.append(df.loc[picked[:need_train]])
        test_parts.append(df.loc[picked[need_train:need_train + need_test]])
    return pd.concat(train_parts), pd.concat(test_parts)


def run_cutoff(df: pd.DataFrame, cutoff: int, seed: int, repeats: int, cfg: dict) -> list[dict]:
    train = df[df["year"] <= cutoff]
    test = df[df["year"] > cutoff]
    if train.empty or test.empty:
        print(f"\n--- cutoff {cutoff}: skipped (one side empty) ---")
        return []

    tr_counts = train["label"].value_counts().to_dict()
    te_counts = test["label"].value_counts().to_dict()
    print(f"\n{'=' * 78}\ncutoff {cutoff}: train <= {cutoff}, test >= {cutoff + 1}\n{'=' * 78}")
    print(f"  chronological train n={len(train):>6} (fake={tr_counts.get(1, 0)}, "
          f"true={tr_counts.get(0, 0)})  years {train['year'].min()}-{train['year'].max()}")
    print(f"  chronological test  n={len(test):>6} (fake={te_counts.get(1, 0)}, "
          f"true={te_counts.get(0, 0)})  years {test['year'].min()}-{test['year'].max()}")

    rows = []
    chrono = _fit_score(train, test, seed, cfg)
    rows.append({"cutoff": cutoff, "arm": "chronological", "repeat": 0, **chrono})

    randoms = []
    for r in range(repeats):
        rtr, rte = _matched_random(df, tr_counts, te_counts, seed + r)
        if rtr is None:
            print("  random control: cannot match those class counts from the pool; skipped")
            break
        m = _fit_score(rtr, rte, seed, cfg)
        randoms.append(m)
        rows.append({"cutoff": cutoff, "arm": "random", "repeat": r, **m})

    print(f"\n  {'metric':<12} {'chronological':>14} {'random control':>16} {'drift effect':>14}")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "best_f1"):
        if not randoms:
            print(f"  {metric:<12} {chrono[metric]:>14.4f} {'n/a':>16} {'n/a':>14}")
            continue
        vals = [m[metric] for m in randoms]
        mean = float(np.nanmean(vals))
        print(f"  {metric:<12} {chrono[metric]:>14.4f} {mean:>16.4f} "
              f"{chrono[metric] - mean:>+14.4f}")
    if randoms:
        spread = float(np.nanstd([m["f1"] for m in randoms], ddof=1)) if len(randoms) > 1 else 0.0
        print(f"\n  random-arm F1 SD over {len(randoms)} draws: {spread:.4f}  "
              "(a drift effect smaller than this is noise)")
    print(f"  trivial all-positive F1 floor: {TRIVIAL_F1:.4f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chronological vs random split, with a matched control.")
    p.add_argument("--data", default=None, help="CSV with text,label,date (default: fakerecogna2).")
    p.add_argument("--cutoffs", default="", help="Comma-separated cutoff years (default: auto).")
    p.add_argument("--repeats", type=int, default=5, help="Random-control draws per cutoff.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--list-years", action="store_true", help="Print the year distribution and exit.")
    args = p.parse_args(argv)

    path = Path(args.data) if args.data else _data_dir() / "processed" / "fakerecogna2.csv"
    df = load_dated(path)
    print(f"[temporal] {path.name}: {len(df)} dated rows "
          f"(fake={int(df['label'].sum())}, true={int((df['label'] == 0).sum())})")
    table = show_years(df)
    if args.list_years:
        return 0

    if args.cutoffs:
        cutoffs = [int(c.strip()) for c in args.cutoffs.split(",") if c.strip()]
    else:
        cutoffs = []
        for year in sorted(table.index):
            tr, te = df[df["year"] <= year], df[df["year"] > year]
            if (len(tr) >= 500 and len(te) >= 500
                    and tr["label"].nunique() == 2 and te["label"].nunique() == 2):
                cutoffs.append(int(year))
        print(f"\n[temporal] viable cutoffs (>=500 rows and both classes each side): {cutoffs}")

    cfg = load_config()
    rows = []
    for cutoff in cutoffs:
        rows.extend(run_cutoff(df, cutoff, args.seed, args.repeats, cfg))

    if rows:
        out = _results_dir() / "temporal_split.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n[temporal] wrote {out}")
        print("\n  Read the 'drift effect' column, not the chronological score alone. A negative")
        print("  effect means the model does worse on genuinely later news than on a random sample")
        print("  of the same size and class balance - which is drift, not difficulty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
