"""Does the model beat counting words?"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CORPORA = ("fakebr", "fakerecogna1", "fakerecogna2", "fakerecogna2_abs")

BANDS: tuple[tuple[int, int] | None, ...] = (None, (20, 200), (30, 150), (40, 120), (45, 90))

SEEDS = 5


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data")) / "processed"


def _results() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def length_auc(words: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of `shorter means fake`."""
    return float(roc_auc_score(labels, -words))


def matched_size_range(
    words: np.ndarray, labels: np.ndarray, n: int, seeds: int = SEEDS
) -> tuple[float, float, float]:
    """Mean and range of the length AUC over `seeds` stratified resamples of size `n`."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    per_class = min(n // 2, len(pos), len(neg))
    aucs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            rng.choice(pos, per_class, replace=False),
            rng.choice(neg, per_class, replace=False),
        ])
        aucs.append(length_auc(words[idx], labels[idx]))
    return float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))


def model_aucs(corpus: str) -> dict[str, tuple[float, int]]:
    """ROC-AUC of every saved model arm for this corpus, read from ml/results/eval/."""
    out: dict[str, tuple[float, int]] = {}
    eval_dir = _results() / "eval"
    for path in sorted(eval_dir.glob(f"{corpus}__*.csv")):
        arm = path.stem.split("__", 1)[1]
        df = pd.read_csv(path).dropna(subset=["label", "prob"])
        if df["label"].nunique() < 2:
            continue
        out[arm] = (float(roc_auc_score(df["label"], df["prob"])), len(df))
    return out


def prob_length_correlation(corpus: str) -> dict[str, float]:
    """Spearman correlation between each model's probability and the row's word count."""
    out: dict[str, float] = {}
    for path in sorted((_results() / "eval").glob(f"{corpus}__*.csv")):
        df = pd.read_csv(path)
        if "words" not in df.columns:
            continue
        out[path.stem.split("__", 1)[1]] = float(df["prob"].corr(df["words"], method="spearman"))
    return out


def model_auc_by_band(corpus: str) -> dict[str, list[tuple[str, int, float, float]]]:
    """Each model's ROC-AUC *within* word-count bands, beside the length baseline for that band."""
    out: dict[str, list[tuple[str, int, float, float]]] = {}
    for path in sorted((_results() / "eval").glob(f"{corpus}__*.csv")):
        df = pd.read_csv(path)
        if "words" not in df.columns:
            continue
        arm, bands = path.stem.split("__", 1)[1], []
        for band in BANDS:
            sub = df if band is None else df[(df["words"] >= band[0]) & (df["words"] <= band[1])]
            if sub["label"].nunique() < 2 or len(sub) < 100:
                continue
            name = "all" if band is None else f"{band[0]}-{band[1]}"
            bands.append((
                name,
                len(sub),
                float(roc_auc_score(sub["label"], sub["prob"])),
                length_auc(sub["words"].to_numpy(), sub["label"].to_numpy()),
            ))
        if bands:
            out[arm] = bands
    return out


def describe_lengths(words: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    stats = {}
    for value, name in ((0, "real"), (1, "fake")):
        w = np.sort(words[labels == value])
        if not len(w):
            continue
        q25, q75 = float(np.percentile(w, 25)), float(np.percentile(w, 75))
        stats[name] = {
            "n": len(w),
            "p5": float(np.percentile(w, 5)),
            "p25": q25,
            "median": float(np.percentile(w, 50)),
            "p75": q75,
            "p95": float(np.percentile(w, 95)),
            "iqr": q75 - q25,
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Length-only control arm for cross-corpus results.")
    parser.add_argument("--corpora", nargs="*", default=list(CORPORA))
    parser.add_argument("--n", type=int, default=5000, help="Matched sample size (models used 5000).")
    parser.add_argument("--out", default="length_baseline.csv", help="Written under RESULTS_DIR.")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for corpus in args.corpora:
        path = _data_dir() / f"{corpus}.csv"
        if not path.exists():
            print(f"\n=== {corpus} === (skipped: {path} not found)")
            continue
        df = pd.read_csv(path).dropna(subset=["text", "label"])
        words = df["text"].astype(str).str.split().str.len().to_numpy()
        labels = df["label"].astype(int).to_numpy()

        full = length_auc(words, labels)
        mean, lo, hi = matched_size_range(words, labels, args.n)
        print(f"\n=== {corpus} ===  n={len(df)}")
        print(f"  length-only ROC-AUC, full corpus : {full:.4f}")
        print(f"  length-only ROC-AUC, n={args.n:<6}   : {mean:.4f}  (5 seeds, {lo:.4f}-{hi:.4f})")
        rows.append({"corpus": corpus, "arm": "length", "n": len(df), "roc_auc": round(full, 4),
                     "seed_min": round(lo, 4), "seed_max": round(hi, 4),
                     "beats_length": "", "note": "len(text.split()), shorter=fake"})

        models = model_aucs(corpus)
        if models:
            print(f"  {'arm':<22} {'ROC-AUC':>8} {'vs length':>10}")
            for arm, (auc, n) in sorted(models.items(), key=lambda kv: -kv[1][0]):
                inside = lo <= auc <= hi
                verdict = "within noise" if inside else ("above" if auc > hi else "BELOW")
                print(f"  {arm:<22} {auc:8.4f} {auc - mean:+10.4f}  {verdict}")
                rows.append({"corpus": corpus, "arm": arm, "n": n, "roc_auc": round(auc, 4),
                             "seed_min": "", "seed_max": "",
                             "beats_length": verdict, "note": ""})

        corr = prob_length_correlation(corpus)
        if corr:
            print("  probability-vs-length Spearman (does the MODEL use length?):")
            for arm, rho in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
                print(f"    {arm:<20} rho={rho:+.4f}")
                rows.append({"corpus": corpus, "arm": f"{arm} (rho vs words)", "n": "",
                             "roc_auc": round(rho, 4), "seed_min": "", "seed_max": "",
                             "beats_length": "", "note": "spearman(prob, words)"})
        else:
            print("  probability-vs-length: no `words` column yet - regenerate ml/results/eval/")

        for arm, bands in model_auc_by_band(corpus).items():
            print(f"  {arm} WITHIN length bands (confound held roughly constant):")
            print(f"    {'band':>12} {'n':>7} {'model':>8} {'length':>8} {'model-length':>13}")
            for name, n, m_auc, l_auc in bands:
                print(f"    {name:>12} {n:7d} {m_auc:8.4f} {l_auc:8.4f} {m_auc - l_auc:+13.4f}")
                rows.append({"corpus": corpus, "arm": f"{arm} [{name} words]", "n": n,
                             "roc_auc": round(m_auc, 4), "seed_min": "", "seed_max": "",
                             "beats_length": round(m_auc - l_auc, 4),
                             "note": "within-band; length control held"})

        print("  tail check (is the length signal only in the extremes?):")
        for band in BANDS:
            mask = np.ones(len(words), bool) if band is None else (words >= band[0]) & (words <= band[1])
            sub_y = labels[mask]
            if sub_y.min() == sub_y.max():
                continue
            auc = length_auc(words[mask], sub_y)
            name = "all" if band is None else f"{band[0]}-{band[1]}"
            print(f"    {name:>10} words  n={mask.sum():6d}  fake={sub_y.mean() * 100:5.1f}%  AUC={auc:.4f}")
            rows.append({"corpus": corpus, "arm": f"length [{name} words]", "n": int(mask.sum()),
                         "roc_auc": round(auc, 4), "seed_min": "", "seed_max": "",
                         "beats_length": "", "note": "tail-effect check"})

        print("  length distribution (a manufactured IQR is the summarisation fingerprint):")
        for name, s in describe_lengths(words, labels).items():
            print(f"    {name:<5} n={s['n']:6.0f}  p5={s['p5']:5.0f} p25={s['p25']:5.0f} "
                  f"median={s['median']:5.0f} p75={s['p75']:5.0f} p95={s['p95']:6.0f}  IQR={s['iqr']:5.0f}")
            rows.append({"corpus": corpus, "arm": f"length distribution [{name}]", "n": int(s["n"]),
                         "roc_auc": "", "seed_min": s["p25"], "seed_max": s["p75"],
                         "beats_length": "", "note": f"median={s['median']:.0f} iqr={s['iqr']:.0f}"})

    missing = [c for c in CORPORA if c not in args.corpora]
    if missing and args.out == parser.get_default("out"):
        print(f"\nREFUSING TO WRITE: this run covers {len(args.corpora)} of {len(CORPORA)} corpora "
              f"(missing {', '.join(missing)}).")
        print(f"Writing to the default '{args.out}' would replace the full table with a partial one.")
        print("Re-run without --corpora, or pass --out to a scratch filename.")
        return 1

    if rows:
        out = _results() / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
