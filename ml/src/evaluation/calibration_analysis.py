"""Calibration analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

CORPORA = ["fakerecogna1", "fakerecogna2", "fakerecogna2_abs"]
MODELS = ["tfidf", "bertugues", "bertimbau", "llm"]
LOW_BAND, HIGH_BAND = 0.4, 0.6
SEED = 42


def _eval_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results")) / "eval"


def load(model: str, corpus: str) -> pd.DataFrame | None:
    path = _eval_dir() / f"{corpus}__{model}.csv"
    return pd.read_csv(path) if path.exists() else None


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(total)


def summarise(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    fake, real = p[y == 1], p[y == 0]
    return {
        "name": name,
        "AUC": roc_auc_score(y, p),
        "ECE": ece(y, p),
        "Brier": brier_score_loss(y, p),
        "median": float(np.median(p)),
        "p5-p95": f"{np.percentile(p, 5):.2f}-{np.percentile(p, 95):.2f}",
        "fakes reassured": float((fake < LOW_BAND).mean()),
        "reals flagged": float((real > HIGH_BAND).mean()),
    }


def fit_platt(raw: np.ndarray, y: np.ndarray):
    """Platt scaling."""
    lr = LogisticRegression(C=1e10).fit(raw.reshape(-1, 1), y)
    return lambda x: lr.predict_proba(x.reshape(-1, 1))[:, 1]


def analysis_1_quality() -> None:
    print("\n" + "=" * 78)
    print("1. Calibration quality of each served model (as currently calibrated)")
    print("=" * 78)
    rows = []
    for corpus in CORPORA:
        for model in MODELS:
            d = load(model, corpus)
            if d is None:
                continue
            r = summarise(model, d["label"].to_numpy(), d["prob"].to_numpy())
            rows.append({"corpus": corpus, **r})
    if not rows:
        print("  no saved probabilities found - run eval_crosscorpus.py first")
        return
    df = pd.DataFrame(rows).rename(columns={"name": "model"}).set_index(["corpus", "model"])
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  ECE near 0 = the displayed percentage can be believed. A well-ranking but badly")
    print("  calibrated model shows a high AUC beside a high ECE.")


def analysis_2_calibrators(model: str = "bertugues") -> None:
    print("\n" + "=" * 78)
    print(f"2. Calibrator comparison for {model} (fit/test on disjoint halves of the SAME corpus)")
    print("=" * 78)
    for corpus in CORPORA:
        d = load(model, corpus)
        if d is None or "raw_prob" not in d.columns:
            continue
        y, raw, cur = d["label"].to_numpy(), d["raw_prob"].to_numpy(), d["prob"].to_numpy()
        i_fit, i_test = train_test_split(
            np.arange(len(y)), test_size=0.5, stratify=y, random_state=SEED
        )
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw[i_fit], y[i_fit])
        platt = fit_platt(raw[i_fit], y[i_fit])
        rows = [
            summarise("current (Fake.Br isotonic)", y[i_test], cur[i_test]),
            summarise("isotonic refit", y[i_test], iso.predict(raw[i_test])),
            summarise("Platt refit", y[i_test], platt(raw[i_test])),
            summarise("raw, uncalibrated", y[i_test], raw[i_test]),
        ]
        print(f"\n--- {corpus} (held-out half, n={len(i_test)}) ---")
        print(pd.DataFrame(rows).set_index("name").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  This is the OPTIMISTIC case: the calibrator has seen this corpus. See analysis 3.")


def analysis_3_transfer(model: str = "bertugues") -> None:
    print("\n" + "=" * 78)
    print(f"3. Leave-one-corpus-out transfer for {model} (the honest case)")
    print("=" * 78)
    for held_out in CORPORA:
        fit_sets = [c for c in CORPORA if c != held_out]
        frames = [load(model, c) for c in fit_sets]
        if any(f is None or "raw_prob" not in f.columns for f in frames):
            continue
        y_fit = np.concatenate([f["label"].to_numpy() for f in frames])
        raw_fit = np.concatenate([f["raw_prob"].to_numpy() for f in frames])
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_fit, y_fit)
        platt = fit_platt(raw_fit, y_fit)

        d = load(model, held_out)
        y, raw, cur = d["label"].to_numpy(), d["raw_prob"].to_numpy(), d["prob"].to_numpy()
        rows = [
            summarise("current (Fake.Br isotonic)", y, cur),
            summarise("isotonic on the other two", y, iso.predict(raw)),
            summarise("Platt on the other two", y, platt(raw)),
        ]
        print(f"\n--- held out: {held_out} (n={len(y)}) ---")
        print(pd.DataFrame(rows).set_index("name").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  If no row here gives a usable spread AND low ECE, then no FIXED calibrator solves it:")
    print("  the raw-score-to-probability relationship is itself corpus-dependent.")


def analysis_4_sweep(model: str = "tfidf") -> None:
    print("\n" + "=" * 78)
    print(f"4. Threshold and UI band sweep for {model}, averaged over {len(CORPORA)} corpora")
    print("=" * 78)
    loaded = [(c, load(model, c)) for c in CORPORA]
    loaded = [(c, d) for c, d in loaded if d is not None]
    if not loaded:
        print("  no saved probabilities found")
        return

    print(f"\n{'thr':>5} {'recall (fakes caught)':>22} {'FP rate (reals flagged)':>24} "
          f"{'precision':>10} {'F1':>7}")
    for t in np.arange(0.30, 0.75, 0.05):
        rec, fpr, pre, f1s = [], [], [], []
        for _, d in loaded:
            y, p = d["label"].to_numpy(), d["prob"].to_numpy()
            pred = (p >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            r = tp / max(1, tp + fn)
            pr = tp / max(1, tp + fp)
            rec.append(r)
            pre.append(pr)
            fpr.append(fp / max(1, int((y == 0).sum())))
            f1s.append(0.0 if pr + r == 0 else 2 * pr * r / (pr + r))
        print(f"{t:>5.2f} {np.mean(rec):>21.1%} {np.mean(fpr):>23.1%} "
              f"{np.mean(pre):>9.1%} {np.mean(f1s):>7.3f}")

    print(f"\n{'low':>5} {'high':>5} | {'fakes reassured':>16} {'fakes flagged':>14} | "
          f"{'reals reassured':>16} {'reals flagged':>14}")
    for lo, hi in [(0.30, 0.50), (0.35, 0.55), (0.40, 0.60), (0.45, 0.65), (0.50, 0.70)]:
        a, b, c_, d_ = [], [], [], []
        for _, d in loaded:
            y, p = d["label"].to_numpy(), d["prob"].to_numpy()
            fake, real = p[y == 1], p[y == 0]
            a.append((fake < lo).mean())
            b.append((fake > hi).mean())
            c_.append((real < lo).mean())
            d_.append((real > hi).mean())
        marker = "  <- current" if (lo, hi) == (LOW_BAND, HIGH_BAND) else ""
        print(f"{lo:>5.2f} {hi:>5.2f} | {np.mean(a):>15.1%} {np.mean(b):>13.1%} | "
              f"{np.mean(c_):>15.1%} {np.mean(d_):>13.1%}{marker}")

    print("\n  fakes reassured = misinformation the UI calls 'poucos sinais' (false-negative harm)")
    print("  reals flagged   = legitimate political speech accused (false-positive harm)")
    print("  These two harms are not interchangeable - see CLAUDE.md section 9.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibration analysis from saved probabilities.")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4], help="Run a single analysis.")
    parser.add_argument("--model", default=None, help="Override the model for analyses 2-4.")
    args = parser.parse_args(argv)

    steps = {
        1: analysis_1_quality,
        2: lambda: analysis_2_calibrators(args.model or "bertugues"),
        3: lambda: analysis_3_transfer(args.model or "bertugues"),
        4: lambda: analysis_4_sweep(args.model or "tfidf"),
    }
    for n in ([args.only] if args.only else [1, 2, 3, 4]):
        steps[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
