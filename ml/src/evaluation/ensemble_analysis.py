"""Ensemble analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_analysis import CORPORA, HIGH_BAND, LOW_BAND, ece, load  # noqa: E402
from eval_crosscorpus import best_f1_threshold  # noqa: E402

ENSEMBLE_MODELS = ("tfidf", "llm")
ALL_MODELS = ("tfidf", "bertugues", "bertimbau", "llm")

PARETO_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

TRIVIAL_F1 = 2 / 3


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def model_score(model: str, corpus: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (labels, usable score) for one model on one corpus, or None if not saved."""
    d = load(model, corpus)
    if d is None:
        return None
    col = "raw_prob" if "raw_prob" in d.columns else "prob"
    return d["label"].to_numpy(), d[col].to_numpy()


def percentile_map(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores onto their percentile within a FIXED reference distribution."""
    ref = np.sort(np.asarray(reference))
    return np.searchsorted(ref, scores, side="left") / len(ref)


def combine(scores: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.ones(len(scores)) / len(scores) if weights is None else np.asarray(weights, dtype=float)
    return np.average(np.vstack(scores), axis=0, weights=w)


def _row(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    _, _, _, f1 = best_f1_threshold(y, s)
    fake, real = s[y == 1], s[y == 0]
    return {
        "method": name,
        "ROC-AUC": roc_auc_score(y, s),
        "PR-AUC": average_precision_score(y, s),
        "best F1": f1,
        "vs trivial": f1 - TRIVIAL_F1,
        "ECE": ece(y, s),
        "fakes reassured": float((fake < LOW_BAND).mean()),
        "reals flagged": float((real > HIGH_BAND).mean()),
    }


def _fmt(df: pd.DataFrame) -> str:
    return df.set_index("method").to_string(float_format=lambda v: f"{v:.4f}")


def analysis_1_agreement() -> None:
    print("\n" + "=" * 78)
    print("1. Do the models actually disagree? (Spearman rank correlation)")
    print("=" * 78)
    rows = []
    for corpus in CORPORA:
        row: dict = {"corpus": corpus}
        for i, a in enumerate(ALL_MODELS):
            for b in ALL_MODELS[i + 1 :]:
                sa, sb = model_score(a, corpus), model_score(b, corpus)
                if sa is None or sb is None:
                    continue
                row[f"{a[:6]}/{b[:6]}"] = float(spearmanr(sa[1], sb[1]).statistic)
        rows.append(row)
    if len(rows[0]) == 1:
        print("  no saved probabilities found - run eval_crosscorpus.py first")
        return
    print(pd.DataFrame(rows).set_index("corpus").to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  Correlation near 1.0 would mean the models rank documents the same way, so combining")
    print("  them could only average out noise. Values well below 1.0 mean they are keying on")
    print("  genuinely different signal, which is the precondition for an ensemble to help.")


def analysis_2_methods() -> None:
    print("\n" + "=" * 78)
    print("2. Combination methods (reference distribution taken from the SAME corpus)")
    print("=" * 78)
    for corpus in CORPORA:
        parts = {m: model_score(m, corpus) for m in ALL_MODELS}
        if any(v is None for v in parts.values()):
            continue
        y = parts["tfidf"][0]
        t, g, m = parts["tfidf"][1], parts[ENSEMBLE_MODELS[1]][1], parts["bertugues"][1]

        rank = {k: rankdata(v) / len(v) for k, v in (("t", t), ("g", g), ("m", m))}
        pct = {k: percentile_map(v, v) for k, v in (("t", t), ("g", g), ("m", m))}

        rows = [
            _row("TF-IDF alone", y, t),
            _row(f"{ENSEMBLE_MODELS[1]} alone", y, g),
            _row("rank-average [offline only]", y, combine([rank["t"], rank["g"]])),
            _row("probability-average", y, combine([t, g])),
            _row("percentile-map [deployable]", y, combine([pct["t"], pct["g"]])),
            _row("+ BERTugues (3-way rank)", y, combine([rank["t"], rank["g"], rank["m"]])),
        ]
        print(f"\n--- {corpus} (n={len(y)}) ---")
        print(_fmt(pd.DataFrame(rows)))

    print("\n  rank-average needs the whole test set in hand to compute ranks, so it cannot serve a")
    print("  single request - it is an offline upper bound. percentile-map approximates it against a")
    print("  fixed reference and IS deployable; analysis 3 tests whether it survives transfer.")
    print("  probability-average mixes two incommensurable scales (a calibrated probability and a")
    print("  raw logit-ish score), which is why it is reported but not recommended.")
    print("\n  These figures are OPTIMISTIC: the reference distribution is the test corpus itself.")


def analysis_3_transfer() -> None:
    print("\n" + "=" * 78)
    print("3. Leave-one-corpus-out (the honest case: reference fitted on the OTHER corpora)")
    print("=" * 78)
    for held_out in CORPORA:
        others = [c for c in CORPORA if c != held_out]
        ref: dict[str, np.ndarray] = {}
        ok = True
        for model in ENSEMBLE_MODELS:
            parts = [model_score(model, c) for c in others]
            if any(p is None for p in parts):
                ok = False
                break
            ref[model] = np.concatenate([p[1] for p in parts])
        held = {m: model_score(m, held_out) for m in ENSEMBLE_MODELS}
        if not ok or any(v is None for v in held.values()):
            continue

        y = held["tfidf"][0]
        mapped = [percentile_map(held[m][1], ref[m]) for m in ENSEMBLE_MODELS]
        rows = [
            _row("TF-IDF alone", y, held["tfidf"][1]),
            _row(f"{ENSEMBLE_MODELS[1]} alone", y, held[ENSEMBLE_MODELS[1]][1]),
            _row("percentile-map ensemble", y, combine(mapped)),
        ]
        print(f"\n--- held out: {held_out} (n={len(y)}; reference from {', '.join(others)}) ---")
        print(_fmt(pd.DataFrame(rows)))

    print("\n  The reference distributions never saw the held-out corpus, so any gain here is a")
    print("  transfer result rather than an in-sample fit. Read the ECE column beside the ROC-AUC:")
    print("  a ranking gain bought with a calibration loss is not free, and the platform displays a")
    print("  probability, not a rank.")


def analysis_4_pareto() -> None:
    print("\n" + "=" * 78)
    print("4. Weight sweep: what each point of ranking gain costs in calibration")
    print("=" * 78)

    records = []
    for w in PARETO_WEIGHTS:
        per_corpus = []
        for held_out in CORPORA:
            others = [c for c in CORPORA if c != held_out]
            ref, held, ok = {}, {}, True
            for model in ENSEMBLE_MODELS:
                parts = [model_score(model, c) for c in others]
                h = model_score(model, held_out)
                if any(p is None for p in parts) or h is None:
                    ok = False
                    break
                ref[model] = np.concatenate([p[1] for p in parts])
                held[model] = h
            if not ok:
                continue
            y = held["tfidf"][0]
            score = (1 - w) * held["tfidf"][1] + w * percentile_map(
                held[ENSEMBLE_MODELS[1]][1], ref[ENSEMBLE_MODELS[1]]
            )
            _, _, _, f1 = best_f1_threshold(y, score)
            fake, real = score[y == 1], score[y == 0]
            per_corpus.append(
                {
                    "ROC-AUC": roc_auc_score(y, score),
                    "ECE": ece(y, score),
                    "best F1": f1,
                    "fakes reassured": float((fake < LOW_BAND).mean()),
                    "reals flagged": float((real > HIGH_BAND).mean()),
                }
            )
        if per_corpus:
            mean = {k: float(np.mean([r[k] for r in per_corpus])) for k in per_corpus[0]}
            records.append({"w_second": w, **mean})

    if not records:
        print("  no saved probabilities found")
        return

    df = pd.DataFrame(records)
    print()
    print(df.set_index("w_second").to_string(float_format=lambda v: f"{v:.4f}"))

    out = _results_dir() / "ensemble_pareto.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}")

    base, best = records[0], max(records, key=lambda r: r["ROC-AUC"])
    print(f"\n  w=0.0 is the currently served model: ROC-AUC {base['ROC-AUC']:.4f}, "
          f"ECE {base['ECE']:.4f}, harms {base['fakes reassured']:.1%} / "
          f"{base['reals flagged']:.1%}")
    print(f"  w={best['w_second']:.2f} maximises ranking: ROC-AUC {best['ROC-AUC']:.4f} "
          f"({best['ROC-AUC'] - base['ROC-AUC']:+.4f}), ECE {best['ECE']:.4f} "
          f"({best['ECE'] - base['ECE']:+.4f})")

    dominating = [
        r for r in records[1:]
        if r["fakes reassured"] <= base["fakes reassured"]
        and r["reals flagged"] <= base["reals flagged"]
        and r["ROC-AUC"] > base["ROC-AUC"]
    ]
    print("\n  Weights that improve ranking AND leave BOTH user-facing harms no worse than w=0:")
    if dominating:
        for r in dominating:
            print(f"    w={r['w_second']:.2f}  ROC-AUC {r['ROC-AUC']:.4f} "
                  f"({r['ROC-AUC'] - base['ROC-AUC']:+.4f})  ECE {r['ECE']:.4f} "
                  f"({r['ECE'] - base['ECE']:+.4f})  harms {r['fakes reassured']:.1%} / "
                  f"{r['reals flagged']:.1%}")
        print("\n  CAUTION: this is chosen over only three corpora, so treat the specific weight as")
        print("  indicative rather than tuned. Reporting the whole curve is the defensible move;")
        print("  quoting a single optimum selected on the same data that measured it is not.")
    else:
        print("    none - every weight that improves ranking worsens at least one harm.")

    print("\n  ECE rising monotonically with w means there is no free lunch on the honesty of the")
    print("  number, so the choice of w is an explicit trade rather than an optimisation. That is")
    print("  the same shape as the SVM (best F1, no predict_proba) and BERTugues (equal ranking,")
    print("  non-transferring probabilities): the best model by the headline metric is not the")
    print("  model that can ship.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ensemble analysis from saved probabilities.")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4], help="Run a single analysis.")
    args = parser.parse_args(argv)

    steps = {
        1: analysis_1_agreement,
        2: analysis_2_methods,
        3: analysis_3_transfer,
        4: analysis_4_pareto,
    }
    for n in [args.only] if args.only else [1, 2, 3, 4]:
        steps[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
