"""Which topics get flagged, and which get missed?"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_analysis import CORPORA, HIGH_BAND, LOW_BAND, load  # noqa: E402

THRESHOLD = 0.5


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, g in df.groupby("category"):
        fake, real = g[g["label"] == 1], g[g["label"] == 0]
        rows.append({
            "category": str(category),
            "n": len(g),
            "n_real": len(real),
            "n_fake": len(fake),
            "FP@0.5": float((real["prob"] >= THRESHOLD).mean()) if len(real) else np.nan,
            "reals_flagged>0.6": float((real["prob"] > HIGH_BAND).mean()) if len(real) else np.nan,
            "FN@0.5": float((fake["prob"] < THRESHOLD).mean()) if len(fake) else np.nan,
            "fakes_reassured<0.4": float((fake["prob"] < LOW_BAND).mean()) if len(fake) else np.nan,
            "mean_p_real": float(real["prob"].mean()) if len(real) else np.nan,
            "mean_p_fake": float(fake["prob"].mean()) if len(fake) else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-category error rates for the served model.")
    p.add_argument("--model", default="tfidf", help="Model whose saved probabilities to analyse.")
    p.add_argument("--min-n", type=int, default=30,
                   help="Hide categories with fewer rows than this from the ranking (still counted "
                        "in the reconciliation). Small groups produce unstable rates.")
    p.add_argument("--min-real", type=int, default=30,
                   help="Minimum REAL articles for a category to carry a false-positive rate. A "
                        "category with one real article can report 0%% or 100%% and mean neither, "
                        "which would otherwise anchor the spread comparison to noise.")
    args = p.parse_args(argv)

    frames = []
    for corpus in CORPORA:
        d = load(args.model, corpus)
        if d is None or "category" not in d.columns:
            continue
        d = d.copy()
        d["corpus"] = corpus
        frames.append(d)

    if not frames:
        print(f"No saved probabilities for '{args.model}' carry a 'category' column.\n"
              "Regenerate them with the current eval_crosscorpus.py, e.g.:\n"
              "  python ml/src/evaluation/eval_crosscorpus.py "
              "--data data/processed/fakerecogna2.csv --sample 5000 --models tfidf")
        return 1

    df = pd.concat(frames, ignore_index=True).dropna(subset=["category"])
    df["category"] = df["category"].astype(str).str.strip().str.casefold()
    print(f"[bias] {args.model}: {len(df)} rows across {df['corpus'].nunique()} corpora, "
          f"{df['category'].nunique()} categories after case-normalising")

    table = per_category(df)

    topical = table[(table["n_real"] >= args.min_real) & (table["n"] >= args.min_n)].sort_values(
        "FP@0.5", ascending=False
    )
    thin = table[(table["n_real"] > 0) & (table["n_real"] < args.min_real)
                 & (table["n"] >= args.min_n)]
    fake_only = table[(table["n_real"] == 0) & (table["n"] >= args.min_n)].sort_values(
        "FN@0.5", ascending=False
    )

    fmt = {c: "{:.3f}".format for c in table.columns if c.startswith(("FP", "FN", "mean", "reals", "fakes"))}
    print(f"\n{'=' * 104}")
    print(f"TOPICAL categories (both classes present, n >= {args.min_n}) - worst suppression harm first")
    print("=" * 104)
    print(topical.to_string(index=False, formatters=fmt, na_rep="-"))

    if len(thin):
        print(f"\n  Too few real articles for a meaningful false-positive rate "
              f"(n_real < {args.min_real}), excluded from the ranking: "
              + ", ".join(f"{r.category} (n_real={r.n_real})" for r in thin.itertuples()))

    if len(fake_only):
        print(f"\n{'-' * 104}")
        print("Fake-only labels - no real articles, so no false-positive rate. Mostly fact-checker")
        print("section names rather than topics; shown for false-negative rate only.")
        print("-" * 104)
        print(fake_only[["category", "n", "n_fake", "FN@0.5", "fakes_reassured<0.4", "mean_p_fake"]]
              .to_string(index=False, formatters=fmt, na_rep="-"))

    hidden = table[table["n"] < args.min_n]
    if len(hidden):
        print(f"\n  {len(hidden)} categories hidden (n < {args.min_n}): "
              f"{', '.join(hidden['category'].head(8))}"
              f"{' ...' if len(hidden) > 8 else ''}")
    shown = topical

    real, fake = df[df["label"] == 0], df[df["label"] == 1]
    overall_fp = float((real["prob"] >= THRESHOLD).mean())
    overall_fn = float((fake["prob"] < THRESHOLD).mean())
    print(f"\n  Corpus-level check: FP@0.5 {overall_fp:.3f}, FN@0.5 {overall_fn:.3f} "
          f"(n_real={len(real)}, n_fake={len(fake)})")
    print(f"  Band-level:         reals flagged >0.6 {float((real['prob'] > HIGH_BAND).mean()):.3f}, "
          f"fakes reassured <0.4 {float((fake['prob'] < LOW_BAND).mean()):.3f}")

    ranked = shown.dropna(subset=["FP@0.5"])
    if len(ranked) >= 2:
        worst, best = ranked.iloc[0], ranked.iloc[-1]
        ratio = worst["FP@0.5"] / best["FP@0.5"] if best["FP@0.5"] else None
        spread = f"a {ratio:.1f}x spread" if ratio else "an unbounded spread"
        print(f"\n  Suppression harm is NOT uniform: '{worst['category']}' at {worst['FP@0.5']:.1%} "
              f"against '{best['category']}' at {best['FP@0.5']:.1%} - {spread},")
        print(f"  around a corpus-level average of {overall_fp:.1%}. That average hides the spread")
        print("  entirely, which is the point of this table.")
        if "política" in str(worst["category"]).lower():
            print("\n  NOTE: the worst-affected topic is POLITICS, which is this platform's stated")
            print("  domain. False positives there are the suppression of legitimate political")
            print("  speech named in the proposal - the harm is concentrated exactly where the")
            print("  artefact is aimed, and that belongs in the ethics chapter, not a footnote.")
    print("\n  FP = legitimate articles flagged (suppression). FN = misinformation passed through.")
    print("  These are not interchangeable and must not be summed into one 'error rate'.")

    out = _results_dir() / "bias_by_category.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
