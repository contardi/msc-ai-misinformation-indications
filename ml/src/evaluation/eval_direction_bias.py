"""Does the classifier's suppression harm fall evenly across the political spectrum?"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

THRESHOLD = 0.5
BAND_HIGH = 0.6
DIRECTIONS = ("esquerda", "direita", "ambos", "neutro", "indefinido")
COMPARED = ("esquerda", "direita")

PARTISAN = ("esquerda", "direita")
NON_PARTISAN = ("ambos", "neutro")


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]], without SciPy."""
    def logfact(k: int) -> float:
        return math.lgamma(k + 1)

    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    const = logfact(r1) + logfact(r2) + logfact(c1) + logfact(c2) - logfact(n)

    def prob(x: int) -> float:
        y, z_, w = c1 - x, r1 - x, r2 - (c1 - x)
        if min(y, z_, w) < 0:
            return 0.0
        return math.exp(const - (logfact(x) + logfact(z_) + logfact(y) + logfact(w)))

    observed = prob(a)
    total = 0.0
    for x in range(max(0, c1 - r2), min(c1, r1) + 1):
        p = prob(x)
        if p <= observed * (1 + 1e-9):
            total += p
    return min(1.0, total)


def min_detectable_difference(n1: int, n2: int, baseline: float,
                              alpha_z: float = 1.96, power_z: float = 0.84) -> float:
    """Smallest difference in proportions this sample could detect at ~80% power."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    for step in range(1, 1001):
        delta = step / 1000
        p2 = baseline + delta
        if p2 >= 1.0:
            return float("nan")
        se = math.sqrt(baseline * (1 - baseline) / n1 + p2 * (1 - p2) / n2)
        if delta >= (alpha_z + power_z) * se:
            return delta
    return float("nan")


def analyse(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    df = sheet.merge(key, on="id", how="inner", validate="one_to_one")
    df["direction"] = df["direction"].fillna("").str.strip().str.lower()

    coded = df[df["direction"].isin(DIRECTIONS)]
    uncoded = len(df) - len(coded)
    unknown = sorted(set(df["direction"]) - set(DIRECTIONS) - {""})
    if unknown:
        print(f"[direction] WARNING unrecognised labels ignored: {unknown}")
    print(f"[direction] {len(coded)} coded, {uncoded} blank or unrecognised, of {len(df)} rows\n")

    rows = []
    for name in DIRECTIONS:
        g = coded[coded["direction"] == name]
        if g.empty:
            continue
        n = len(g)
        fp50 = int((g["prob"] >= THRESHOLD).sum())
        fp60 = int((g["prob"] > BAND_HIGH).sum())
        lo50, hi50 = wilson(fp50, n)
        rows.append({
            "direction": name, "n": n,
            "fp_at_0.5": fp50, "fp_rate_0.5": fp50 / n, "ci_lo": lo50, "ci_hi": hi50,
            "fp_above_band_0.6": fp60, "fp_rate_0.6": fp60 / n,
            "mean_prob": g["prob"].mean(),
            "median_words": g["words"].median() if "words" in g else float("nan"),
        })
    return pd.DataFrame(rows)


def report(table: pd.DataFrame, coded: pd.DataFrame) -> list[dict]:
    print(f"{'direction':<12}{'n':>5}{'FP@0.5':>9}{'95% CI':>18}{'FP>0.6':>9}"
          f"{'mean p':>9}{'med words':>11}")
    print("-" * 73)
    for r in table.to_dict("records"):
        ci = f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]"
        print(f"{r['direction']:<12}{r['n']:>5}{r['fp_rate_0.5']:>8.1%}{ci:>18}"
              f"{r['fp_rate_0.6']:>9.1%}{r['mean_prob']:>9.3f}{r['median_words']:>11.0f}")

    left = table[table["direction"] == "esquerda"]
    right = table[table["direction"] == "direita"]
    if left.empty or right.empty:
        print("\n[direction] both 'esquerda' and 'direita' are needed for the comparison.")
        return []

    ln, rn = int(left["n"].iloc[0]), int(right["n"].iloc[0])
    lf, rf = int(left["fp_at_0.5"].iloc[0]), int(right["fp_at_0.5"].iloc[0])
    lr, rr = lf / ln, rf / rn
    p = fisher_exact(lf, ln - lf, rf, rn - rf)
    mdd = min_detectable_difference(ln, rn, min(lr, rr))

    print(f"\nesquerda {lr:.1%} ({lf}/{ln})  vs  direita {rr:.1%} ({rf}/{rn})")
    print(f"observed difference : {abs(lr - rr):.1%} ({'esquerda' if lr > rr else 'direita'} higher)")
    print(f"Fisher exact p      : {p:.4f}")
    print(f"detectable at ~80%  : {mdd:.1%}" if mdd == mdd else "detectable at ~80%  : n/a")

    print()
    if p < 0.05:
        print("FINDING: the false-positive rate differs by political direction. This belongs in the")
        print("ethics discussion as a headline, not a footnote - it is a measured asymmetry in whose")
        print("legitimate speech the platform would suppress.")
    else:
        print(f"NULL: no significant asymmetry. Report it as 'no difference larger than about")
        print(f"{mdd:.0%} was detectable at this sample size', never as 'no difference'.")

    lw = coded[coded["direction"] == "esquerda"]["words"].median()
    rw = coded[coded["direction"] == "direita"]["words"].median()
    if lw == lw and rw == rw and max(lw, rw) > 0 and abs(lw - rw) / max(lw, rw) > 0.20:
        print(f"\nCONFOUND WARNING: median length differs by direction ({lw:.0f} vs {rw:.0f} words).")
        print("Word count alone reaches ROC-AUC 0.809 on FakeRecogna 2.0, so any difference above")
        print("may be a length effect wearing a politics label. Report within length bands.")

    return _exploratory(coded)


def _exploratory(coded: pd.DataFrame) -> list[dict]:
    """Single-protagonist coverage against institutional and two-sided coverage."""
    par = coded[coded["direction"].isin(PARTISAN)]
    non = coded[coded["direction"].isin(NON_PARTISAN)]
    if par.empty or non.empty:
        return []

    print("\n" + "=" * 73)
    print("EXPLORATORY (post-hoc): single-protagonist coverage vs institutional / two-sided")
    print("Formed after the pre-specified test above returned a null. Treat accordingly.")
    print("=" * 73)

    rows = []
    for label, cut, strict in (("flagged at 0.5 (decision threshold)", THRESHOLD, False),
                               ("shown above the 0.6 band (what users meet)", BAND_HIGH, True)):
        pf = int((par["prob"] > cut).sum() if strict else (par["prob"] >= cut).sum())
        nf = int((non["prob"] > cut).sum() if strict else (non["prob"] >= cut).sum())
        pr_, nr = pf / len(par), nf / len(non)
        p = fisher_exact(pf, len(par) - pf, nf, len(non) - nf)
        mdd = min_detectable_difference(len(par), len(non), min(pr_, nr))
        print(f"\n{label}")
        print(f"  partisan      {pr_:>6.1%}  ({pf}/{len(par)})")
        print(f"  non-partisan  {nr:>6.1%}  ({nf}/{len(non)})")
        print(f"  difference {pr_ - nr:+.1%}   Fisher exact p = {p:.5f}"
              + (f"   detectable at ~80%: {mdd:.1%}" if mdd == mdd else ""))
        rows.append({
            "cut": cut,
            "cut_meaning": "decision threshold" if not strict else "interface band",
            "partisan_n": len(par), "partisan_flagged": pf, "partisan_rate": round(pr_, 4),
            "non_partisan_n": len(non), "non_partisan_flagged": nf, "non_partisan_rate": round(nr, 4),
            "difference": round(pr_ - nr, 4),
            "fisher_p": round(p, 6),
            "min_detectable_difference": round(mdd, 4) if mdd == mdd else None,
            "partisan_median_words": int(par["words"].median()),
            "non_partisan_median_words": int(non["words"].median()),
            "prespecified": False,
        })

    pw, nw = par["words"].median(), non["words"].median()
    worst = max(pw, nw)
    print(f"\nlength control : partisan {pw:.0f} vs non-partisan {nw:.0f} median words"
          + ("  <- CONFOUNDED, report within bands" if worst > 0 and abs(pw - nw) / worst > 0.20
             else "  -> matched, length does not explain it"))

    parts = [(g, coded[coded["direction"] == g]) for g in NON_PARTISAN]
    detail = "   ".join(f"{g} {len(d)} items {(d['prob'] >= THRESHOLD).mean():.1%}"
                        for g, d in parts if not d.empty)
    print(f"internal check : {detail}")
    print("                 (coded as separate categories; agreement here is what makes the")
    print("                  grouping a finding rather than an artefact of how it was drawn)")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--key", default=None)
    args = ap.parse_args(argv)

    results = _results_dir()
    sheet_path = Path(args.sheet) if args.sheet else results / "direction_sheet.csv"
    key_path = Path(args.key) if args.key else results / "direction_key.csv"
    for path in (sheet_path, key_path):
        if not path.exists():
            raise SystemExit(f"[direction] missing {path}. Run make_direction_sheet.py first.")

    sheet = pd.read_csv(sheet_path, dtype={"direction": str})
    key = pd.read_csv(key_path)
    if sheet["direction"].fillna("").str.strip().eq("").all():
        raise SystemExit(
            f"[direction] {sheet_path} has no labels yet. Code the 'direction' column following "
            "documents/political-direction-coding.md, then re-run."
        )

    table = analyse(sheet, key)
    merged = sheet.merge(key, on="id")
    merged["direction"] = merged["direction"].fillna("").str.strip().str.lower()
    exploratory = report(table, merged)

    out = results / "direction_bias.csv"
    table.to_csv(out, index=False)
    print(f"\n[direction] -> {out}")
    if exploratory:
        out2 = results / "direction_bias_exploratory.csv"
        pd.DataFrame(exploratory).to_csv(out2, index=False)
        print(f"[direction] -> {out2}  (post-hoc, labelled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
