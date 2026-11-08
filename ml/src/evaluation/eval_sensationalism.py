"""Which method actually detects exaggerated framing?"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from sensationalism import FEATURE_NAMES, PATTERNS, features  # noqa: E402

SENSATIONAL_AT = 2


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


def roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    """Rank-based AUC, computed here so this runs without scikit-learn if it has to."""
    pos, neg = scores[target == 1], scores[target == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return (ranks[target == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


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


def _cue_fired(text: str) -> int:
    """The SHIPPED cue set, imported rather than reimplemented."""
    try:
        from app.explanation.cues import extract_cues
    except ImportError:  # pragma: no cover - only outside the backend image
        return -1
    return 1 if extract_cues(text) else 0


def build_matrix(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (morphology counts, structural features, full feature matrix)."""
    rows = [features(t) for t in texts]
    full = np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
    morph_idx = [i for i, n in enumerate(FEATURE_NAMES) if n.replace("_rate", "") in PATTERNS]
    struct_idx = [i for i in range(len(FEATURE_NAMES)) if i not in morph_idx]
    return full[:, morph_idx], full[:, struct_idx], full


def out_of_fold(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Cross-validated scores, so nothing is evaluated on data it was fitted to."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover
        return np.full(len(y), np.nan)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.full(len(y), np.nan)
    scores = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train]), y[train])
        scores[test] = model.predict_proba(scaler.transform(X[test]))[:, 1]
    return scores


def report(df: pd.DataFrame) -> pd.DataFrame:
    y = (df["exagero"].to_numpy() >= SENSATIONAL_AT).astype(int)
    texts = df["text"].fillna("").tolist()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    print(f"\n{len(df)} items coded | {n_pos} sensationalist (>= {SENSATIONAL_AT}), {n_neg} not")
    print(f"base rate {n_pos / len(df):.1%}   (Zamith et al. report 54.4% for their Brazilian sample)")
    dist = df["exagero"].value_counts().sort_index()
    print("rating distribution: " + "  ".join(f"{k}:{v}" for k, v in dist.items()))
    if n_pos < 10 or n_neg < 10:
        print("\nWARNING: one class has fewer than 10 items. Every figure below is unstable.")

    morph, struct, full = build_matrix(texts)
    arms: dict[str, np.ndarray] = {
        "morphology": morph.sum(axis=1),
        "structural": out_of_fold(struct, y),
        "combined": out_of_fold(full, y),
        "length (control)": np.array([len(t.split()) for t in texts], dtype=float),
    }
    if "prob" in df:
        arms["served classifier"] = df["prob"].to_numpy(dtype=float)

    print(f"\n{'arm':22}{'ROC-AUC':>10}{'vs length':>12}")
    print("-" * 44)
    raw_length = roc_auc(arms["length (control)"], y)
    length_auc = max(raw_length, 1.0 - raw_length)
    out = []
    for name, score in arms.items():
        auc = length_auc if name.startswith("length") else roc_auc(score, y)
        margin = "" if name.startswith("length") else f"{auc - length_auc:+.4f}"
        print(f"{name:22}{auc:>10.4f}{margin:>12}")
        out.append({"arm": name, "roc_auc": round(auc, 4), "n": len(df),
                    "n_sensational": n_pos, "vs_length": None if not margin else round(auc - length_auc, 4)})

    fired = np.array([_cue_fired(t) for t in texts])
    if fired.min() >= 0:
        tp = int(((fired == 1) & (y == 1)).sum())
        fp = int(((fired == 1) & (y == 0)).sum())
        fn = int(((fired == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        lo, hi = wilson(tp, tp + fp) if tp + fp else (float("nan"),) * 2
        print(f"\nSHIPPED CUE SET (the baseline to beat), binary so no AUC:")
        print(f"  fires on {int((fired == 1).sum())} of {len(df)} items")
        print(f"  precision {prec:.1%}  [{lo:.1%}, {hi:.1%}]     recall {rec:.1%}")
        mdd = min_detectable_difference(n_pos, n_neg, min(0.5, max(prec, 0.01)))
        print(f"  smallest difference this sample could detect at ~80% power: {mdd:.1%}"
              if mdd == mdd else "  detectable difference: n/a")
        out.append({"arm": "shipped cues", "roc_auc": None, "n": len(df), "n_sensational": n_pos,
                    "precision": round(prec, 4) if prec == prec else None,
                    "recall": round(rec, 4) if rec == rec else None})

    if "prob" in df and "label" in df:
        print("\nDoes the misinformation classifier fire on SENSATIONALIST BUT TRUE text?")
        print(f"  {'group':34}{'n':>5}{'mean prob':>11}{'flagged >0.6':>14}")
        for sens in (1, 0):
            for lab in (0, 1):
                g = df[(y == sens) & (df["label"] == lab)]
                if g.empty:
                    continue
                tag = f"{'sensationalist' if sens else 'plain':15} + {'TRUE' if lab == 0 else 'false':6}"
                print(f"  {tag:34}{len(g):>5}{g['prob'].mean():>11.3f}"
                      f"{(g['prob'] > 0.6).mean():>13.1%}")
        print("  The first row is the reframe's own false-positive harm: honest reporting, written")
        print("  with heat, that the platform would nonetheless flag.")

    return pd.DataFrame(out)


def _fixture() -> pd.DataFrame:
    """Synthetic data with a KNOWN answer, so the harness is verified before real labels exist."""
    rng = np.random.default_rng(42)
    LOUD = ("urgentíssimo compartilhem isso agora!! não querem que você saiba disso. "
            "você precisa ver!! é gravíssimo!!")
    PLAIN = ("o instituto divulgou nesta terça-feira os dados de inflação do mês anterior. "
             "segundo o relatório, a variação ficou em 0,32 por cento, informou a assessoria.")
    FILL = " o texto segue com mais informações sobre o assunto tratado nesta reportagem."
    rows = []
    for i in range(160):
        loud = i % 2 == 0
        label = 1 if loud else 0
        if i % 4 in (0, 3):
            label = 1 - label
        text = (LOUD if loud else PLAIN) + FILL * int(rng.integers(0, 4))
        rows.append({
            "id": i,
            "text": text,
            "exagero": int(rng.integers(3, 5)) if loud else int(rng.integers(0, 2)),
            "label": label,
            "prob": float(rng.uniform(0.6, 0.9) if loud else rng.uniform(0.1, 0.4)),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--fixture", action="store_true",
                    help="run against synthetic data with a known answer, no labels needed")
    args = ap.parse_args(argv)

    results = _results_dir()
    if args.fixture:
        print("=== FIXTURE: synthetic data, known answer. Every arm should separate cleanly. ===")
        report(_fixture())
        print("\n[sensationalism] fixture complete, nothing written")
        return 0

    sheet_path = Path(args.sheet) if args.sheet else results / "exagero_sheet.csv"
    key_path = Path(args.key) if args.key else results / "exagero_key.csv"
    for path in (sheet_path, key_path):
        if not path.exists():
            raise SystemExit(f"[sensationalism] missing {path}. Run make_sensationalism_sheet.py first.")

    sheet = pd.read_csv(sheet_path, dtype={"exagero": str})
    if sheet["exagero"].fillna("").str.strip().eq("").all():
        raise SystemExit(
            f"[sensationalism] {sheet_path} has no ratings yet. Code the 'exagero' column following "
            "documents/sensationalism-coding.md, then re-run. Use --fixture to test the harness."
        )
    key = pd.read_csv(key_path)
    df = sheet.merge(key, on="id", how="inner", validate="one_to_one")
    df = df[df["exagero"].fillna("").str.strip() != ""].copy()
    df["exagero"] = df["exagero"].astype(int)

    table = report(df)
    out = results / "sensationalism_methods.csv"
    table.to_csv(out, index=False)
    print(f"\n[sensationalism] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
