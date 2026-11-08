"""Cross-corpus evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _artifacts() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "ml/artifacts"))


def _results() -> Path:
    """Version-controlled evidence dir."""
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def default_baseline_path() -> Path:
    return _artifacts() / "baseline_tfidf_logreg.joblib"


def tfidf_probs(
    texts: list[str], path: Path | None = None
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Return (served_probs, raw_probs)."""
    path = path or default_baseline_path()
    if not path.exists():
        return None
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    idx = list(pipe.named_steps["clf"].classes_).index(1)
    return pipe.predict_proba(texts)[:, idx], None


def transformer_probs(
    model_name: str, texts: list[str], batch_size: int
) -> tuple[np.ndarray, np.ndarray | None] | None:
    model_dir = _artifacts() / model_name
    if not model_dir.exists():
        return None
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    cal_path = model_dir / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    meta = json.loads((model_dir / "meta.json").read_text()) if (model_dir / "meta.json").exists() else {}
    max_length = int(meta.get("max_length", 256))

    raw: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw.extend((e / e.sum(axis=1, keepdims=True))[:, 1].tolist())
    raw_probs = np.asarray(raw)
    if calibrator is None:
        return raw_probs, None
    return calibrator.predict(raw_probs), raw_probs


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """The canonical evaluation sample."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    per_class = n // 2
    parts = [g.sample(min(len(g), per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
    """Return the threshold maximising F1, plus the precision/recall/F1 achieved there."""
    prec, rec, thresholds = precision_recall_curve(y_true, probs, pos_label=1)
    if thresholds.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    denom = prec + rec
    f1_curve = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1_curve[:-1]))
    return float(thresholds[i]), float(prec[i]), float(rec[i]), float(f1_curve[i])


def report(name: str, y_true: np.ndarray, probs: np.ndarray, raw: np.ndarray | None = None) -> None:
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print(f"\n=== {name} ===")
    print(f"  @0.5     accuracy {acc:.4f} | precision {p:.4f} | recall {r:.4f} | f1 {f1:.4f}")
    print("  confusion (rows=true, cols=pred):  true0  fake1")
    print(f"    true(0)  {cm[0, 0]:6d} {cm[0, 1]:6d}")
    print(f"    fake(1)  {cm[1, 0]:6d} {cm[1, 1]:6d}")
    print(f"  false positives (legit->fake): {cm[0, 1]}   false negatives (fake->legit): {cm[1, 0]}")

    if np.unique(y_true).size < 2:
        print("  ranking  skipped (only one class present in this sample)")
        return
    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    thr, best_p, best_r, best_f1 = best_f1_threshold(y_true, probs)
    print(f"  ranking  ROC-AUC {auc:.4f} | PR-AUC {ap:.4f}   (0.5 = chance)")
    print(f"  @best    threshold {thr:.4f} | precision {best_p:.4f} | recall {best_r:.4f} | f1 {best_f1:.4f}")
    print(f"  headroom f1 {f1:.4f} -> {best_f1:.4f} (+{best_f1 - f1:.4f}) from re-thresholding alone")

    pos_rate = float(np.mean(y_true))
    trivial_f1 = 2 * pos_rate / (1 + pos_rate)
    print(f"  trivial  always-fake baseline f1 {trivial_f1:.4f} -> best F1 is {best_f1 - trivial_f1:+.4f} vs trivial")

    distinct = int(np.unique(probs).size)
    print(f"  spread   {distinct} distinct served probabilities over {probs.size} rows "
          f"({100 * (1 - distinct / probs.size):.1f}% ties)")
    if raw is not None:
        raw_auc = roc_auc_score(y_true, raw)
        raw_ap = average_precision_score(y_true, raw)
        raw_distinct = int(np.unique(raw).size)
        print(f"  raw      ROC-AUC {raw_auc:.4f} | PR-AUC {raw_ap:.4f} | {raw_distinct} distinct "
              f"({100 * (1 - raw_distinct / raw.size):.1f}% ties)")
        print(f"  calib    costs {raw_auc - auc:+.4f} ROC-AUC vs the raw score "
              f"(positive = the calibrator discards ranking information)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-corpus evaluation of the trained models.")
    parser.add_argument("--data", required=True, help="Labelled CSV with columns text,label.")
    parser.add_argument("--sample", type=int, default=5000, help="Stratified sample size (0/--full = all).")
    parser.add_argument("--full", action="store_true", help="Use the whole file.")
    parser.add_argument("--models", default="tfidf,bertugues,bertimbau", help="Comma-separated models.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-probs",
        default=None,
        help="Directory for <corpus>__<model>.csv of label,prob[,raw_prob]. Defaults to "
        "<RESULTS_DIR>/eval (committed; see ml/results/README.md). Pass an empty string to skip "
        "saving. Inference is the expensive step on CPU, so persisting the probabilities lets the "
        "metrics be recomputed, re-thresholded or plotted as ROC curves without repeating it.",
    )
    parser.add_argument(
        "--baseline-artefact",
        default=None,
        help="Path to the TF-IDF artefact to evaluate (default: <ARTIFACTS_DIR>/"
        "baseline_tfidf_logreg.joblib). Use this to evaluate a baseline trained on a different "
        "corpus without overwriting the served one.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix appended to each model name in the report and in the saved probability "
        "filename, e.g. --tag combined writes <corpus>__tfidf-combined.csv. Without it a "
        "non-default artefact would silently overwrite the served model's saved probabilities.",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    if not args.full and args.sample:
        df = stratified_sample(df, args.sample, args.seed)
    texts = df["text"].astype(str).tolist()
    y = df["label"].to_numpy()
    print(f"[eval] {len(df)} rows  (fake={int(y.sum())}, true={int((y == 0).sum())})  from {args.data}")

    if args.save_probs is None:
        probs_dir = _results() / "eval"
    elif args.save_probs.strip():
        probs_dir = Path(args.save_probs)
    else:
        probs_dir = None
    if probs_dir is not None:
        probs_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_artefact) if args.baseline_artefact else None
    suffix = f"-{args.tag}" if args.tag else ""
    if baseline_path is not None:
        print(f"[eval] baseline artefact: {baseline_path}")

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        result = (
            tfidf_probs(texts, baseline_path)
            if name == "tfidf"
            else transformer_probs(name, texts, args.batch_size)
        )
        label = f"{name}{suffix}"
        if result is None:
            print(f"\n=== {label} === (skipped: artefact not found)")
            continue
        probs, raw = result
        report(label, y, probs, raw)
        if probs_dir is not None:
            out = probs_dir / f"{Path(args.data).stem}__{label}.csv"
            columns = {"label": y, "prob": probs}
            if raw is not None:
                columns["raw_prob"] = raw
            if "category" in df.columns:
                columns["category"] = df["category"].to_numpy()
            columns["words"] = df["text"].astype(str).str.split().str.len().to_numpy()
            pd.DataFrame(columns).to_csv(out, index=False)
            print(f"  saved    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
