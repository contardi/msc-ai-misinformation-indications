"""Train the TF-IDF + Logistic Regression baseline on the prepared Fake.Br corpus."""

from __future__ import annotations

import argparse
import math
import json
import os
import platform
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABEL_MAP = {0: "likely_legitimate", 1: "likely_misinformation"}


def _config_path() -> Path:
    return Path(os.environ.get("CONFIG_PATH", "ml/configs/default.yaml"))


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "ml/artifacts"))


def _results_dir() -> Path:
    """Committed evidence dir, unlike ARTIFACTS_DIR."""
    return Path(os.environ.get("RESULTS_DIR", "ml/results"))


def _library_versions() -> dict[str, str]:
    """Versions of everything that can change a result or make an artefact unloadable."""
    out = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "numpy", "scipy", "pandas", "joblib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def load_config() -> dict:
    path = _config_path()
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


PT_STOPWORDS = [
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos", "em", "no",
    "na", "nos", "nas", "num", "numa", "por", "pelo", "pela", "para", "pra", "com", "sem", "sob",
    "sobre", "entre", "até", "após", "ante", "e", "ou", "mas", "que", "se", "como", "quando",
    "porque", "pois", "porém", "contudo", "todavia", "ao", "aos", "à", "às", "é", "são", "ser",
    "está", "estão", "estava", "foi", "foram", "era", "eram", "ser", "tem", "têm", "tinha", "há",
    "havia", "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas", "isso", "isto",
    "aquele", "aquela", "aquilo", "ele", "ela", "eles", "elas", "eu", "tu", "nós", "vós", "você",
    "vocês", "me", "te", "lhe", "nos", "vos", "lhes", "meu", "minha", "teu", "tua", "seu", "sua",
    "seus", "suas", "nosso", "nossa", "dele", "dela", "deles", "delas", "já", "não", "sim", "também",
    "mais", "menos", "muito", "muita", "muitos", "pouco", "todo", "toda", "todos", "todas", "cada",
    "qual", "quais", "quem", "onde", "cujo", "cuja", "mesmo", "mesma", "ainda", "só", "apenas",
    "então", "assim", "aqui", "ali", "lá", "agora", "hoje", "ontem", "amanhã", "depois", "antes",
    "sua", "ser", "ter", "estar", "fazer", "vai", "vão", "foi", "será", "seria", "deve", "pode",
]


BASELINE_MODELS = ("logreg", "svm", "random_forest")


def _vectoriser() -> TfidfVectorizer:
    """Shared across every baseline, so the comparison isolates the classifier."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        stop_words=PT_STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_features=50_000,
        sublinear_tf=True,
    )


def build_classifier(model: str, seed: int, cfg: dict):
    """Instantiate one of the three WP4 baselines from the versioned config."""
    params = (cfg.get("baselines") or {}).get(model, {})
    if model == "logreg":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=seed,
        )
    if model == "svm":
        return LinearSVC(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 5000)),
            random_state=seed,
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown baseline '{model}' (expected one of {BASELINE_MODELS})")


def build_pipeline(seed: int, model: str = "logreg", cfg: dict | None = None) -> Pipeline:
    """TF-IDF features → classifier."""
    return Pipeline(
        steps=[("tfidf", _vectoriser()), ("clf", build_classifier(model, seed, cfg or {}))]
    )


def print_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n[eval] Held-out test metrics (positive class = fake/misinformation):")
    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall   : {recall:.4f}")
    print(f"  f1-score : {f1:.4f}")
    print("\n[eval] Confusion matrix (rows = true label, cols = predicted):")
    print("            pred_true  pred_fake")
    print(f"  true(0)   {cm[0, 0]:9d}  {cm[0, 1]:9d}")
    print(f"  fake(1)   {cm[1, 0]:9d}  {cm[1, 1]:9d}")
    print(f"\n[eval] False positives (legit flagged as fake): {cm[0, 1]}")
    print(f"[eval] False negatives (fake passed as legit):  {cm[1, 0]}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def mean_ci(scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean of k fold scores and a two-sided confidence interval."""
    n = len(scores)
    mean = float(np.mean(scores))
    if n < 2:
        return mean, mean, mean
    sem = float(np.std(scores, ddof=1)) / math.sqrt(n)
    half = float(t_dist.ppf(0.5 + confidence / 2, df=n - 1)) * sem
    return mean, mean - half, mean + half


def _save_folds(per_fold: dict, folds: int, seed: int) -> None:
    """Persist every per-fold score as committed evidence."""
    rows = [
        {"model": model, "metric": metric, "fold": i, "score": float(s), "folds": folds, "seed": seed}
        for model, metrics in per_fold.items()
        for metric, scores in metrics.items()
        for i, s in enumerate(scores)
    ]
    out = _results_dir() / "baseline_cv_folds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[cv] per-fold scores -> {out}")


def compare_paired(per_fold: dict, confidence: float, metric: str = "f1") -> None:
    """Compare models on the PER-FOLD differences, not by eyeballing interval overlap."""
    names = list(per_fold)
    if len(names) < 2:
        return
    print(f"\n[cv] Paired comparison on {metric} (per-fold differences, {int(confidence * 100)}% CI):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = per_fold[a][metric] - per_fold[b][metric]
            mean, lo, hi = mean_ci(diff, confidence)
            _stat, p = ttest_rel(per_fold[a][metric], per_fold[b][metric])
            verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
            print(f"  {a:>14} - {b:<14} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"p={p:.4f}  ({verdict})")
    print("  'includes 0' = no reliable difference on this evidence.")


def run_cross_validation(X, y, folds: int, seed: int, cfg: dict, models=BASELINE_MODELS) -> pd.DataFrame:
    """Stratified k-fold CV over every WP4 baseline, reporting all four metrics with CIs."""
    confidence = float((cfg.get("eval") or {}).get("confidence", 0.95))
    metrics = ["accuracy", "precision", "recall", "f1"]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    print(f"\n[cv] Stratified {folds}-fold cross-validation, positive class = fake")
    print(f"[cv] {int(confidence * 100)}% CI via Student's t (df={folds - 1}); folds are shared "
          f"across models, so comparisons are paired.")

    rows = []
    per_fold: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        pipeline = build_pipeline(seed, model, cfg)
        print(f"\n[cv] {model}...")
        cv = cross_validate(pipeline, X, y, cv=skf, scoring=metrics, n_jobs=-1, error_score="raise")
        per_fold[model] = {m: np.asarray(cv[f"test_{m}"]) for m in metrics}
        row = {"model": model}
        for m in metrics:
            mean, lo, hi = mean_ci(per_fold[model][m], confidence)
            row[m] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
            print(f"  {m:<9} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
        row["fit_time_s"] = float(np.mean(cv["fit_time"]))
        rows.append(row)

    compare_paired(per_fold, confidence, metric="f1")
    _save_folds(per_fold, folds, seed)
    df = pd.DataFrame(rows).set_index("model")
    print(f"\n[cv] Summary (mean [{int(confidence * 100)}% CI]):")
    print(f"  {'model':<15}" + "".join(f"{m:>26}" for m in metrics))
    for model, r in df.iterrows():
        cells = "".join(f"{r[m]:>14.4f} [{r[f'{m}_lo']:.3f},{r[f'{m}_hi']:.3f}]" for m in metrics)
        print(f"  {model:<15}{cells}")
    return df


def load_frames(paths: list[Path], seed: int) -> pd.DataFrame:
    """Load one or more labelled CSVs into a single frame."""
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")
        f = pd.read_csv(p)
        print(f"[train] loaded {len(f)} rows from {p.name}")
        frames.append(f)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    if len(paths) == 1:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[train] combined {before} -> {len(df)} rows after exact-text dedup "
          f"(fake={int(df['label'].sum())}, true={int((df['label'] == 0).sum())})")

    n = int(df["label"].value_counts().min())
    df = (
        pd.concat([g.sample(n, random_state=seed) for _, g in df.groupby("label")])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    print(f"[train] balanced to {len(df)} rows  (fake={int(df['label'].sum())}, "
          f"true={int((df['label'] == 0).sum())})")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + LogReg misinformation baseline.")
    parser.add_argument(
        "--data",
        default=None,
        help="CSV with text,label (default: <DATA_DIR>/processed/fakebr.csv). Accepts several, "
        "comma-separated, which are concatenated, de-duplicated and class-balanced.",
    )
    parser.add_argument("--out", default=None, help="Artefact path (default: <ARTIFACTS_DIR>/baseline_tfidf_logreg.joblib).")
    parser.add_argument("--cv", action="store_true", help="Also run stratified k-fold cross-validation.")
    parser.add_argument(
        "--cv-models",
        default=",".join(BASELINE_MODELS),
        help=f"Comma-separated baselines to cross-validate (default: all of {BASELINE_MODELS}). "
        "The SERVED artefact is always logistic regression; the others are evaluation baselines.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Stratified subsample of the loaded data to this many rows (0 = all). Used to build a "
        "learning curve: holding the test set fixed and varying only the training size separates "
        "'the model needs more data' from 'the ceiling is the task'.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run cross-validation and exit without refitting or overwriting the served artefact.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    folds = int(cfg.get("eval", {}).get("cv_folds", 10))

    if args.data:
        data_paths = [Path(p.strip()) for p in args.data.split(",") if p.strip()]
    else:
        data_paths = [_data_dir() / "processed" / "fakebr.csv"]
    out_path = Path(args.out) if args.out else _artifacts_dir() / "baseline_tfidf_logreg.joblib"

    missing = [p for p in data_paths if not p.exists()]
    if missing:
        print(f"[train] ERROR: dataset(s) not found: {', '.join(str(p) for p in missing)}. "
              "Run prepare_fakebr.py / prepare_faketruebr.py first.")
        return 1

    corpora = ",".join(p.stem for p in data_paths)
    print(f"[train] corpus: {corpora}")
    df = load_frames(data_paths, seed)
    if args.limit_rows and len(df) > args.limit_rows:
        per_class = args.limit_rows // 2
        df = (
            pd.concat([g.sample(min(len(g), per_class), random_state=seed)
                       for _, g in df.groupby("label")])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        print(f"[train] subsampled to {len(df)} rows  (fake={int(df['label'].sum())}, "
              f"true={int((df['label'] == 0).sum())})")
    X = df["text"].astype(str).tolist()
    y = df["label"].to_numpy()
    print(f"[train] loaded {len(df)} rows  (fake={int(y.sum())}, true={int((y == 0).sum())})")

    if args.cv or args.cv_only:
        wanted = [m.strip() for m in args.cv_models.split(",") if m.strip()]
        cv_df = run_cross_validation(X, y, folds, seed, cfg, wanted)
        results = _results_dir()
        results.mkdir(parents=True, exist_ok=True)
        cv_path = results / "baseline_cv.csv"
        cv_df.assign(folds=folds, seed=seed, n_rows=len(df)).to_csv(cv_path)
        print(f"\n[cv] saved -> {cv_path}")
        if args.cv_only:
            print("[cv] --cv-only: not refitting; the served artefact is untouched.")
            return 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    pipeline = build_pipeline(seed)
    print("\n[train] fitting on the training split...")
    pipeline.fit(X_train, y_train)
    metrics = print_metrics(y_test, pipeline.predict(X_test))

    print("\n[train] refitting on the full dataset for the served artefact...")
    final_pipeline = build_pipeline(seed)
    final_pipeline.fit(X, y)

    feature_means = np.asarray(
        final_pipeline.named_steps["tfidf"].transform(X).mean(axis=0)
    ).ravel()

    meta = {
        "model": "tfidf+logreg",
        "label_map": LABEL_MAP,
        "positive_label": 1,
        "seed": seed,
        "train_corpus": corpora,
        "n_rows": int(len(df)),
        "created": date.today().isoformat(),
        "heldout_metrics": metrics,
        "attribution": "shap (linear, closed-form): coef * (x - mean)",
        "versions": _library_versions(),
        "feature_means": feature_means,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipeline, "meta": meta}, out_path)
    print(f"\n[train] saved artefact -> {out_path}")

    results_models = _results_dir() / "models"
    results_models.mkdir(parents=True, exist_ok=True)
    provenance = {k: v for k, v in meta.items() if k != "feature_means"}
    provenance["artefact"] = out_path.name
    prov_path = results_models / f"{out_path.stem}.meta.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] provenance -> {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
