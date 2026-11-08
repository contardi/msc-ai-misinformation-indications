"""Fine-tune a Portuguese BERT (BERTimbau, later BERTugues) for misinformation classification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

LABEL_MAP = {0: "likely_legitimate", 1: "likely_misinformation"}

MODEL_IDS = {
    "bertugues": os.environ.get("BERTUGUES_HF_ID", "ricardoz/BERTugues-base-portuguese-cased"),
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
}

PROBE = [
    ("REAL long journalistic", "O Instituto Brasileiro de Geografia e Estatistica divulgou que a taxa de desemprego recuou para 7,8% no trimestre encerrado em maio."),
    ("REAL short factual", "A capital do Brasil e Brasilia."),
    ("REAL short mundane", "Gosto de tomar cafe pela manha antes de trabalhar."),
    ("FAKE sensational", "URGENTE! Vacina causa autismo, a midia esconde, compartilhe antes que apaguem!"),
    ("USER Venezuela example", "O governo federal informou que resgatou 13 brasileiros que estavam na Venezuela durante o terremoto que atingiu o pais. Os cidadaos foram levados de aviao de Caracas para o Rio de Janeiro pela Forca Aerea Brasileira."),
]


def _library_versions() -> dict[str, str]:
    """Versions of everything that can change a result or make an artefact unloadable."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "scikit-learn", "numpy", "pandas", "joblib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default))


def load_config() -> dict:
    with open(_env_path("CONFIG_PATH", "ml/configs/default.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


TRAIN_CORPORA = ["fakerecogna2.csv", "fakerecogna2_abs.csv", "faketruebr.csv"]


def load_dataset_frame() -> pd.DataFrame:
    """Combine Fake.Br (anchor) with any present recent corpora (TRAIN_CORPORA); dedup; report."""
    processed = _env_path("DATA_DIR", "data") / "processed"
    anchor = processed / "fakebr.csv"
    if not anchor.exists():
        raise FileNotFoundError(f"{anchor} not found. Run prepare_fakebr.py first.")
    frames = [pd.read_csv(anchor)]
    for name in TRAIN_CORPORA:
        p = processed / name
        if p.exists():
            frames.append(pd.read_csv(p))
            print(f"[train] including {name}")
    if len(frames) == 1:
        print("[train] no recent corpora found; training on Fake.Br alone.")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[train] combined rows: {len(df)}  (fake={int(df['label'].sum())}, true={int((df['label'] == 0).sum())})")

    n = int(df["label"].value_counts().min())
    df = (
        pd.concat([g.sample(n, random_state=42) for _, g in df.groupby("label")])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"[train] balanced rows: {len(df)}  (fake={int(df['label'].sum())}, true={int((df['label'] == 0).sum())})")
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _softmax_fake_prob(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune BERTimbau/BERTugues for misinformation.")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset + 1 epoch (CPU pipeline check).")
    parser.add_argument(
        "--model",
        choices=["bertugues", "bertimbau"],
        default=None,
        help="Which model to fine-tune. Default: config model.primary (bertugues). "
        "Run once per model to get the primary + the baseline for WP4.",
    )
    parser.add_argument("--out", default=None, help="Artefact dir (default: <ARTIFACTS_DIR>/<model>).")
    args = parser.parse_args(argv)

    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    model_name = args.model or cfg.get("model", {}).get("primary", "bertugues")
    if args.smoke and not args.model:
        model_name = "bertimbau"
    hf_id = MODEL_IDS.get(model_name, "")
    if not hf_id:
        print(f"[train] ERROR: no HuggingFace id for model '{model_name}'. Set BERTUGUES_HF_ID.")
        return 1
    max_length = int(cfg.get("model", {}).get("max_length", 256))
    tcfg = cfg.get("train", {})
    epochs = 1 if args.smoke else int(tcfg.get("epochs", 4))
    batch_size = int(tcfg.get("batch_size", 16))
    lr = float(tcfg.get("learning_rate", 2e-5))
    patience = int(tcfg.get("early_stopping_patience", 2))

    out_dir = Path(args.out) if args.out else _env_path("ARTIFACTS_DIR", "ml/artifacts") / model_name

    df = load_dataset_frame()
    if args.smoke:
        df = df.groupby("label", group_keys=False).head(100).reset_index(drop=True)
        print(f"[train] SMOKE subset: {len(df)} rows")

    train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    def to_ds(frame):
        return Dataset.from_pandas(frame[["text", "label"]], preserve_index=False).map(tok, batched=True)

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2)

    ckpt_dir = Path(tempfile.mkdtemp(prefix="hf_ckpt_"))
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        seed=seed,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    print(f"[train] fine-tuning {hf_id} ({'smoke' if args.smoke else 'full'})...")
    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    print("\n[eval] test metrics:")
    for k in ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1"):
        print(f"  {k.replace('eval_', ''):10s} {test_metrics.get(k, float('nan')):.4f}")

    val_logits = trainer.predict(val_ds).predictions
    val_raw = _softmax_fake_prob(np.asarray(val_logits))
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_df["label"].to_numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    meta = {
        "model": model_name,
        "hf_id": hf_id,
        "label_map": LABEL_MAP,
        "max_length": max_length,
        "seed": seed,
        "smoke": args.smoke,
        "created": date.today().isoformat(),
        "test_metrics": {k.replace("eval_", ""): test_metrics.get(k) for k in
                         ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1")},
        "versions": _library_versions(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    results_models = _env_path("RESULTS_DIR", "ml/results") / "models"
    results_models.mkdir(parents=True, exist_ok=True)
    (results_models / f"{model_name}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train] provenance -> {results_models / f'{model_name}.meta.json'}")
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"\n[train] saved artefact -> {out_dir}")

    print("\n[probe] raw -> calibrated p(fake):")
    enc = tokenizer([t for _, t in PROBE], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v for k, v in enc.items()}).logits.cpu().numpy()
    raw = _softmax_fake_prob(logits)
    cal = calibrator.predict(raw)
    for (name, _), r, c in zip(PROBE, raw, cal):
        print(f"  {r:5.3f} -> {c:5.3f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
