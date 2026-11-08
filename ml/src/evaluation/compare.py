"""Compare the baseline (TF-IDF + LogReg) against the fine-tuned transformer on the OOD probe."""

from __future__ import annotations

import os
from pathlib import Path

import joblib

PROBE = [
    ("REAL long journalistic", "O Instituto Brasileiro de Geografia e Estatistica divulgou que a taxa de desemprego recuou para 7,8% no trimestre encerrado em maio."),
    ("REAL short factual", "A capital do Brasil e Brasilia."),
    ("REAL short mundane", "Gosto de tomar cafe pela manha antes de trabalhar."),
    ("FAKE sensational", "URGENTE! Vacina causa autismo, a midia esconde, compartilhe antes que apaguem!"),
    ("USER Venezuela example", "O governo federal informou que resgatou 13 brasileiros que estavam na Venezuela durante o terremoto que atingiu o pais. Os cidadaos foram levados de aviao de Caracas para o Rio de Janeiro pela Forca Aerea Brasileira."),
]


def _artifacts() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "ml/artifacts"))


def baseline_probs(texts: list[str]) -> list[float] | None:
    path = _artifacts() / "baseline_tfidf_logreg.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    idx = list(pipe.named_steps["clf"].classes_).index(1)
    return [float(pipe.predict_proba([t])[0][idx]) for t in texts]


def transformer_probs(texts: list[str], model_name: str) -> list[float] | None:
    model_dir = _artifacts() / model_name
    if not model_dir.exists():
        return None
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    cal_path = model_dir / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    out: list[float] = []
    for t in texts:
        enc = tok([t], truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw = float((e / e.sum(axis=1, keepdims=True))[0, 1])
        out.append(float(calibrator.predict([raw])[0]) if calibrator is not None else raw)
    return out


def main() -> int:
    texts = [t for _, t in PROBE]
    cols = {
        "tfidf": baseline_probs(texts),
        "bertimbau": transformer_probs(texts, "bertimbau"),
        "bertugues": transformer_probs(texts, "bertugues"),
    }
    present = {k: v for k, v in cols.items() if v is not None}
    header = "  ".join(f"{k:>10s}" for k in present)
    print(f"{'text':28s}  {header}")
    for i, (name, _) in enumerate(PROBE):
        row = "  ".join(f"{present[k][i]:>10.3f}" for k in present)
        print(f"{name:28s}  {row}")
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        print(f"\n[compare] not found (skipped): {', '.join(missing)} "
              f"(train them or check ml/artifacts/<name>).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
