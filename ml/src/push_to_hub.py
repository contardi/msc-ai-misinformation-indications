"""Upload the fine-tuned model artefacts to the HuggingFace Hub (preservation + distribution)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_MODELS = ["bertugues", "bertimbau"]


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "ml/artifacts"))


def _model_card(model_name: str, meta: dict) -> str:
    """A model card grounded in meta.json, carrying the responsible-AI framing."""
    m = meta.get("test_metrics", {}) or {}
    base = meta.get("hf_id", "unknown")

    def pct(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

    ood_note = (
        "- On the local out-of-distribution probe this checkpoint over-predicted 'misinformation' "
        "for genuine text (a known limitation of this run; see the project's detection-changelog).\n"
        if model_name == "bertimbau"
        else "- On the out-of-distribution probe it discriminated well (real text low, fake high).\n"
    )

    return f"""---
language: pt
license: other
base_model: {base}
datasets:
- fake_br
tags:
- text-classification
- misinformation
- fake-news
- portuguese
pipeline_tag: text-classification
---

# {model_name} fine-tuned for Brazilian Portuguese misinformation detection

Fine-tuned **{base}** on the **Fake.Br** corpus (Monteiro et al., 2018) for binary classification:
`0 = likely_legitimate`, `1 = likely_misinformation`. Part of an MSc dissertation artefact (an
explainable misinformation-detection platform for Brazilian Portuguese).

## Held-out test metrics (Fake.Br)
- accuracy: {pct(m.get('accuracy'))}
- precision: {pct(m.get('precision'))}
- recall: {pct(m.get('recall'))}
- f1: {pct(m.get('f1'))}

## Intended use and limitations
- Outputs a **probabilistic** misinformation assessment, **never a verdict** of truth/falsity.
- In-corpus metrics **overstate real-world performance**: the model learns the register of the
  training corpus, so short or novel claims are hard, and a content classifier cannot verify whether
  a specific factual claim actually happened (that is a fact-checking task).
{ood_note}- Use responsibly: false positives can suppress legitimate speech; false negatives let
  misinformation through. Not for definitive "true/false" judgements.

## Files
- `model.safetensors`, `config.json`, tokenizer files: standard, load with
  `AutoModelForSequenceClassification` / `AutoTokenizer`.
- `calibrator.joblib`: an isotonic calibrator (scikit-learn) applied to the raw fake-class
  probability so the reported score is better calibrated.
- `meta.json`: provenance (base model, seed, max_length, metrics).

## Load
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForSequenceClassification.from_pretrained("REPO_ID")
# calibrator: huggingface_hub.hf_hub_download("REPO_ID", "calibrator.joblib") then joblib.load(...)
```
"""


def push_model(token: str, user: str, model_name: str, suffix: str, private: bool) -> str | None:
    api = HfApi(token=token)
    model_dir = _artifacts_dir() / model_name
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        print(f"[push] skip '{model_name}': {model_dir} has no meta.json")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    repo_id = f"{user}/{model_name}{suffix}"
    (model_dir / "README.md").write_text(
        _model_card(model_name, meta).replace("REPO_ID", repo_id), encoding="utf-8"
    )

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {model_name} (fine-tuned on Fake.Br)",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[push] uploaded {model_name} -> {url} ({'private' if private else 'public'})")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Push fine-tuned models to the HuggingFace Hub.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model dirs.")
    parser.add_argument("--suffix", default="-fakebr", help="Repo name suffix (default: -fakebr).")
    parser.add_argument("--public", action="store_true", help="Create public repos (default: private).")
    parser.add_argument("--include-baseline", action="store_true", help="Also push the TF-IDF baseline.")
    args = parser.parse_args(argv)

    token = os.environ.get("HUGGINGFACE_API_KEY", "").strip()
    if not token:
        print("[push] ERROR: HUGGINGFACE_API_KEY is not set (add it to .env).")
        return 1

    try:
        user = HfApi(token=token).whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - bad/expired token
        print(f"[push] ERROR: could not authenticate with HuggingFace: {exc}")
        return 1
    print(f"[push] authenticated as: {user}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    urls = [push_model(token, user, m, args.suffix, not args.public) for m in models]

    if args.include_baseline:
        base = _artifacts_dir() / "baseline_tfidf_logreg.joblib"
        if base.exists():
            api = HfApi(token=token)
            repo_id = f"{user}/baseline-tfidf-fakebr"
            api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(base), path_in_repo="baseline_tfidf_logreg.joblib",
                repo_id=repo_id, repo_type="model", commit_message="Upload TF-IDF baseline",
            )
            print(f"[push] uploaded baseline -> https://huggingface.co/{repo_id}")

    ok = [u for u in urls if u]
    print(f"\n[push] done: {len(ok)}/{len(models)} model(s) uploaded.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
