#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
echo "[gpu-train] repo root: ${REPO_ROOT}"

SMOKE="${SMOKE:-0}"
ONLY_BERTUGUES="${ONLY_BERTUGUES:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"  # don't hang forever on a slow mirror
SMOKE_FLAG=""
[ "${SMOKE}" = "1" ] && SMOKE_FLAG="--smoke"

if [ "${SMOKE}" = "1" ]; then
  echo "[gpu-train] SMOKE=1 -> CPU dry run; skipping GPU assertion."
else
  echo "[gpu-train] checking for a CUDA GPU..."
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv || true
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "[gpu-train] ERROR: no CUDA GPU visible to PyTorch. Rent a GPU box, or run with SMOKE=1 for a CPU dry run." >&2
    exit 1
  }
fi

echo "[gpu-train] installing dependencies..."
python -m pip install -q --upgrade pip
if [ "${PIN_DEPS:-1}" = "1" ]; then
  python -m pip install -q \
    transformers==5.12.1 accelerate==1.14.0 datasets==5.0.0 scikit-learn==1.9.0 \
    pandas==3.0.4 numpy==2.5.0 pyyaml==6.0.3 joblib==1.5.3 scipy==1.18.0 huggingface-hub==1.21.0
else
  echo "[gpu-train] WARNING: PIN_DEPS=0 - installing unpinned; this run is not exactly reproducible."
  python -m pip install -q "transformers>=4.44" accelerate datasets scikit-learn pandas pyyaml joblib huggingface_hub
fi

HF_FETCH_TIMEOUT="${HF_FETCH_TIMEOUT:-1200}"   # seconds per attempt; ~440 MB per model
echo "[gpu-train] pre-fetching pretrained weights (fail fast, before any training)..."
FETCH_IDS="ricardoz/BERTugues-base-portuguese-cased"
[ "${ONLY_BERTUGUES}" = "1" ] || FETCH_IDS="${FETCH_IDS} neuralmind/bert-base-portuguese-cased"
for HF_ID in ${FETCH_IDS}; do
  fetched=0
  for attempt in 1 2 3; do
    if HF_ID="${HF_ID}" timeout "${HF_FETCH_TIMEOUT}" python -c '
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
hf_id = os.environ["HF_ID"]
AutoTokenizer.from_pretrained(hf_id)
AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2)
print(f"[gpu-train] cached {hf_id}")
'; then
      fetched=1
      break
    fi
    echo "[gpu-train] fetch of ${HF_ID} failed or stalled past ${HF_FETCH_TIMEOUT}s" \
         "(attempt ${attempt}/3); retrying in 15s..." >&2
    sleep 15
  done
  if [ "${fetched}" != "1" ]; then
    echo "[gpu-train] ERROR: could not fetch ${HF_ID} after 3 attempts. Nothing has been trained," >&2
    echo "[gpu-train]        so terminate the box now and retry when the network settles." >&2
    exit 1
  fi
done

echo "[gpu-train] preparing Fake.Br (training anchor)..."
python ml/src/data/prepare_fakebr.py

if [ "${SMOKE}" = "1" ]; then
  echo "[gpu-train] SMOKE -> skipping the large FakeRecogna 2.0 downloads (Fake.Br alone is enough for a pipeline check)."
else
  echo "[gpu-train] preparing FakeRecogna 2.0 (extractive + abstractive)..."
  python ml/src/data/prepare_faketruebr.py --hf-dataset recogna-nlp/fakerecogna2-extrativa  --out data/processed/fakerecogna2.csv
  python ml/src/data/prepare_faketruebr.py --hf-dataset recogna-nlp/fakerecogna2-abstrativa --out data/processed/fakerecogna2_abs.csv
fi

echo "[gpu-train] fine-tuning BERTugues (primary)..."
python ml/src/training/train_transformer.py --model bertugues ${SMOKE_FLAG}

if [ "${ONLY_BERTUGUES}" = "1" ]; then
  echo "[gpu-train] ONLY_BERTUGUES=1 -> skipping the BERTimbau baseline."
else
  echo "[gpu-train] fine-tuning BERTimbau (baseline)..."
  python ml/src/training/train_transformer.py --model bertimbau ${SMOKE_FLAG}
fi

if [ "${SMOKE}" = "1" ]; then
  echo "[gpu-train] SMOKE -> skipping the FakeRecogna 1.0 reference evaluation."
else
  echo "[gpu-train] reference evaluation on FakeRecogna 1.0 (full text)."
  echo "[gpu-train] WARNING: this is NOT a held-out test. 49.3% of FakeRecogna 1.0 overlaps"
  echo "[gpu-train]          FakeRecogna 2.0 (76.3% of its fakes), which IS in training."
  echo "[gpu-train]          Report these numbers as contaminated, never as generalisation."
  python ml/src/data/prepare_faketruebr.py --hf-dataset recogna-nlp/FakeRecogna --invert-label --out data/processed/fakerecogna1.csv
  python ml/src/evaluation/eval_crosscorpus.py --data data/processed/fakerecogna1.csv --sample 5000
fi

if [ "${SMOKE}" = "1" ]; then
  echo "[gpu-train] SMOKE -> NOT pushing to HuggingFace (a toy model must never overwrite the real weights)."
  echo "[gpu-train] smoke artefacts are in ${ARTIFACTS_DIR:-ml/artifacts} and can be deleted."
elif [ -n "${HUGGINGFACE_API_KEY:-}" ]; then
  echo "[gpu-train] pushing artefacts to HuggingFace Hub..."
  PUSH_ARGS=""
  [ "${ONLY_BERTUGUES}" = "1" ] && PUSH_ARGS="--models bertugues"
  python ml/src/push_to_hub.py ${PUSH_ARGS}
  echo "[gpu-train] done. Artefacts are on HuggingFace - safe to TERMINATE the box now."
else
  echo "[gpu-train] WARNING: HUGGINGFACE_API_KEY not set - artefacts are only in ml/artifacts/ on THIS box."
  echo "[gpu-train] Copy them off before terminating, e.g.:"
  echo "            tar czf artifacts.tgz ml/artifacts && <scp/rsync it down>"
  echo "            or set HUGGINGFACE_API_KEY and re-run push: python ml/src/push_to_hub.py"
fi

echo "[gpu-train] finished."
