"""Transformer serving tests."""

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.models import transformer_classifier as tc  # noqa: E402

ARTEFACT = Path(os.environ.get("MODEL_DIR", "/app/artifacts/bertimbau"))

pytestmark = pytest.mark.skipif(
    not ARTEFACT.exists(), reason="no transformer artefact at MODEL_DIR"
)


def test_loads_and_predicts():
    assert tc.load_model(ARTEFACT) is True
    assert tc.is_ready() is True
    pred = tc.predict("urgente compartilhe essa mentira sobre o governo")
    assert 0.0 <= pred.prob <= 1.0
    assert pred.label in {"likely_misinformation", "likely_legitimate"}
    assert isinstance(pred.top_tokens, list)


def test_missing_dir_not_ready(tmp_path):
    assert tc.load_model(tmp_path / "nope") is False
    assert tc.is_ready() is False
