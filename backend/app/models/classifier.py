"""Load the trained baseline artefact and serve predictions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODEL_PATH = "/app/artifacts/baseline_tfidf_logreg.joblib"
DECISION_THRESHOLD = 0.5
TOP_K_TOKENS = 8


@dataclass
class Prediction:
    prob: float
    label: str
    top_tokens: list[dict] = field(default_factory=list)


class _ModelState:
    """Holds the loaded artefact and a readiness flag, so a missing artefact degrades honestly."""

    def __init__(self) -> None:
        self.pipeline = None
        self.meta: dict = {}
        self.ready: bool = False
        self.error: str | None = None


_state = _ModelState()


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(path: Path | None = None) -> bool:
    """Load the artefact into module state."""
    target = path or model_path()
    try:
        bundle = joblib.load(target)
        _state.pipeline = bundle["pipeline"]
        _state.meta = bundle.get("meta", {})
        _state.ready = True
        _state.error = None
    except Exception as exc:  # noqa: BLE001 - surfaced via is_ready()/last_error()
        _state.pipeline = None
        _state.meta = {}
        _state.ready = False
        _state.error = f"{type(exc).__name__}: {exc}"
    return _state.ready


def is_ready() -> bool:
    return _state.ready


def last_error() -> str | None:
    return _state.error


def _positive_index(clf) -> int:
    """Index of the misinformation class (label 1) in the classifier's class ordering."""
    classes = list(clf.classes_)
    return classes.index(1) if 1 in classes else len(classes) - 1


def _shap_top_tokens(text: str, k: int = TOP_K_TOKENS) -> list[dict]:
    """Tokens that most influenced the decision, as SHAP values for the linear model."""
    vectorizer = _state.pipeline.named_steps["tfidf"]
    clf = _state.pipeline.named_steps["clf"]

    row = vectorizer.transform([text])
    present = row.indices
    if present.size == 0:
        return []

    coef = clf.coef_[0]
    tfidf_values = row.data

    means = _state.meta.get("feature_means")
    if means is not None:
        baseline = np.asarray(means)[present]
    else:
        baseline = np.zeros_like(tfidf_values)
    shap_values = (tfidf_values - baseline) * coef[present]

    order = np.argsort(np.abs(shap_values))[::-1][:k]
    feature_names = vectorizer.get_feature_names_out()
    return [
        {"token": str(feature_names[present[i]]), "weight": round(float(shap_values[i]), 4)}
        for i in order
    ]


def predict(text: str) -> Prediction:
    """Run the model on one piece of text."""
    if not _state.ready:
        raise RuntimeError("Model artefact not loaded.")

    clf = _state.pipeline.named_steps["clf"]
    proba = _state.pipeline.predict_proba([text])[0]
    prob_fake = float(proba[_positive_index(clf)])

    label_map = _state.meta.get(
        "label_map", {0: "likely_legitimate", 1: "likely_misinformation"}
    )
    predicted = 1 if prob_fake >= DECISION_THRESHOLD else 0
    label = label_map.get(predicted, label_map.get(str(predicted), str(predicted)))

    return Prediction(prob=prob_fake, label=label, top_tokens=_shap_top_tokens(text))
