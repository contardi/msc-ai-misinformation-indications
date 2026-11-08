"""Shared test fixtures."""

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_TOY_TEXTS = [
    "urgente compartilhe agora essa mentira absurda sobre o governo",
    "boato falso viralizou nas redes compartilhe urgente",
    "mentira escandalosa que a mídia esconde compartilhe",
    "o instituto divulgou os dados oficiais da pesquisa nesta terça",
    "o ministério anunciou a nova medida em comunicado oficial",
    "a agência publicou o relatório técnico com os números do estudo",
]
_TOY_LABELS = [1, 1, 1, 0, 0, 0]


@pytest.fixture(scope="session")
def toy_artifact(tmp_path_factory):
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ).fit(_TOY_TEXTS, _TOY_LABELS)

    feature_means = np.asarray(pipeline.named_steps["tfidf"].transform(_TOY_TEXTS).mean(axis=0)).ravel()

    path = tmp_path_factory.mktemp("artifacts") / "toy.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "meta": {
                "label_map": {0: "likely_legitimate", 1: "likely_misinformation"},
                "feature_means": feature_means,
            },
        },
        path,
    )
    return path


@pytest.fixture(autouse=True)
def _set_model_path(toy_artifact, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(toy_artifact))
    monkeypatch.setenv("RATE_LIMIT", "1000/minute")
    monkeypatch.delenv("GOOGLE_FACTCHECK_API_KEY", raising=False)
