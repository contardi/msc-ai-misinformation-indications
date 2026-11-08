"""Unit tests for the model serving layer (loading, prediction, interim attribution)."""

from app.models import classifier


def test_loads_toy_artifact(toy_artifact):
    assert classifier.load_model(toy_artifact) is True
    assert classifier.is_ready() is True
    assert classifier.last_error() is None


def test_predict_returns_probability_and_label(toy_artifact):
    classifier.load_model(toy_artifact)
    pred = classifier.predict("urgente compartilhe essa mentira sobre o governo")
    assert 0.0 <= pred.prob <= 1.0
    assert pred.label in {"likely_misinformation", "likely_legitimate"}


def test_fake_like_text_scores_higher_than_factual(toy_artifact):
    classifier.load_model(toy_artifact)
    fake_like = classifier.predict("urgente boato mentira compartilhe agora")
    factual = classifier.predict("o instituto divulgou os dados oficiais da pesquisa")
    assert fake_like.prob > factual.prob


def test_top_tokens_are_present_in_input(toy_artifact):
    classifier.load_model(toy_artifact)
    pred = classifier.predict("urgente compartilhe essa mentira")
    assert pred.top_tokens, "expected at least one attributed token"
    for tw in pred.top_tokens:
        assert isinstance(tw["token"], str) and tw["token"]
        assert isinstance(tw["weight"], float)
    joined = " ".join(t["token"] for t in pred.top_tokens)
    assert any(word in joined for word in ("urgente", "compartilhe", "mentira"))


def test_missing_artifact_marks_not_ready(tmp_path):
    assert classifier.load_model(tmp_path / "does-not-exist.joblib") is False
    assert classifier.is_ready() is False
    assert classifier.last_error() is not None
