"""Tests for the descriptive linguistic cues."""

from app.explanation.cues import extract_cues


def _labels(text):
    return {ind.label for ind in extract_cues(text)}


def test_link_is_no_longer_a_cue():
    """A web address fired MORE on legitimate articles in every corpus."""
    assert not any(c.kind == "link" for c in extract_cues("Leia em https://exemplo.com a notícia"))


def test_caps_is_no_longer_a_cue():
    """48.85% precision over 3,081 fires on Fake.Br - worse than a coin flip, and the single most frequent cue in the product."""
    assert not any(c.kind == "caps" for c in extract_cues("O IBGE divulgou DADOS OFICIAIS do PIB"))


def test_detects_exclamation_spam():
    assert "Pontuação sensacionalista" in _labels("Inacreditável!! Compartilhe!!")


def test_detects_sensational_terms():
    assert "Termos de apelo" in _labels("urgente, compartilhe essa bomba agora")


def test_clean_text_has_no_cues():
    text = "O instituto divulgou os dados oficiais da pesquisa nesta terça-feira."
    assert extract_cues(text) == []


def test_every_cue_carries_a_kind_and_its_variable_separately():
    """The front-end rebuilds these sentences in the reader's language."""
    text = "urgente!! compartilhe isso agora!! é uma bomba"
    cues = {ind.kind: ind for ind in extract_cues(text)}

    assert cues, "expected this text to trigger cues"
    assert "" not in cues, "a cue fired without a kind"

    assert cues["exclamations"].count and str(cues["exclamations"].count) in cues["exclamations"].detail
    assert cues["sensational_terms"].terms
    for term in cues["sensational_terms"].terms:
        assert term in cues["sensational_terms"].detail
    assert cues["sensational_terms"].count is None
    assert cues["exclamations"].terms == []


def test_cue_kinds_are_the_stable_set_the_ui_has_copy_for():
    """These strings are a contract with frontend/src/i18n/*.ts."""
    text = "urgente!! compartilhe isso agora!! é uma bomba"
    assert {ind.kind for ind in extract_cues(text)} == {"exclamations", "sensational_terms"}


def test_removed_terms_stay_removed():
    """`veja`, `alerta`, `repasse`, `exclusivo` and `escândalo` were measured out."""
    for text in ("veja o relatório do IBGE divulgado hoje",
                 "segundo a Veja, o alerta foi emitido",
                 "conteúdo exclusivo sobre o escândalo, repasse adiante"):
        assert extract_cues(text) == [], text
