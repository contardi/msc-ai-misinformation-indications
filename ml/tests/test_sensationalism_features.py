"""The candidate patterns must fire on sensationalist writing and stay silent on ordinary news."""

import pytest
from sensationalism import ATTRIBUTION, FEATURE_NAMES, IMPERATIVE, PATTERNS, features, hits

ORDINARY = [
    "os ministros marco aurélio mello e gilmar mendes afirmaram nesta quinta-feira (6), antes da "
    "sessão do supremo tribunal federal (stf), que cabe ao senado somente cumprir a decisão do "
    "tribunal superior eleitoral que cassou o mandato de juíza selma.",
    "a agência nacional de vigilância sanitária (anvisa) emitiu um alerta na segunda-feira sobre "
    "uma investigação em curso do possível primeiro caso positivo no país de candida auris.",
    "o instituto divulgou nesta terça-feira os dados de inflação referentes ao mês anterior, "
    "apontando variação de 0,32% no índice oficial de preços ao consumidor.",
]

LOUD = [
    "urgentíssimo, compartilhem isso antes que apaguem tudo",
    "não querem que você saiba disso, repassem agora",
    "a mídia esconde o que aconteceu de verdade",
]


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_journalism_stays_quiet(text):
    """The failure mode that removed `link` and `caps` from the product."""
    fired = {h.kind for h in hits(text)}
    assert not fired, f"fired on ordinary reporting: {fired}"


@pytest.mark.parametrize("text", LOUD)
def test_sensationalist_writing_fires(text):
    assert hits(text), "no pattern fired on plainly sensationalist text"


def test_repasse_the_noun_does_not_fire():
    """Regression lock on a defect this module shipped with for about ten minutes."""
    for noun in (
        "o valor do repasse mensal será de mais de r$ 960 mil",
        "o governo deixou de utilizar, repassar ou devolver mais de r$ 37 milhões",
        "familiarizados com os cálculos do repasse do previne",
    ):
        assert not IMPERATIVE.search(noun), f"the noun `repasse` fired: {noun}"

    assert IMPERATIVE.search("repassem para todos os grupos")


def test_superlative_is_morphological_not_a_list():
    """The point of the suffix pattern."""
    for word in ("urgentíssimo", "gravíssima", "absurdíssimos", "chiquérrimo", "pobríssima"):
        assert PATTERNS["superlative"].search(f"isso é {word} mesmo"), word


def test_second_person_excludes_the_ambiguous_possessives():
    """`seu`/`sua` are also third-person possessives, so including them would be the `veja` mistake in a different disguise."""
    assert not PATTERNS["second_person"].search("o ministro apresentou seu relatório à comissão")
    assert PATTERNS["second_person"].search("você precisa ver isso")


def test_attribution_is_counted_so_its_absence_can_be_a_feature():
    assert ATTRIBUTION.search("segundo o relatório, a variação ficou em 0,32%")
    assert not ATTRIBUTION.search("isso é um absurdo total e ninguém faz nada")


def test_features_are_rates_not_counts():
    """Length is this project's known confound, so a feature must not be a proxy for it."""
    once = features("urgentíssimo!! compartilhem isso agora")
    twice = features("urgentíssimo!! compartilhem isso agora " * 2)
    for name in ("exclam_rate", "superlative_rate", "imperative_rate"):
        assert once[name] == pytest.approx(twice[name], rel=0.02), name


def test_word_count_is_not_a_feature():
    """It is the trivial CONTROL every arm is compared against, never an input."""
    assert not any("word" in name or "length" in name for name in FEATURE_NAMES)


def test_feature_order_is_stable():
    """A model's coefficients only mean something if the column order does."""
    assert tuple(features("qualquer texto").keys()) == FEATURE_NAMES


def test_hits_carry_spans_that_index_the_text():
    text = "isso é urgentíssimo, compartilhem agora"
    for h in hits(text):
        assert text[h.start:h.end] == h.match
