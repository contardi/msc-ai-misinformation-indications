"""The canonical verdict must never turn an affirmative or unreadable rating into a debunk."""

from app.api.verdicts import VERDICTS, canonical_verdict


def test_affirmative_ratings_are_not_debunks():
    for rating in ("verdadeiro", "Verdadeiro", "VERDADEIRO", "comprovado", "confirmado", "correto"):
        assert canonical_verdict(rating) == "true", rating


def test_hedged_verdict_is_partial_not_true():
    for rating in ("verdadeiro, mas...", "Verdadeiro, mas exagerado"):
        assert canonical_verdict(rating) == "partial", rating


def test_debunks_still_read_as_false():
    for rating in ("falso", "Falso", "falsa", "É boato", "mentira"):
        assert canonical_verdict(rating) == "false", rating


def test_publisher_specific_vocabulary_maps_to_false():
    """Boatos.org's own words for the same conclusion, measured over 100 consecutive posts."""
    for rating in ("Boato", "Bulo", "Rumor", "Hoax", "RUMOR", "boato"):
        assert canonical_verdict(rating) == "false", rating


def test_qualified_verdicts_are_partial():
    for rating in ("enganoso", "exagerado", "distorcido", "fora de contexto", "contextualizando",
                   "insustentável", "subestimado", "sátira"):
        assert canonical_verdict(rating) == "partial", rating


def test_unreadable_rating_falls_back_to_unknown_never_false():
    for rating in ("", "   ", "Ainda é cedo", "De olho", "MadeiraMadeira tem clientes satisfeitos?"):
        assert canonical_verdict(rating) == "unknown", repr(rating)


def test_paragraph_ratings_are_read_from_their_leading_verdict():
    assert canonical_verdict(
        "falso: é mentira que portarias revogadas pelo ministério da saúde em 2023 influenciaram "
        "no caso de transplante de órgãos contaminados"
    ) == "false"
    assert canonical_verdict(
        "enganoso: posts enganam ao editar um vídeo de 2022, feito no contexto das eleições"
    ) == "partial"


def test_result_is_always_one_of_the_declared_verdicts():
    for rating in ("", "falso", "verdadeiro", "???", "Façso", "\n\t", "1234"):
        assert canonical_verdict(rating) in VERDICTS, repr(rating)


def test_never_raises_on_hostile_input():
    for rating in (None, "x" * 10_000, "🇧🇷", "<script>"):
        assert canonical_verdict(rating) in VERDICTS  # type: ignore[arg-type]
