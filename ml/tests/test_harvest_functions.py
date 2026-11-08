"""The harvester's judgement, tested against the functions that actually ship."""

from datetime import timezone

import harvest_factchecks as h


def test_boatos_excerpt_yields_a_quotable_verdict_and_the_claim():
    rating, claim = h.split_verdict("Boato - Lula foi flagrado recebendo propina em Brasília")
    assert rating == "Boato"
    assert claim == "Lula foi flagrado recebendo propina em Brasília"


def test_every_observed_boatos_verdict_word_is_handled():
    """Measured over 100 posts."""
    for word in ("Boato", "Bulo", "Rumor", "Hoax", "Fake", "Fato"):
        rating, claim = h.split_verdict(f"{word} - alguma alegação que circulou")
        assert rating == word, f"{word} was not recognised as a verdict prefix"
        assert claim == "alguma alegação que circulou"


def test_the_verdict_word_is_kept_out_of_the_claim():
    """Not cosmetic."""
    _, claim = h.split_verdict("Boato - vacina causa autismo")
    assert "Boato" not in claim


def test_an_excerpt_with_no_prefix_returns_an_empty_verdict():
    rating, claim = h.split_verdict("Uma manchete comum sem prefixo de veredito")
    assert rating == ""
    assert claim == "Uma manchete comum sem prefixo de veredito"


def test_html_entities_and_tags_are_removed():
    assert h.strip_html("<p>ol&aacute; <b>mundo</b></p>") == "olá  mundo"


def test_non_breaking_spaces_become_ordinary_ones():
    """WordPress excerpts are full of \\xa0; left in, they change tokenisation."""
    assert "\xa0" not in h.strip_html("uma\xa0alegação")


def test_strip_html_tolerates_none():
    assert h.strip_html(None) == ""


def test_foreign_editions_are_excluded():
    """38% of a 200-post Boatos sample was Spanish or English (2026-08-26)."""
    exclude = ("english", "espanol")
    assert h.excluded_by_path("https://www.boatos.org/english/some-post", exclude) is True
    assert h.excluded_by_path("https://www.boatos.org/espanol/alguna-nota", exclude) is True


def test_brazilian_posts_survive_the_filter():
    assert h.excluded_by_path("https://www.boatos.org/politica/alguma-nota", ("english",)) is False


def test_an_empty_exclusion_list_excludes_nothing():
    assert h.excluded_by_path("https://www.boatos.org/english/x", ()) is False


def test_the_filter_matches_a_path_segment_not_a_substring():
    """'englishman' must not be caught by the 'english' rule."""
    assert h.excluded_by_path("https://www.boatos.org/englishman/x", ("english",)) is False


def _item(title="Título da checagem", excerpt="", link="https://www.boatos.org/x/y",
          date_gmt="2026-01-15T10:30:00"):
    return {"link": link, "title": {"rendered": title},
            "excerpt": {"rendered": excerpt}, "date_gmt": date_gmt}


def test_a_title_never_becomes_a_verdict():
    """THE important one."""
    row = h.to_row(_item(title="É falso que Lula virou jacaré"), "boatos", "title", "Boatos.org")
    assert row["rating"] == ""
    assert row["verdict"] == "unknown", "absence of a publisher verdict must never become 'false'"


def test_the_publishers_own_word_is_stored_verbatim_and_canonicalised():
    row = h.to_row(
        _item(title="É falso que a vacina causa autismo", excerpt="Boato - vacina causa autismo"),
        "boatos", "title", "Boatos.org",
    )
    assert row["rating"] == "Boato"
    assert row["verdict"] == "false"


def test_the_title_is_indexed_by_default_and_recorded_as_such():
    """Measured decision (2026-08-26)."""
    row = h.to_row(_item(title="Uma manchete", excerpt="Boato - a alegação"),
                   "boatos", "title", "Boatos.org")
    assert row["claim"] == "Uma manchete"
    assert row["claim_is_title"] is True


def test_asking_for_the_excerpt_indexes_the_claim_without_the_verdict_word():
    row = h.to_row(_item(title="Uma manchete", excerpt="Boato - a alegação como circulou"),
                   "boatos", "excerpt", "Boatos.org")
    assert row["claim"] == "a alegação como circulou"
    assert row["claim_is_title"] is False


def test_an_excerpt_request_falls_back_to_the_title_when_there_is_no_excerpt():
    row = h.to_row(_item(title="Uma manchete", excerpt=""), "boatos", "excerpt", "Boatos.org")
    assert row["claim"] == "Uma manchete"
    assert row["claim_is_title"] is True, "the fallback must be recorded, not hidden"


def test_rows_without_a_url_or_title_are_dropped():
    assert h.to_row(_item(link=""), "boatos", "title", "Boatos.org") is None
    assert h.to_row(_item(title=""), "boatos", "title", "Boatos.org") is None


def test_excluded_editions_are_dropped_before_anything_else():
    assert h.to_row(_item(link="https://www.boatos.org/english/x"), "boatos", "title",
                    "Boatos.org", ("english",)) is None


def test_dates_are_parsed_as_utc():
    """WordPress date_gmt has no offset."""
    row = h.to_row(_item(date_gmt="2026-01-15T10:30:00"), "boatos", "title", "Boatos.org")
    assert row["published_at"].tzinfo == timezone.utc
    assert row["published_at"].hour == 10


def test_an_unparseable_date_yields_null_rather_than_raising():
    row = h.to_row(_item(date_gmt="not a date"), "boatos", "title", "Boatos.org")
    assert row["published_at"] is None


def test_a_missing_date_yields_null():
    row = h.to_row(_item(date_gmt=""), "boatos", "title", "Boatos.org")
    assert row["published_at"] is None


def _g1_item(title, description="corpo do artigo " * 400, link="https://g1.globo.com/fato-ou-fake/noticia/2026/11/06/x.ghtml"):
    return {"title": title, "link": link, "pubDate": "Fri, 06 Nov 2026 22:44:14 -0000",
            "description": description}


def test_a_single_claim_keeps_its_verdict_and_loses_the_marker():
    row = h.rss_to_row(_g1_item("É #FAKE que Lula bloqueou o Bolsa Família após as eleições"),
                       "g1", "G1 Fato ou Fake")
    assert row["rating"] == "#FAKE", "the publisher's own marker, stored verbatim"
    assert row["verdict"] == "false"
    assert row["claim"] == "Lula bloqueou o Bolsa Família após as eleições"
    assert "#FAKE" not in row["claim"]
    assert row["claim_is_title"] is True


def test_the_affirmative_and_partial_markers_are_not_read_as_debunks():
    """A TRUE verdict rendering as a debunk is the defect this file was written after."""
    assert h.rss_to_row(_g1_item("#FATO: o salário mínimo subiu acima da inflação em 2026"),
                        "g1", "G1 Fato ou Fake")["verdict"] == "true"
    assert h.rss_to_row(_g1_item("#NÃOÉBEMASSIM: o reajuste existiu, mas não na proporção divulgada"),
                        "g1", "G1 Fato ou Fake")["verdict"] == "partial"


def test_a_debate_roundup_is_refused():
    """24 of 100 feed items carry many claims and no single verdict."""
    for title in (
        "Veja o que é #FATO ou #FAKE no debate da Globo com candidatos ao Governo de São Paulo",
        "Veja o que é #FATO ou #FAKE na entrevista de Lula à Globo no 2º turno das eleições 2026",
        "Fato ou Fake: entenda como funcionam as checagens e o trabalho da equipe",
        "Saiba como o Fato ou Fake checará declarações dos candidatos no debate",
    ):
        assert h.g1_marker(title) is None, f"should have been refused: {title}"
        assert h.rss_to_row(_g1_item(title), "g1", "G1 Fato ou Fake") is None


def test_an_unrecognised_headline_is_refused_rather_than_stored_as_unknown():
    """`unknown` means the publisher published no verdict, not that we declined to read one."""
    assert h.rss_to_row(_g1_item("Uma manchete qualquer sem marcador de veredito"),
                        "g1", "G1 Fato ou Fake") is None


def test_the_article_body_goes_to_summary_and_never_to_the_claim():
    """THE SEPARATION EVERY REPORTED RETRIEVAL FIGURE DEPENDS ON."""
    body = "O governo foi obrigado a contingenciar recursos em razão do teto de gastos. " * 60
    row = h.rss_to_row(_g1_item("É #FAKE que o governo bloqueou o Bolsa Família", description=body),
                       "g1", "G1 Fato ou Fake")

    assert row["claim"] == "o governo bloqueou o Bolsa Família"
    assert "contingenciar" not in row["claim"], "article prose reached the claim"
    assert len(row["claim"]) < 120
    assert row["summary"] and "contingenciar" in row["summary"], "the body belongs in summary"


def test_html_and_entities_are_stripped_from_both_fields():
    row = h.rss_to_row(_g1_item("É #FAKE que o &quot;governo&quot; bloqueou benef&iacute;cios",
                                description="<p>texto&nbsp;com <b>marcação</b></p>"),
                       "g1", "G1 Fato ou Fake")
    assert "&quot;" not in row["claim"] and "&iacute;" not in row["claim"]
    assert "<p>" not in row["summary"] and "&nbsp;" not in row["summary"]


def test_the_rss_date_is_parsed_rather_than_dropped():
    """RSS uses RFC 822, which datetime.fromisoformat cannot read - so this silently produced NULL published_at before parse_rss_date existed, and a NULL date means the watermark never advances."""
    row = h.rss_to_row(_g1_item("É #FAKE que algo aconteceu ontem em Brasília"),
                       "g1", "G1 Fato ou Fake")
    assert row["published_at"] is not None
    assert row["published_at"].year == 2026 and row["published_at"].month == 11
    assert row["published_at"].tzinfo is timezone.utc
    assert h.parse_rss_date("") is None
    assert h.parse_rss_date("not a date") is None


def test_every_publisher_declares_a_mechanism_the_harvester_implements():
    """A config entry naming a mechanism with no code path would fail at request time, not here."""
    for slug, (_endpoint, _field, _exclude, mechanism) in h.PUBLISHERS.items():
        assert mechanism in {"wordpress", "rss"}, f"{slug} declares unknown mechanism {mechanism}"
