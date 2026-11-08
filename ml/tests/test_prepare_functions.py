"""Corpus preparation and the paired-field fetcher, tested against the real modules."""

import csv

import fetch_publisher_fields as f
import prepare_fakebr as p


def test_slug_is_the_last_path_segment():
    assert f.slug_of("https://www.boatos.org/politica/lula-jacare.html") == "lula-jacare"


def test_a_trailing_slash_does_not_produce_an_empty_slug():
    """WordPress links usually end in '/'."""
    assert f.slug_of("https://www.boatos.org/politica/lula-jacare/") == "lula-jacare"


def test_query_strings_and_fragments_do_not_leak_into_the_slug():
    assert f.slug_of("https://www.boatos.org/x/minha-nota/?utm_source=x") == "minha-nota"
    assert f.slug_of("https://www.boatos.org/x/minha-nota/#comentarios") == "minha-nota"


def test_the_fetcher_and_the_harvester_split_verdicts_identically():
    """These are genuinely separate implementations in two modules, and the field-comparison study (7,550 rows, 2026-08-26) is only interpretable if both read a publisher's verdict the same way."""
    import harvest_factchecks as h

    for excerpt in (
        "Boato - Lula virou jacaré",
        "Bulo - alguma alegação",
        "Rumor - outra alegação",
        "Hoax - mais uma",
        "Fake - e outra",
        "Fato - esta procede",
        "Uma manchete sem prefixo nenhum",
        "",
    ):
        assert f.split_verdict(excerpt) == h.split_verdict(excerpt), excerpt


def test_both_strip_html_implementations_agree():
    import harvest_factchecks as h

    for raw in ("<p>ol&aacute; <b>mundo</b></p>", "texto\xa0com espaço fixo", "", None):
        assert f.strip_html(raw) == h.strip_html(raw)


def _corpus(root, layout, n=3):
    """Build a miniature Fake.Br tree."""
    for label in ("true", "fake"):
        d = root / layout / label
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i}.txt").write_text(f"conteúdo {label} {i}", encoding="utf-8")
    return root


def test_parse_corpus_labels_fake_as_one_and_true_as_zero(tmp_path):
    """Inverting these would train a model that is exactly wrong while every metric still looks plausible, because accuracy near 0 is as unlikely to be noticed as accuracy near 1 is."""
    rows = p.parse_corpus(_corpus(tmp_path, "size_normalized_texts"), "size_normalized_texts")
    labels = {label for _, label in rows}
    assert labels == {0, 1}
    assert all(label == 0 for text, label in rows if "true" in text)
    assert all(label == 1 for text, label in rows if "fake" in text)


def test_the_preferred_parent_wins_when_both_layouts_exist(tmp_path):
    """Fake.Br ships full_texts AND size_normalized_texts."""
    _corpus(tmp_path, "full_texts", n=2)
    _corpus(tmp_path, "size_normalized_texts", n=5)
    rows = p.parse_corpus(tmp_path, "size_normalized_texts")
    assert len(rows) == 10, "expected the 5+5 normalised set, not the 2+2 full-text one"


def test_a_missing_layout_raises_rather_than_returning_an_empty_corpus(tmp_path):
    """Loudness matters here."""
    (tmp_path / "nothing").mkdir()
    try:
        p.parse_corpus(tmp_path)
        raise AssertionError("expected FileNotFoundError for a corpus with no label directories")
    except FileNotFoundError:
        pass


def test_empty_articles_are_skipped(tmp_path):
    root = _corpus(tmp_path, "size_normalized_texts", n=2)
    (root / "size_normalized_texts" / "true" / "empty.txt").write_text("", encoding="utf-8")
    rows = p.parse_corpus(root, "size_normalized_texts")
    assert len(rows) == 4, "an empty file must not become a zero-length training row"


def test_a_directory_without_txt_files_is_not_chosen(tmp_path):
    """_find_label_dir requires actual .txt content, so an empty same-named folder cannot win."""
    (tmp_path / "decoy" / "true").mkdir(parents=True)
    (tmp_path / "decoy" / "fake").mkdir(parents=True)
    _corpus(tmp_path, "size_normalized_texts", n=2)
    assert len(p.parse_corpus(tmp_path, "size_normalized_texts")) == 4


def test_write_csv_round_trips_text_containing_commas_and_newlines(tmp_path):
    """News articles contain both."""
    out = tmp_path / "out" / "corpus.csv"
    p.write_csv([("uma, frase\ncom quebra", 1), ("outra", 0)], out)
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    body = rows[1:] if rows and rows[0][0] not in ("uma, frase\ncom quebra",) else rows
    assert any(r[0] == "uma, frase\ncom quebra" and r[1] == "1" for r in body)
