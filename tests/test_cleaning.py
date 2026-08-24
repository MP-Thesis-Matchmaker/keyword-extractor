"""Drift guard for the prepare_text copy (origin: backend-core indexing/documents.py)."""

from keyword_extractor.cleaning import prepare_text


def test_strips_html_tags():
    assert prepare_text("<p>Deep <b>learning</b> for graphs</p>") == "Deep learning for graphs"


def test_unescapes_entities():
    assert prepare_text("models &amp; methods") == "models & methods"


def test_escaped_tag_stays_text():
    # Tags are stripped BEFORE unescaping, so text about a tag survives.
    assert prepare_text("&lt;p&gt;") == "<p>"


def test_collapses_whitespace():
    assert prepare_text("a\n\n   b\t\tc  ") == "a b c"


def test_markup_only_collapses_to_empty():
    assert prepare_text("<div><br/></div>") == ""
