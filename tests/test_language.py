"""Language detection and KeyBERT stop words -- no sklearn, no model, no DB."""

import builtins

from keyword_extractor.language import (
    DETECTOR_MARKERS,
    ENGLISH_FUNCTION_WORDS,
    GERMAN_FUNCTION_WORDS,
    detect_language,
    keybert_stop_words,
    language_scores,
)

GERMAN = (
    "In der Forschung wird häufig davon ausgegangen, dass Kommunikationsprozesse auf "
    "lokaler Ebene anders ablaufen als bei bundespolitischen Entscheidungen und dass "
    "die Rezipienten dabei über weniger Wissen verfügen."
)
ENGLISH = (
    "This study evaluated whether air blasting or rinsing particle remnants with water "
    "would impair the adhesion of resin composite to metal, and how the results compare "
    "with the untreated control group."
)
FRENCH = (
    "Cette étude porte sur les processus de communication dans les organisations et sur "
    "la manière dont les acteurs qui participent aux décisions se représentent leur rôle "
    "au sein de ces institutions."
)
ITALIAN = (
    "Questo studio analizza i processi di comunicazione nelle organizzazioni e il modo in "
    "cui gli attori che partecipano alle decisioni interpretano il loro ruolo all'interno "
    "di queste istituzioni."
)


def test_detects_german():
    assert detect_language(GERMAN) == "deu"


def test_detects_english():
    assert detect_language(ENGLISH) == "eng"


def test_french_is_not_mistaken_for_german():
    # The naive two-list version called 74 of 255 French abstracts German.
    assert detect_language(FRENCH) != "deu"


def test_italian_is_not_mistaken_for_german():
    assert detect_language(ITALIAN) != "deu"


def test_short_text_is_undecidable():
    assert detect_language("Auction theory") is None
    assert detect_language("") is None


def test_ambiguous_tokens_removed_from_marker_sets():
    # "des" is German and French; "in" is German, English and Italian. A token
    # carried by more than one language decides nothing and must not be scored.
    for token in ("des", "in"):
        holders = [lang for lang, words in DETECTOR_MARKERS.items() if token in words]
        assert holders == [], f"{token!r} still scores for {holders}"


def test_marker_sets_keep_distinctive_tokens():
    assert "und" in DETECTOR_MARKERS["deu"]
    assert "für" in DETECTOR_MARKERS["deu"]
    assert "the" in DETECTOR_MARKERS["eng"]


def test_language_scores_reports_every_language():
    scores = language_scores(GERMAN)
    assert set(scores) == {"deu", "eng", "fra", "ita"}
    assert scores["deu"] > scores["eng"]


def test_keybert_stop_words_covers_both_languages():
    words = keybert_stop_words()
    assert "the" in words
    assert "und" in words
    assert words == sorted(words)  # deterministic ordering


def test_keybert_stop_words_without_sklearn(monkeypatch):
    """The base install has no sklearn; the curated fallback must still work."""
    real_import = builtins.__import__

    def fail_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn in the base install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_sklearn)
    words = set(keybert_stop_words())
    assert words == set(ENGLISH_FUNCTION_WORDS | GERMAN_FUNCTION_WORDS)
