"""Function-word lists, KeyBERT stop words, and text language detection.

Two jobs, one vocabulary:

* **Stop words** for KeyBERT use the raw curated lists. Here a word shared
  between languages (`des`, `in`) is still worth stripping from candidate
  keyphrases, so ambiguity is harmless.
* **Detection** uses `DETECTOR_MARKERS`, where every token appearing in more
  than one list has been removed. Ambiguity is the whole failure mode there: a
  two-list German/English detector scored 74 of 255 French abstracts as German,
  because `des`, `en` and `in` are French too.

Detection is needed rather than optional because `posting.language` is NULL for
every posting in the corpus, and because `publication.language` describes the
record and not the abstract -- of 300 `deu`-labelled publications, 87 carry a
fully English abstract. The text is the only honest signal.
"""

from __future__ import annotations

import re

# Tokens must keep their diacritics: umlauts carry much of the German signal,
# and accents much of the French.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# A winner needs this many hits before the call is made at all; short titles
# simply do not carry enough function words to decide on.
_MIN_HITS = 4
# ...and it has to beat the runner-up by this factor. Below it, the text is
# treated as undecidable rather than guessed at.
_MIN_MARGIN = 1.3

GERMAN_FUNCTION_WORDS = frozenset(
    """
    der die das den dem des ein eine einer eines einem einen kein keine keinen
    und oder aber sondern denn also somit daher deshalb darum jedoch zwar
    nicht nichts etwas jemand niemand man wir sie ich es ihm ihn uns euch
    mit von zu im in am an auf für über durch bei aus nach vor als wie um
    dass sich werden wurde wurden wird worden sein ist sind war waren
    haben hat hatte hatten kann können konnte soll sollen sollte will wollen
    muss müssen möchte würde würden dürfen mögen
    diese dieser dieses diesen dies welche welcher welches jener jene
    auch nur noch schon sowie sowohl zwischen wegen ohne gegen bis dann wenn
    weil damit dabei dazu während seit obwohl falls sofern indem wobei
    darauf davon darin dafür dagegen darüber hierbei
    ihre ihrer ihren seine seiner seinen unser unsere mein meine dein deine
    unter über mehr sehr zum zur beim vom etwa bereits erst nun eben gerade
    nämlich ferner zudem außerdem insbesondere beispielsweise
    allen alle aller alles beide beiden viele vielen wenige jeder jede jedes
    andere anderen anderer erste ersten zweite neue neuen
    groß große großen klein kleine hier dort heute jetzt immer nie oft meist
    """.split()
)

# Fallback for `keybert_stop_words` when scikit-learn is not installed (it
# arrives transitively with keybert, i.e. only under the `extraction` extra).
ENGLISH_FUNCTION_WORDS = frozenset(
    """
    the a an and or but not with of to in on for from by at as is are was were
    be been being this that these those it its which who whom whose
    have has had do does did can could should would will may might must
    we our us they their them there here you your he she his her him
    when where why how what while than then if because so such between
    during through over under above below both more most other others some
    only own same too very about against each any all no nor
    into out up down off again further once few many much
    itself themselves however therefore thus hence moreover furthermore
    """.split()
)

# Present only so the detector can recognise French and Italian and decline,
# rather than mistaking them for German. Not used as stop words.
FRENCH_FUNCTION_WORDS = frozenset(
    """
    le la les un une des du de et ou mais ne pas plus très aussi comme
    avec pour dans sur par au aux chez sans contre entre depuis selon
    ce cette ces cet celui celle qui que quoi dont où
    est sont était étaient être avoir ont été
    nous vous ils elles leur leurs son sa ses notre nos votre vos
    ainsi donc alors lors même autre autres tout tous toute toutes
    """.split()
)

ITALIAN_FUNCTION_WORDS = frozenset(
    """
    il lo la i gli le un uno una di del della dei delle degli
    e ed o ma non con per in su da al allo alla ai alle dal dalla nel nella
    che chi cui è sono era erano essere avere ha hanno stato stati
    più molto anche come tra fra senza contro
    noi voi loro suo sua suoi sue nostro nostra questo questa questi queste
    quel quella quello così quindi allora dopo prima durante secondo
    stesso stessa altro altri tutto tutti tutte
    """.split()
)

_RAW_LISTS: dict[str, frozenset[str]] = {
    "deu": GERMAN_FUNCTION_WORDS,
    "eng": ENGLISH_FUNCTION_WORDS,
    "fra": FRENCH_FUNCTION_WORDS,
    "ita": ITALIAN_FUNCTION_WORDS,
}


def _distinctive(lists: dict[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    """Keep only tokens unique to one language; shared tokens decide nothing."""
    return {
        language: frozenset(
            word
            for word in words
            if not any(word in other for name, other in lists.items() if name != language)
        )
        for language, words in lists.items()
    }


#: Per-language marker sets with cross-language tokens removed.
DETECTOR_MARKERS: dict[str, frozenset[str]] = _distinctive(_RAW_LISTS)

#: Languages this module is willing to name. A French or Italian winner still
#: resolves to None, because callers only branch on German.
_REPORTABLE = ("deu", "eng")


def keybert_stop_words() -> list[str]:
    """English + German stop words for KeyBERT's candidate vectorizer.

    scikit-learn's English list (318 entries) is used when available -- it is
    considerably better than the curated fallback -- unioned with the German
    list, since English and German together are over 99% of the corpus. Sorted
    so a run is reproducible. Rarer languages keep their function words as
    candidates; MMR filters most of them in practice.
    """
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    except ImportError:  # base install: no keybert, hence no scikit-learn
        english: frozenset[str] = ENGLISH_FUNCTION_WORDS
    else:
        english = frozenset(ENGLISH_STOP_WORDS)
    return sorted(english | GERMAN_FUNCTION_WORDS)


def language_scores(text: str) -> dict[str, int]:
    """Marker hits per language. Exposed for debugging a detection call."""
    words = [word.lower() for word in _TOKEN_RE.findall(text)]
    return {
        language: sum(word in markers for word in words)
        for language, markers in DETECTOR_MARKERS.items()
    }


def detect_language(text: str) -> str | None:
    """`"deu"`, `"eng"`, or None when the text does not clearly say.

    None means "no strong signal, or a language we do not phrase queries in";
    callers fall back to English phrasing. Deliberately conservative: a wrong
    call only picks the wrong query template, but a benchmark full of confidently
    mislabelled rows would be worse than one that admits uncertainty.
    """
    if not text:
        return None
    scores = language_scores(text)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    (winner, top), (_, runner_up) = ranked[0], ranked[1]
    if top < _MIN_HITS or top < runner_up * _MIN_MARGIN:
        return None
    return winner if winner in _REPORTABLE else None
