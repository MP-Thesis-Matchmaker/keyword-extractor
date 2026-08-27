"""Text cleaning for harvested abstracts and posting descriptions.

`prepare_text` is copied verbatim from backend-core
(projects/matcher/src/themis_matcher/indexing/documents.py) rather than
imported: a path dependency for one pure function would couple the default
install to the sibling checkout. That reasoning has since become absolute --
there is no path dependency on backend-core left at all, since evaluation moved
to the matcher's HTTP API -- so this copy is the only thing keeping the two
repos' text preparation identical. tests/test_cleaning.py pins it against
drift, which matters: keywords extracted from differently-cleaned text are not
comparable with the documents the matcher indexed.

Measured over the harvested corpus there: 329 abstracts carry HTML tags, 579
carry entities, 2,300 carry runs of three or more whitespace characters --
uncleaned, those pollute KeyBERT's candidate n-grams.
"""

from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def prepare_text(value: str) -> str:
    """Strip markup and collapse whitespace ahead of keyword extraction.

    Tags go before entities are unescaped, so an escaped `&lt;p&gt;` -- which is
    text *about* a tag -- cannot be turned into a real tag by the unescape and
    then stripped as one. (Ordering note preserved from the original.)
    """
    return _WHITESPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()
