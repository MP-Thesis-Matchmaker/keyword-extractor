"""Text cleaning for harvested abstracts and posting descriptions.

`prepare_text` is copied verbatim from backend-core
(src/thesis_matchmaker/indexing/documents.py) rather than imported: a path
dependency for one pure function would couple the default install to the
sibling checkout. tests/test_cleaning.py pins this copy against drift.

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
