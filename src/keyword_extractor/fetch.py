"""Read-only access to the matchmaker Postgres: supervisors, publications, postings.

Raw SQL over psycopg3, matching backend-core's no-ORM idiom. SQL is kept in
module-level builder functions returning (sql, params) so tests can assert on
the predicates without a database.

Data-protection rule inherited from backend-core (contracts/sources.py):
supervisor emails travel no further than the row they arrive on. `PostingRow`
drops them at parse time, so nothing downstream can leak one into a benchmark
artefact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import psycopg
from pydantic import BaseModel

from keyword_extractor.config import Settings

_CONNECT_TIMEOUT = 5


def connect(settings: Settings) -> psycopg.Connection:
    """Open a read-only-use connection to the matchmaker database."""
    return psycopg.connect(settings.database_url, connect_timeout=_CONNECT_TIMEOUT)


class PublicationRow(BaseModel):
    """One publication, as read from the `publication` table."""

    id: str
    title: str
    abstract: str | None = None
    uzh_authors: list[str] = []
    author_authority_map: dict[str, str | None] = {}
    year: int | None = None
    department: str | None = None
    language: str | None = None
    keywords: list[str] = []


class PostingSupervisor(BaseModel):
    """A posting's supervisor with the email already stripped."""

    name: str
    chair: str | None = None


class PostingRow(BaseModel):
    """One open thesis posting, as read from the `posting` table."""

    id: str
    title: str
    description: str | None = None
    supervisors: list[PostingSupervisor] = []
    faculty: str | None = None
    department: str | None = None
    language: str | None = None
    keywords: list[str] = []


@dataclass(frozen=True)
class SupervisorCandidate:
    """A supervisor-eligible author with enough publications to sample from."""

    name: str
    cris_uuid: str | None
    publication_count: int


def supervisor_candidates_query(
    *, min_publications: int, language: str | None
) -> tuple[str, dict[str, Any]]:
    """Aggregate uzh_authors into per-name counts with the best-known CRIS UUID.

    max() over the authority-map lookup: the same name maps to its UUID on some
    rows and to null on others, and max() over text ignores NULLs, so any row
    that knows the UUID wins.
    """
    language_clause = "AND p.language = %(language)s" if language else ""
    sql = f"""
        SELECT name,
               max(p.author_authority_map ->> name) AS cris_uuid,
               count(*) AS publication_count
        FROM publication p, unnest(p.uzh_authors) AS name
        WHERE p.abstract IS NOT NULL AND p.abstract <> ''
          {language_clause}
        GROUP BY name
        HAVING count(*) >= %(min_publications)s
        ORDER BY name
    """
    return sql, {"min_publications": min_publications, "language": language}


def publications_query(
    *, supervisor: str, language: str | None, require_abstract: bool, max_pubs: int | None
) -> tuple[str, dict[str, Any]]:
    """Publications for one supervisor name, most recent first.

    `= ANY(uzh_authors)` hits the GIN index backend-core created for exactly
    this lookup; `ORDER BY year DESC NULLS LAST, id` caps to the most recent
    work with a deterministic tiebreak (backend-core's sources.py precedent).
    """
    abstract_clause = "AND abstract IS NOT NULL AND abstract <> ''" if require_abstract else ""
    language_clause = "AND language = %(language)s" if language else ""
    limit_clause = "LIMIT %(max_pubs)s" if max_pubs else ""
    sql = f"""
        SELECT id, title, abstract, uzh_authors, author_authority_map,
               year, department, language, keywords
        FROM publication
        WHERE %(supervisor)s = ANY(uzh_authors)
          {abstract_clause}
          {language_clause}
        ORDER BY year DESC NULLS LAST, id
        {limit_clause}
    """
    return sql, {"supervisor": supervisor, "language": language, "max_pubs": max_pubs}


def postings_query(*, language: str | None) -> tuple[str, dict[str, Any]]:
    """Open postings, mirroring backend-core's read filter for visible postings."""
    language_clause = "AND language = %(language)s" if language else ""
    sql = f"""
        SELECT id, title, description, supervisors, faculty, department, language, keywords
        FROM posting
        WHERE (status IS NULL OR status NOT IN ('assigned', 'private'))
          {language_clause}
        ORDER BY id
    """
    return sql, {"language": language}


def list_supervisor_candidates(
    conn: psycopg.Connection,
    *,
    min_publications: int = 3,
    language: str | None = "eng",
    require_cris_uuid: bool = True,
) -> list[SupervisorCandidate]:
    """Supervisor-eligible authors with at least `min_publications` abstracts.

    `require_cris_uuid` drops names whose authority entry is null everywhere --
    the weakest attributions, and a partial mitigation of the known ORCID
    conflation bug that inflates `uzh_authors` (see backend-core zora/README.md).
    """
    sql, params = supervisor_candidates_query(
        min_publications=min_publications, language=language
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    candidates = [
        SupervisorCandidate(name=name, cris_uuid=cris_uuid, publication_count=count)
        for name, cris_uuid, count in rows
    ]
    if require_cris_uuid:
        candidates = [c for c in candidates if c.cris_uuid]
    return candidates


def sample_supervisors(
    candidates: list[SupervisorCandidate], n: int, seed: int = 42
) -> list[SupervisorCandidate]:
    """Reproducible sample of n candidates; n <= 0 or n >= len means all."""
    if n <= 0 or n >= len(candidates):
        return list(candidates)
    ordered = sorted(candidates, key=lambda c: c.name)
    return random.Random(seed).sample(ordered, n)


def fetch_publications(
    conn: psycopg.Connection,
    supervisor: str,
    *,
    language: str | None = "eng",
    require_abstract: bool = True,
    max_pubs: int | None = 20,
) -> list[PublicationRow]:
    """Publications naming `supervisor` in uzh_authors, most recent first."""
    sql, params = publications_query(
        supervisor=supervisor,
        language=language,
        require_abstract=require_abstract,
        max_pubs=max_pubs,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        PublicationRow(
            id=r[0],
            title=r[1] or "",
            abstract=r[2],
            uzh_authors=r[3] or [],
            author_authority_map=r[4] or {},
            year=r[5],
            department=r[6],
            language=r[7],
            keywords=r[8] or [],
        )
        for r in rows
    ]


def parse_posting_supervisors(raw: list[dict[str, Any]] | None) -> list[PostingSupervisor]:
    """Keep name and chair; the email is dropped here and never re-enters."""
    supervisors = []
    for entry in raw or []:
        name = (entry.get("name") or "").strip()
        if name:
            supervisors.append(PostingSupervisor(name=name, chair=entry.get("chair")))
    return supervisors


def fetch_postings(
    conn: psycopg.Connection,
    *,
    require_supervisor: bool = True,
    language: str | None = None,
) -> list[PostingRow]:
    """Visible postings; by default only those naming at least one supervisor.

    Postings without a named supervisor cannot be gold-labelled (and are
    invisible to person-level retrieval anyway), so they are skipped.
    """
    sql, params = postings_query(language=language)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    postings = [
        PostingRow(
            id=r[0],
            title=r[1] or "",
            description=r[2],
            supervisors=parse_posting_supervisors(r[3]),
            faculty=r[4],
            department=r[5],
            language=r[6],
            keywords=r[7] or [],
        )
        for r in rows
    ]
    if require_supervisor:
        postings = [p for p in postings if p.supervisors]
    return postings
