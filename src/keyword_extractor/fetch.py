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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import psycopg
from pydantic import BaseModel, ConfigDict

from keyword_extractor.config import Settings

_CONNECT_TIMEOUT = 5

_T = TypeVar("_T")


def connect(settings: Settings) -> psycopg.Connection:
    """Open a read-only-use connection to the matchmaker database."""
    return psycopg.connect(settings.database_url, connect_timeout=_CONNECT_TIMEOUT)


class AuthorAuthority(BaseModel):
    """One entry of `publication.author_authority_map`: a typed identity, not a bare id.

    Two kinds exist upstream (backend-core schema.sql, the `publication` table):
    `{"type": "cris", "id": "<Person item UUID>"}`, which resolves in `person.uuid`
    and marks a registered UZH researcher, and `{"type": "orcid", "id": "<bare
    ORCID>"}`, which does not resolve to a Person and carries no affiliation. The
    distinction is the whole basis of supervisor eligibility, so it is modelled
    rather than flattened: reading only the id would silently equate the two.

    `extra="ignore"` so a field added upstream widens the contract instead of
    breaking a run.
    """

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    id: str | None = None


CRIS = "cris"


def authority_id(authority_map: dict[str, AuthorAuthority | None], name: str) -> str | None:
    """The authority id recorded for `name`, or None if it has none."""
    entry = authority_map.get(name)
    return entry.id if entry else None


def authority_type(authority_map: dict[str, AuthorAuthority | None], name: str) -> str | None:
    """The authority kind recorded for `name` ("cris", "orcid"), or None."""
    entry = authority_map.get(name)
    return entry.type if entry else None


def cris_uuid_for(authority_map: dict[str, AuthorAuthority | None], name: str) -> str | None:
    """The CRIS Person UUID for `name`, or None when the authority is not CRIS-typed.

    The only correct way to read a UUID out of the map: an orcid-typed entry has
    an `id` too, and returning it here would put an ORCID in a field called
    `cris_uuid`.
    """
    entry = authority_map.get(name)
    if entry is None or entry.type != CRIS:
        return None
    return entry.id


class PublicationRow(BaseModel):
    """One publication, as read from the `publication` table."""

    id: str
    title: str
    abstract: str | None = None
    uzh_authors: list[str] = []
    author_authority_map: dict[str, AuthorAuthority | None] = {}
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
    authority_type: str | None = None


@dataclass(frozen=True)
class SupervisorPool:
    """The candidate list plus the counts that show which attribution regime produced it.

    `total` and `cris_typed` are equal on the corpus as it stands (1,871 of 1,871
    at min_publications=3), because backend-core now admits only CRIS-typed
    authorities into `uzh_authors`. Reporting both anyway costs nothing and turns
    a silent tautology into a visible invariant: the day they diverge, the
    eligibility rule upstream has moved and every gold label drawn here is
    suspect.
    """

    candidates: list[SupervisorCandidate]
    total: int
    cris_typed: int


def supervisor_candidates_query(
    *, min_publications: int, language: str | None
) -> tuple[str, dict[str, Any]]:
    """Aggregate uzh_authors into per-name counts with the best-known authority.

    `-> name ->> \'id\'`, not `->> name`: the map\'s values are typed objects
    (`{"type": "cris", "id": "..."}`), so `->>` on the map itself returns the
    whole object serialized as text rather than the id inside it. Reading it
    that way put a JSON blob in a column called cris_uuid.

    max() over each lookup: the same name maps to its authority on some rows and
    to null on others, and max() over text ignores NULLs, so any row that knows
    the authority wins. Type and id are aggregated separately, which is safe
    because a name carries at most one authority across the corpus -- and if
    that ever stopped holding, `require_cris_uuid` below is what would notice.
    """
    language_clause = "AND p.language = %(language)s" if language else ""
    sql = f"""
        SELECT name,
               max(p.author_authority_map -> name ->> 'id') AS authority_id,
               max(p.author_authority_map -> name ->> 'type') AS authority_type,
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
) -> SupervisorPool:
    """Supervisor-eligible authors with at least `min_publications` abstracts.

    `require_cris_uuid` keeps only names whose authority is CRIS-typed, i.e. that
    resolve to a `person` row. It used to be a mitigation: backend-core admitted
    any DSpace authority into `uzh_authors`, so orcid-typed co-authors at other
    universities became supervisor candidates. That is fixed upstream, and the
    filter now drops nothing. It stays because it is the cheapest available
    monitor of the eligibility rule -- see `SupervisorPool` for why a drop of
    zero is worth printing.
    """
    sql, params = supervisor_candidates_query(
        min_publications=min_publications, language=language
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    candidates = [
        SupervisorCandidate(
            name=name,
            # Never the raw id: an orcid-typed entry has one too.
            cris_uuid=auth_id if auth_type == CRIS else None,
            publication_count=count,
            authority_type=auth_type,
        )
        for name, auth_id, auth_type, count in rows
    ]
    cris_typed = sum(1 for c in candidates if c.cris_uuid)
    if require_cris_uuid:
        candidates = [c for c in candidates if c.cris_uuid]
    return SupervisorPool(candidates=candidates, total=len(rows), cris_typed=cris_typed)


def _reproducible_sample(
    items: list[_T], n: int, *, key: Callable[[_T], str], seed: int
) -> list[_T]:
    """Seeded sample of n items; n <= 0 or n >= len means all.

    Sorted by `key` before sampling, so which items are drawn depends on the
    seed and the set, never on the order the database happened to return.
    """
    if n <= 0 or n >= len(items):
        return list(items)
    ordered = sorted(items, key=key)
    return random.Random(seed).sample(ordered, n)


def sample_supervisors(
    candidates: list[SupervisorCandidate], n: int, seed: int = 42
) -> list[SupervisorCandidate]:
    """Reproducible sample of n supervisor candidates; 0 means all."""
    return _reproducible_sample(candidates, n, key=lambda c: c.name, seed=seed)


def sample_postings(postings: list[PostingRow], n: int, seed: int = 42) -> list[PostingRow]:
    """Reproducible sample of n postings; 0 means all.

    A posting is a benchmark row in its own right -- unlike publications it is
    not pooled under a supervisor -- so this is the only lever bounding how
    much of the corpus a run touches. Without it `--supervisor-sample 1` still
    walked all 579 postings.
    """
    return _reproducible_sample(postings, n, key=lambda p: p.id, seed=seed)


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


def corpus_stats(conn: psycopg.Connection, *, min_publications: int) -> dict[str, int]:
    """Counts describing the corpus a benchmark run drew its gold labels from.

    Cheap aggregates, run once per generate. The pair that matters is
    `publications_with_uzh_authors` against `publications`: it is the attribution
    regime expressed as a number, and the reason the previous benchmark went
    stale is that nobody had recorded it.
    """
    queries = {
        "publications": "SELECT count(*) FROM publication",
        "publications_with_uzh_authors": (
            "SELECT count(*) FROM publication WHERE cardinality(uzh_authors) > 0"
        ),
        "distinct_uzh_author_names": (
            "SELECT count(DISTINCT n) FROM publication p, unnest(p.uzh_authors) n"
        ),
        "postings": "SELECT count(*) FROM posting",
    }
    stats: dict[str, int] = {}
    with conn.cursor() as cur:
        for label, sql in queries.items():
            cur.execute(sql)
            stats[label] = cur.fetchone()[0]
    return stats


def schema_fingerprint(conn: psycopg.Connection) -> str | None:
    """backend-core's schema_version fingerprint, or None if the table is absent.

    One string pinning the migration state of the database the gold labels came
    from. A diff of two manifests answers "did the ground truth move?".
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fingerprint FROM schema_version WHERE id = 1")
            row = cur.fetchone()
    except psycopg.Error:
        conn.rollback()
        return None
    return row[0] if row else None
