"""SQL builders and pure helpers -- no database."""

from keyword_extractor.fetch import (
    AuthorAuthority,
    PostingRow,
    PostingSupervisor,
    SupervisorCandidate,
    cris_uuid_for,
    list_supervisor_candidates,
    parse_posting_supervisors,
    postings_query,
    publications_query,
    sample_postings,
    sample_supervisors,
    supervisor_candidates_query,
)


def test_supervisor_candidates_query_predicates():
    sql, params = supervisor_candidates_query(min_publications=3, language="eng")
    assert "unnest(p.uzh_authors)" in sql
    assert "abstract IS NOT NULL" in sql
    assert "p.language = %(language)s" in sql
    assert "count(*) >= %(min_publications)s" in sql
    assert params == {"min_publications": 3, "language": "eng"}


def test_supervisor_candidates_query_reads_into_the_typed_authority():
    """The map holds objects, so the id has to be read from inside the value.

    The regression this pins: `->> name` returns the whole
    `{"type": ..., "id": ...}` object serialized as text, which looks like a
    populated column and is not a UUID.
    """
    sql, _ = supervisor_candidates_query(min_publications=1, language=None)
    assert "p.author_authority_map -> name ->> 'id'" in sql
    assert "p.author_authority_map -> name ->> 'type'" in sql
    assert "author_authority_map ->> name" not in sql


def test_supervisor_candidates_query_without_language():
    sql, _ = supervisor_candidates_query(min_publications=1, language=None)
    assert "%(language)s" not in sql


def test_publications_query_predicates():
    sql, params = publications_query(
        supervisor_names=["Doe, Jane"], language="eng", require_abstract=True, max_pubs=20
    )
    assert "uzh_authors && %(supervisor_names)s" in sql
    assert "abstract IS NOT NULL" in sql
    assert "ORDER BY year DESC NULLS LAST, id" in sql
    assert "LIMIT %(max_pubs)s" in sql
    assert params["supervisor_names"] == ["Doe, Jane"]


def test_publications_query_unlimited():
    sql, _ = publications_query(
        supervisor_names=["Doe, Jane"], language=None, require_abstract=False, max_pubs=None
    )
    assert "LIMIT" not in sql
    assert "abstract IS NOT NULL" not in sql


def test_postings_query_mirrors_backend_visibility_filter():
    sql, _ = postings_query(language=None)
    assert "status IS NULL OR status NOT IN ('assigned', 'private')" in sql


def test_parse_posting_supervisors_drops_email():
    parsed = parse_posting_supervisors(
        [
            {"name": "Doe, Jane", "email": "x@example.org", "chair": "AI Lab"},
            {"name": "  ", "email": "y@example.org"},
        ]
    )
    assert len(parsed) == 1
    assert parsed[0].name == "Doe, Jane"
    assert parsed[0].chair == "AI Lab"
    assert "email" not in parsed[0].model_dump()


def test_sample_supervisors_reproducible_and_order_independent():
    candidates = [
        SupervisorCandidate(name=f"Name, {i}", cris_uuid=str(i), publication_count=i)
        for i in range(20)
    ]
    a = sample_supervisors(candidates, 5, seed=7)
    b = sample_supervisors(list(reversed(candidates)), 5, seed=7)
    assert a == b
    assert len(a) == 5


def test_sample_supervisors_zero_means_all():
    candidates = [SupervisorCandidate(name="A", cris_uuid="1", publication_count=1)]
    assert sample_supervisors(candidates, 0) == candidates


def _postings(n: int) -> list[PostingRow]:
    return [
        PostingRow(
            id=f"posting-{i}",
            title=f"Thesis {i}",
            supervisors=[PostingSupervisor(name=f"Name, {i}")],
        )
        for i in range(n)
    ]


def test_sample_postings_reproducible_and_order_independent():
    postings = _postings(20)
    a = sample_postings(postings, 5, seed=7)
    b = sample_postings(list(reversed(postings)), 5, seed=7)
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == 5


def test_sample_postings_zero_means_all():
    """0 is how a run deliberately takes the whole 579-posting corpus."""
    postings = _postings(3)
    assert sample_postings(postings, 0) == postings
    assert sample_postings(postings, 99) == postings


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.sql, self.params = sql, params

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


_ROWS = [
    ("Doe, Jane", "uuid-doe", "cris", 7),
    ("Foreign, Fred", "0000-0002-1825-0097", "orcid", 4),
    ("Nameless, Nick", None, None, 3),
]


def test_list_supervisor_candidates_drops_non_cris_authority():
    pool = list_supervisor_candidates(_FakeConn(_ROWS), require_cris_uuid=True)
    assert [c.name for c in pool.candidates] == ["Doe, Jane"]
    assert pool.total == 3
    assert pool.cris_typed == 1


def test_list_supervisor_candidates_never_reports_an_orcid_as_a_cris_uuid():
    """An orcid-typed entry carries an id too; it must not reach `cris_uuid`."""
    pool = list_supervisor_candidates(_FakeConn(_ROWS), require_cris_uuid=False)
    by_name = {c.name: c for c in pool.candidates}
    assert by_name["Foreign, Fred"].cris_uuid is None
    assert by_name["Foreign, Fred"].authority_type == "orcid"
    assert by_name["Doe, Jane"].cris_uuid == "uuid-doe"


def test_cris_uuid_for_ignores_orcid_and_missing_entries():
    authority_map = {
        "Doe, Jane": AuthorAuthority(type="cris", id="uuid-doe"),
        "Foreign, Fred": AuthorAuthority(type="orcid", id="0000-0002-1825-0097"),
        "Nameless, Nick": None,
    }
    assert cris_uuid_for(authority_map, "Doe, Jane") == "uuid-doe"
    assert cris_uuid_for(authority_map, "Foreign, Fred") is None
    assert cris_uuid_for(authority_map, "Nameless, Nick") is None
    assert cris_uuid_for(authority_map, "Absent, Alice") is None
