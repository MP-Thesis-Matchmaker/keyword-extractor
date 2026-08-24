"""SQL builders and pure helpers -- no database."""

from keyword_extractor.fetch import (
    SupervisorCandidate,
    parse_posting_supervisors,
    postings_query,
    publications_query,
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


def test_supervisor_candidates_query_without_language():
    sql, _ = supervisor_candidates_query(min_publications=1, language=None)
    assert "%(language)s" not in sql


def test_publications_query_predicates():
    sql, params = publications_query(
        supervisor="Doe, Jane", language="eng", require_abstract=True, max_pubs=20
    )
    assert "%(supervisor)s = ANY(uzh_authors)" in sql
    assert "abstract IS NOT NULL" in sql
    assert "ORDER BY year DESC NULLS LAST, id" in sql
    assert "LIMIT %(max_pubs)s" in sql
    assert params["supervisor"] == "Doe, Jane"


def test_publications_query_unlimited():
    sql, _ = publications_query(
        supervisor="Doe, Jane", language=None, require_abstract=False, max_pubs=None
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
