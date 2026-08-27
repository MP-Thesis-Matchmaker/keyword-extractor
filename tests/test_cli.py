"""generate end-to-end with the fake extractor and monkeypatched DB access."""

import contextlib
import json

import pytest

import keyword_extractor.cli as cli
from keyword_extractor.benchmark import read_jsonl
from keyword_extractor.fetch import PostingRow, PostingSupervisor, PublicationRow
from keyword_extractor.matcher_client import IndexNotBuiltError

_PUBS = [
    PublicationRow(
        id="20.500.14742/1",
        title="Auction theory and mechanism design",
        abstract="<p>We study auctions &amp; equilibria in large markets.</p>",
        uzh_authors=["Doe, Jane"],
        author_authority_map={"Doe, Jane": {"type": "cris", "id": "uuid-doe"}},
        year=2024,
        department="Department of Economics",
        language="eng",
    ),
    PublicationRow(
        id="20.500.14742/2",
        title="Market equilibria under uncertainty",
        abstract="Equilibria in auction markets   with noisy signals.",
        uzh_authors=["Doe, Jane"],
        author_authority_map={"Doe, Jane": {"type": "cris", "id": "uuid-doe"}},
        year=2023,
        department="Department of Economics",
        language="eng",
    ),
]

_POSTINGS = [
    PostingRow(
        id="posting-1",
        title="Thesis: reinforcement learning for robotics",
        description="Sim-to-real transfer for robot manipulation policies, evaluated with "
        "the standard benchmarks that are used in this area of research.",
        supervisors=[PostingSupervisor(name="Roe, Richard", chair="Robotics Lab")],
        department="Department of Informatics",
        language="eng",
    ),
    # Postings carry no language in the database (NULL for all 678 of them), so
    # German ones can only be recognised from their text.
    PostingRow(
        id="posting-2",
        title="Masterarbeit: Kommunikation in der lokalen Politik",
        description="In dieser Arbeit wird untersucht, wie sich die Kommunikation über "
        "lokalpolitische Entscheidungen von jener auf nationaler Ebene unterscheidet und "
        "welche Rolle die Medien dabei spielen.",
        supervisors=[PostingSupervisor(name="Udris, Linards", chair="IKMZ")],
        department="Institut für Kommunikationswissenschaft",
        language=None,
    ),
]

# German-language publications for one supervisor, plus one row that carries no
# text at all -- it must not turn up in source_ids.
_GERMAN_PUBS = [
    PublicationRow(
        id="20.500.14742/10",
        title="Lokal kompetenter? Einflüsse von Kommunikationsprozessen",
        abstract="In der Forschung wird häufig davon ausgegangen, dass Kommunikationsprozesse "
        "auf lokaler Ebene anders ablaufen als bei bundespolitischen Entscheidungen und dass "
        "die Rezipienten dabei über weniger Wissen verfügen.",
        uzh_authors=["Udris, Linards"],
        author_authority_map={"Udris, Linards": {"type": "cris", "id": "uuid-udris"}},
        year=2022,
        department="Institut für Kommunikationswissenschaft",
        language="deu",
    ),
    PublicationRow(
        id="20.500.14742/11",
        title="Medienwandel und Öffentlichkeit",
        abstract="Der Beitrag untersucht, wie sich der Medienwandel auf die Qualität der "
        "öffentlichen Kommunikation auswirkt und welche Folgen sich daraus für die "
        "Meinungsbildung ergeben.",
        uzh_authors=["Udris, Linards"],
        author_authority_map={"Udris, Linards": {"type": "cris", "id": "uuid-udris"}},
        year=2021,
        department="Institut für Kommunikationswissenschaft",
        language="deu",
    ),
    PublicationRow(
        id="20.500.14742/12-empty",
        title="",
        abstract="   ",
        uzh_authors=["Udris, Linards"],
        author_authority_map={"Udris, Linards": {"type": "cris", "id": "uuid-udris"}},
        year=2020,
        language="deu",
    ),
]

# A supervisor whose only authority is an ORCID: DSpace records an id for them,
# but it resolves to no `person` row, so nothing here may call it a CRIS UUID.
_ORCID_PUBS = [
    PublicationRow(
        id="20.500.14742/9",
        title="Retrieval-augmented generation for scientific abstracts",
        abstract="Dense retrieval over scientific abstracts with reranking.",
        uzh_authors=["Foreign, Fred"],
        author_authority_map={
            "Foreign, Fred": {"type": "orcid", "id": "0000-0002-1825-0097"}
        },
        year=2025,
        department="Department of Informatics",
        language="eng",
    ),
]

_BY_SUPERVISOR = {
    "Doe, Jane": _PUBS,
    "Udris, Linards": _GERMAN_PUBS,
    "Foreign, Fred": _ORCID_PUBS,
}


def _patch_db(monkeypatch):
    monkeypatch.setattr(cli, "connect", lambda settings: contextlib.nullcontext(None))
    monkeypatch.setattr(
        cli, "fetch_publications", lambda conn, name, **kw: _BY_SUPERVISOR.get(name, [])
    )
    monkeypatch.setattr(cli, "fetch_postings", lambda conn, **kw: _POSTINGS)
    # The manifest's corpus aggregates run on the same (absent) connection.
    monkeypatch.setattr(cli, "corpus_stats", lambda conn, **kw: {"publications": 2})
    monkeypatch.setattr(cli, "schema_fingerprint", lambda conn: "test-fingerprint")


def test_generate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    _patch_db(monkeypatch)
    out = tmp_path / "bench.jsonl"

    rc = cli.main(
        ["generate", "--out", str(out), "--supervisor-names", "Doe, Jane", "--query-style", "auto"]
    )
    assert rc == 0

    rows = read_jsonl(out)
    assert {r.source_type for r in rows} == {"publication", "posting"}

    pub_row = next(r for r in rows if r.source_type == "publication" and r.complexity == "medium")
    assert pub_row.gold_supervisors[0].name == "Doe, Jane"
    assert pub_row.gold_supervisors[0].cris_uuid == "uuid-doe"
    assert pub_row.source_ids == ["20.500.14742/1", "20.500.14742/2"]
    assert pub_row.query_style == "template"
    assert pub_row.keywords  # extraction produced something
    # Cleaning ran: no HTML fragments in extracted keywords.
    assert not any("<" in k or "&" in k for k in pub_row.keywords)

    post_row = next(r for r in rows if r.source_type == "posting")
    assert post_row.gold_supervisors[0].name == "Roe, Richard"
    assert post_row.granularity == "posting"

    # Every source yields the full complexity ladder.
    assert {r.complexity for r in rows} == {"low", "medium", "high"}

    # PII guard: no email-shaped string anywhere in the artefact.
    assert "@" not in out.read_text(encoding="utf-8")


def test_german_supervisor_gets_german_query_and_recorded_languages(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    _patch_db(monkeypatch)
    out = tmp_path / "bench_de.jsonl"

    rc = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisor-names",
            "Udris, Linards",
            "--sources",
            "publications",
        ]
    )
    assert rc == 0

    rows = read_jsonl(out)
    assert len(rows) == 3  # one supervisor, three complexity levels
    row = next(r for r in rows if r.complexity == "medium")
    assert row.metadata["text_language"] == "deu"
    assert row.metadata["db_language"] == "deu"
    assert "Ich" in row.query  # German phrasing
    # The publication with an empty title and a whitespace abstract contributed
    # no text, so it must not be cited as a source.
    assert row.source_ids == ["20.500.14742/10", "20.500.14742/11"]
    # All three levels share provenance and differ only in query and keywords.
    assert {tuple(r.source_ids) for r in rows} == {tuple(row.source_ids)}
    assert len({r.query for r in rows}) == 3
    low = next(r for r in rows if r.complexity == "low")
    assert len(low.keywords) <= 2
    assert len(low.keywords) < len(next(r for r in rows if r.complexity == "high").keywords)


def test_posting_language_is_detected_when_the_column_is_null(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    out = tmp_path / "bench_postings.jsonl"
    assert cli.main(["generate", "--out", str(out), "--sources", "postings"]) == 0

    rows = {
        (r.source_ids[0], r.complexity): r for r in read_jsonl(out)
    }
    german = rows[("posting-2", "medium")]
    assert german.metadata["text_language"] == "deu"
    assert german.metadata["db_language"] is None  # never labelled in the DB
    assert "Ich" in german.query
    english = rows[("posting-1", "medium")]
    assert english.metadata["text_language"] == "eng"
    assert "Ich" not in english.query


def test_fallback_rows_admit_they_are_template_text(tmp_path, monkeypatch, capsys):
    """A run that asks for LLM but gets 400s must not label the rows "llm"."""
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.setenv("LLM_BASE_URL", "http://unreachable.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    _patch_db(monkeypatch)
    out = tmp_path / "bench_fallback.jsonl"

    rc = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisor-names",
            "Doe, Jane",
            "--sources",
            "publications",
            "--query-style",
            "llm",
        ]
    )
    assert rc == 0
    rows = read_jsonl(out)
    assert rows and all(r.query_style == "template" for r in rows)
    assert "fell back to template phrasing" in capsys.readouterr().err


def test_language_all_is_a_synonym_for_no_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    seen = {}

    def fetch(conn, name, **kw):
        seen["language"] = kw.get("language")
        return _BY_SUPERVISOR.get(name, [])

    _patch_db(monkeypatch)
    monkeypatch.setattr(cli, "fetch_publications", fetch)
    out = tmp_path / "bench_all.jsonl"
    cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisor-names",
            "Doe, Jane",
            "--sources",
            "publications",
            "--publication-language",
            "all",
        ]
    )
    assert seen["language"] is None


def test_generate_skips_supervisor_without_pubs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    out = tmp_path / "bench.jsonl"
    rc = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisor-names",
            "Nobody, Here",
            "--sources",
            "publications",
        ]
    )
    assert rc == 0
    assert read_jsonl(out) == []
    assert "skipping" in capsys.readouterr().err


_INDEX_STATUS = {
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": 1024,
    "document_count": 215451,
    "sources": "postgres",
}


class _StubClient:
    """Stands in for MatcherClient: answers the preflight and every query."""

    base_url = "http://matcher.test"

    def __init__(self, matches=None, status=None, error=None):
        self._matches = matches or []
        self._status = status or _INDEX_STATUS
        self._error = error
        self.queries = []
        self.closed = False

    def index_status(self):
        if self._error:
            raise self._error
        return self._status

    def match(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self._matches

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _patch_matcher(monkeypatch, stub):
    monkeypatch.setattr(cli, "build_matcher_client", lambda settings, base_url=None: stub)
    return stub


def _generate_bench(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    bench = tmp_path / "bench.jsonl"
    cli.main(["generate", "--out", str(bench), "--supervisor-names", "Doe, Jane"])
    capsys.readouterr()  # discard generate output
    return bench


def test_evaluate_cli_with_injected_report(tmp_path, monkeypatch, capsys):
    bench = _generate_bench(tmp_path, monkeypatch, capsys)
    stub = _patch_matcher(
        monkeypatch,
        _StubClient(
            # Every gold supervisor in the generated benchmark, German row included.
            matches=[
                {"supervisor": "Doe, Jane"},
                {"supervisor": "Roe, Richard"},
                {"supervisor": "Udris, Linards"},
            ]
        ),
    )

    rc = cli.main(["evaluate", "--benchmark", str(bench), "--top-k", "5"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["overall"]["hit@5"] == 1.0
    assert stub.closed


def test_evaluate_report_records_which_index_answered(tmp_path, monkeypatch, capsys):
    """A score that cannot name the index that produced it is not reproducible."""
    bench = _generate_bench(tmp_path, monkeypatch, capsys)
    _patch_matcher(monkeypatch, _StubClient(matches=[{"supervisor": "Doe, Jane"}]))

    rc = cli.main(
        ["evaluate", "--benchmark", str(bench), "--label", "require_uzh_author=true"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["index"] == _INDEX_STATUS
    assert report["label"] == "require_uzh_author=true"
    # And on stderr, before any query ran.
    assert "BAAI/bge-m3" in captured.err


def test_evaluate_preflight_failure_exits_before_any_query(tmp_path, monkeypatch, capsys):
    bench = _generate_bench(tmp_path, monkeypatch, capsys)
    stub = _patch_matcher(
        monkeypatch,
        _StubClient(error=IndexNotBuiltError("the matcher has no index; themis-matcher index")),
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["evaluate", "--benchmark", str(bench)])
    assert "themis-matcher index" in str(excinfo.value)
    assert stub.queries == []
    assert stub.closed


def test_evaluate_clamps_concurrency_to_the_pool_ceiling(tmp_path, monkeypatch, capsys):
    bench = _generate_bench(tmp_path, monkeypatch, capsys)
    _patch_matcher(monkeypatch, _StubClient(matches=[{"supervisor": "Doe, Jane"}]))

    rc = cli.main(["evaluate", "--benchmark", str(bench), "--concurrency", "12"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "clamping to 4" in err


def test_posting_sample_bounds_the_run(tmp_path, monkeypatch):
    """--supervisor-sample 1 once still walked all 579 postings; it must not."""
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    out = tmp_path / "bench_one_posting.jsonl"

    rc = cli.main(
        ["generate", "--out", str(out), "--sources", "postings", "--posting-sample", "1"]
    )

    assert rc == 0
    rows = read_jsonl(out)
    assert len(rows) == 3  # one posting, three complexity levels
    assert len({r.source_ids[0] for r in rows}) == 1


def test_posting_sample_is_seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)

    def chosen(seed: str) -> set[str]:
        out = tmp_path / f"bench_{seed}.jsonl"
        cli.main(
            [
                "generate",
                "--out",
                str(out),
                "--sources",
                "postings",
                "--posting-sample",
                "1",
                "--seed",
                seed,
            ]
        )
        return {r.source_ids[0] for r in read_jsonl(out)}

    assert chosen("42") == chosen("42")


def test_progress_reaches_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    out = tmp_path / "bench_progress.jsonl"

    cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisor-names",
            "Doe, Jane",
            "--sources",
            "publications",
        ]
    )

    err = capsys.readouterr().err
    assert "publications" in err
    assert "3/3 rows" in err


def test_renamed_flags_name_their_replacement(monkeypatch):
    """A line recalled from shell history must say where the flag went."""
    _patch_db(monkeypatch)
    for old, new in cli._RENAMED_FLAGS.items():
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["generate", old, "1"])
        assert new in str(excinfo.value)


def test_orcid_only_supervisor_gets_no_cris_uuid(tmp_path, monkeypatch):
    """An ORCID is an id, but not a Person id -- the gold label must stay empty."""
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    _patch_db(monkeypatch)
    out = tmp_path / "bench.jsonl"

    rc = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--sources",
            "publications",
            "--supervisor-names",
            "Foreign, Fred",
        ]
    )
    assert rc == 0

    rows = read_jsonl(out)
    assert rows
    for row in rows:
        assert row.gold_supervisors[0].name == "Foreign, Fred"
        assert row.gold_supervisors[0].cris_uuid is None
