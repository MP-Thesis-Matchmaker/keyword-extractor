"""generate end-to-end with the fake extractor and monkeypatched DB access."""

import contextlib
import json

import keyword_extractor.cli as cli
from keyword_extractor.benchmark import read_jsonl
from keyword_extractor.fetch import PostingRow, PostingSupervisor, PublicationRow

_PUBS = [
    PublicationRow(
        id="20.500.14742/1",
        title="Auction theory and mechanism design",
        abstract="<p>We study auctions &amp; equilibria in large markets.</p>",
        uzh_authors=["Doe, Jane"],
        author_authority_map={"Doe, Jane": "uuid-doe"},
        year=2024,
        department="Department of Economics",
        language="eng",
    ),
    PublicationRow(
        id="20.500.14742/2",
        title="Market equilibria under uncertainty",
        abstract="Equilibria in auction markets   with noisy signals.",
        uzh_authors=["Doe, Jane"],
        author_authority_map={"Doe, Jane": "uuid-doe"},
        year=2023,
        department="Department of Economics",
        language="eng",
    ),
]

_POSTINGS = [
    PostingRow(
        id="posting-1",
        title="Thesis: reinforcement learning for robotics",
        description="Sim-to-real transfer for robot manipulation policies.",
        supervisors=[PostingSupervisor(name="Roe, Richard", chair="Robotics Lab")],
        department="Department of Informatics",
        language="eng",
    )
]


def _patch_db(monkeypatch):
    monkeypatch.setattr(cli, "connect", lambda settings: contextlib.nullcontext(None))
    monkeypatch.setattr(
        cli, "fetch_publications", lambda conn, name, **kw: _PUBS if name == "Doe, Jane" else []
    )
    monkeypatch.setattr(cli, "fetch_postings", lambda conn, **kw: _POSTINGS)


def test_generate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    _patch_db(monkeypatch)
    out = tmp_path / "bench.jsonl"

    rc = cli.main(
        ["generate", "--out", str(out), "--supervisors", "Doe, Jane", "--query-style", "auto"]
    )
    assert rc == 0

    rows = read_jsonl(out)
    assert {r.source_type for r in rows} == {"publication", "posting"}

    pub_row = next(r for r in rows if r.source_type == "publication")
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

    # PII guard: no email-shaped string anywhere in the artefact.
    assert "@" not in out.read_text(encoding="utf-8")


def test_generate_skips_supervisor_without_pubs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    out = tmp_path / "bench.jsonl"
    rc = cli.main(
        [
            "generate",
            "--out",
            str(out),
            "--supervisors",
            "Nobody, Here",
            "--sources",
            "publications",
        ]
    )
    assert rc == 0
    assert read_jsonl(out) == []
    assert "skipping" in capsys.readouterr().err


def test_evaluate_cli_with_injected_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEYBERT_MODEL", "fake")
    _patch_db(monkeypatch)
    bench = tmp_path / "bench.jsonl"
    cli.main(["generate", "--out", str(bench), "--supervisors", "Doe, Jane"])
    capsys.readouterr()  # discard generate output

    import keyword_extractor.evaluate as evaluate_module

    def fake_find(query, top_k):
        return [{"supervisor": "Doe, Jane"}, {"supervisor": "Roe, Richard"}]

    monkeypatch.setattr(evaluate_module, "_default_find", lambda: fake_find)
    rc = cli.main(["evaluate", "--benchmark", str(bench), "--top-k", "5"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["overall"]["hit@5"] == 1.0
