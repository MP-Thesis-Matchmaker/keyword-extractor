"""JSONL round-trip and schema stability."""

import json

from keyword_extractor.benchmark import BenchmarkRow, GoldSupervisor, read_jsonl, write_jsonl


def _row() -> BenchmarkRow:
    return BenchmarkRow(
        query="I am looking for a thesis about auction theory.",
        keywords=["auction theory", "mechanism design"],
        source_type="publication",
        source_ids=["20.500.14742/1", "20.500.14742/2"],
        gold_supervisors=[GoldSupervisor(name="Doe, Jane", cris_uuid="abc-123")],
        granularity="supervisor",
        query_style="template",
        metadata={"department": "Department of Economics", "seed": 42},
    )


def test_roundtrip(tmp_path):
    path = tmp_path / "bench.jsonl"
    count = write_jsonl([_row(), _row()], path)
    assert count == 2
    rows = read_jsonl(path)
    assert rows == [_row(), _row()]


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(_row().model_dump_json() + "\n\n\n", encoding="utf-8")
    assert len(read_jsonl(path)) == 1


def test_rows_written_before_complexity_existed_still_load(tmp_path):
    """Benchmark files committed before the complexity ladder must stay readable."""
    legacy = _row().model_dump()
    del legacy["complexity"]
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    [row] = read_jsonl(path)
    assert row.complexity == "medium"


def test_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "bench.jsonl"
    write_jsonl([_row()], path)
    assert path.exists()


def test_manifest_round_trip(tmp_path):
    from keyword_extractor.benchmark import (
        BenchmarkManifest,
        manifest_path,
        read_manifest,
        utc_now,
        write_manifest,
    )

    bench = tmp_path / "benchmark_v3.jsonl"
    manifest = BenchmarkManifest(
        generated_at=utc_now(),
        schema_fingerprint="032028c3e280",
        corpus={"publications": 214756, "publications_with_uzh_authors": 53545},
        index={"embedding_model": "BAAI/bge-m3", "document_count": 215451},
        run={"seed": 42},
    )
    write_manifest(manifest, bench)

    assert manifest_path(bench).name == "benchmark_v3.manifest.json"
    loaded = read_manifest(bench)
    assert loaded is not None
    assert loaded.schema_fingerprint == "032028c3e280"
    assert loaded.corpus["publications_with_uzh_authors"] == 53545
    assert loaded.index["embedding_model"] == "BAAI/bge-m3"


def test_read_manifest_absent_is_not_an_error(tmp_path):
    """Every file generated before manifests existed has none; that is a fact."""
    from keyword_extractor.benchmark import read_manifest

    assert read_manifest(tmp_path / "no_such.jsonl") is None


def test_gold_supervisor_authority_type_defaults_for_older_files(tmp_path):
    """A pre-fix file must still load, and its silence is the staleness signal."""
    from keyword_extractor.benchmark import read_jsonl

    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"query": "q", "keywords": ["k"], "source_type": "publication", '
        '"source_ids": ["id"], "gold_supervisors": [{"name": "Doe, Jane", '
        '"cris_uuid": "uuid-doe"}], "granularity": "supervisor", '
        '"query_style": "llm"}\n',
        encoding="utf-8",
    )
    rows = read_jsonl(path)
    assert rows[0].gold_supervisors[0].authority_type is None
