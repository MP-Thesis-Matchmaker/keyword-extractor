"""JSONL round-trip and schema stability."""

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


def test_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "bench.jsonl"
    write_jsonl([_row()], path)
    assert path.exists()
