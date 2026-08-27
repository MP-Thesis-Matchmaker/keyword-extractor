"""Benchmark row model, JSONL persistence, and the run manifest.

A benchmark is only as valid as the corpus its gold labels were drawn from, and
that is not something a row can tell you by itself. The first published set was
generated while backend-core still admitted any DSpace authority into
`uzh_authors`; when that was narrowed to registered UZH researchers, 87 of its
105 gold supervisors stopped being supervisor-eligible at all -- unreachable by
construction, so every score computed from it was wrong. Nothing in the file
said so, because a row recorded the extractor (`keybert_model`, `seed`) and
nothing about the corpus.

`BenchmarkManifest` is the fix, written as a sidecar next to the JSONL. A JSONL
header line was the alternative and is worse: it breaks `read_jsonl` and every
other consumer for the sake of one record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class GoldSupervisor(BaseModel):
    """Expected match for a query. The CRIS UUID is the stable identity for
    audit; retrieval itself only returns names, so evaluation matches on the
    casefolded name."""

    name: str
    cris_uuid: str | None = None
    # Which kind of authority backed this label: "cris" for a registered UZH
    # researcher, "orcid" for an identity that resolves to no person row.
    # Defaulted so files written before the field existed still load -- and
    # their silence is itself the signal that they predate it.
    authority_type: str | None = None


class BenchmarkRow(BaseModel):
    """One synthesised benchmark query with its provenance and gold labels."""

    query: str
    keywords: list[str]
    source_type: Literal["publication", "posting"]
    source_ids: list[str] = Field(
        description="Publication ids or the posting id the keywords came from; "
        "kept so any suspicious gold label can be audited against the source rows."
    )
    gold_supervisors: list[GoldSupervisor]
    granularity: Literal["supervisor", "posting"]
    # What actually produced the text, not what the run asked for: an LLM
    # synthesiser that fell back to the template records "template".
    query_style: Literal["template", "llm"]
    # Defaulted so benchmark files written before complexity levels existed
    # still load.
    complexity: Literal["low", "medium", "high"] = "medium"
    metadata: dict[str, Any] = {}


class BenchmarkManifest(BaseModel):
    """What a benchmark file was generated from, so staleness is detectable.

    `schema_fingerprint` is the single field that would have caught the previous
    set going stale: it pins the migration state of the database the labels came
    from, and a diff of two manifests answers "did the ground truth move?" in one
    line.
    """

    generated_at: str
    schema_fingerprint: str | None = None
    # Corpus shape at generation time: publications, publications with a
    # non-empty uzh_authors, distinct uzh_author names, candidates at this run's
    # threshold, how many of those were CRIS-typed, postings.
    corpus: dict[str, int] = {}
    # GET /v1/index/status, when a matcher was reachable. Generation does not
    # need one; recording it makes a generate/evaluate pair traceable.
    index: dict[str, Any] | None = None
    run: dict[str, Any] = {}


def manifest_path(benchmark_path: Path) -> Path:
    """The sidecar path for a benchmark file: `x.jsonl` -> `x.manifest.json`."""
    return benchmark_path.with_suffix(".manifest.json")


def write_manifest(manifest: BenchmarkManifest, path: Path) -> None:
    """Write the manifest sidecar for the benchmark at `path`."""
    target = manifest_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> BenchmarkManifest | None:
    """The manifest beside the benchmark at `path`, or None if it has none.

    Absent for every file generated before manifests existed, which is a fact
    worth reporting rather than an error.
    """
    target = manifest_path(path)
    if not target.exists():
        return None
    return BenchmarkManifest.model_validate(json.loads(target.read_text(encoding="utf-8")))


def utc_now() -> str:
    """Current time as a UTC ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def write_jsonl(rows: Iterable[BenchmarkRow], path: Path) -> int:
    """Write rows as JSON Lines; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[BenchmarkRow]:
    """Read rows back; blank lines are skipped."""
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(BenchmarkRow.model_validate_json(line))
    return rows
