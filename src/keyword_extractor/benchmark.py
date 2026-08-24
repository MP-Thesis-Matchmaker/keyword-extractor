"""Benchmark row model and JSONL persistence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class GoldSupervisor(BaseModel):
    """Expected match for a query. The CRIS UUID is the stable identity for
    audit; retrieval itself only returns names, so evaluation matches on the
    casefolded name."""

    name: str
    cris_uuid: str | None = None


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
    query_style: Literal["template", "llm"]
    metadata: dict[str, Any] = {}


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
