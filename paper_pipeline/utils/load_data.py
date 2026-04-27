"""Helpers for locating and streaming the raw datasets discovered in stage 0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class SourceFile:
    source: str
    path: Path


def locate_input_files(project_root: Path) -> dict[str, list[SourceFile]]:
    search_roots = [project_root, project_root / "data" / "raw"]
    guardian: Path | None = None
    papers: Path | None = None
    tenk_files: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in search_root.iterdir():
            if not candidate.is_file():
                continue
            lower_name = candidate.name.lower()
            if lower_name == "guardian_dataset.jsonl":
                guardian = candidate
            elif lower_name == "papers_dataset.jsonl":
                papers = candidate
            elif lower_name.startswith("10k") and candidate.suffix.lower() == ".jsonl":
                tenk_files.append(candidate)

    if guardian is None or papers is None or not tenk_files:
        raise FileNotFoundError(
            "Could not locate all required inputs. Expected guardian_dataset.jsonl, "
            "papers_dataset.jsonl and one or more 10K*.jsonl files."
        )

    return {
        "media": [SourceFile(source="media", path=guardian)],
        "academic": [SourceFile(source="academic", path=papers)],
        "corporate": [SourceFile(source="corporate", path=path) for path in sorted(tenk_files)],
    }


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield line_number, json.loads(stripped)
