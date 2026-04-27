#!/usr/bin/env python3
"""Shared helpers for cross-source matched microtopic analysis."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from micro_topic_evolution_common import is_stage2_narrative_ready


LOGGER = logging.getLogger("cross_source_microtopics")
WORKFLOW_ROOT = Path("paper_pipeline")
MICRO_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_unsupervised"
OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_cross_source_pairs"
DEFAULT_THRESHOLD_065_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_cross_source_pairs_threshold_065"
DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT = (
    WORKFLOW_ROOT / "outputs" / "microtopic_same_issue_discourse_function_top2_threshold055"
)
DEFAULT_DISCOURSE_FUNCTION_WITH_ACADEMIC_MEDIA_OUTPUT_ROOT = (
    WORKFLOW_ROOT / "outputs" / "microtopic_same_issue_discourse_function_top2_threshold055_with_academic_media"
)
YEAR_SUMMARIES_PATH = (
    WORKFLOW_ROOT / "outputs" / "bertopic_micro_evolution" / "year_summaries" / "micro_topic_year_summaries.csv"
)
EVOLUTION_NARRATIVES_PATH = (
    WORKFLOW_ROOT
    / "outputs"
    / "bertopic_micro_evolution"
    / "evolution_summaries"
    / "micro_topic_evolution_narratives.csv"
)

SOURCE_ORDER = ["academic", "media", "corporate"]
MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
DYADS: list[tuple[str, str]] = [("media", "corporate"), ("academic", "corporate")]
DYAD_LABELS = {
    ("media", "corporate"): "media ↔ corporate",
    ("academic", "corporate"): "academic ↔ corporate",
}
DISCOURSE_DYADS: list[tuple[str, str]] = [("media", "corporate"), ("academic", "corporate"), ("academic", "media")]
DISCOURSE_DYAD_LABELS = {
    ("media", "corporate"): "media ↔ corporate",
    ("academic", "corporate"): "academic ↔ corporate",
    ("academic", "media"): "academic ↔ media",
}
SIMILARITY_THRESHOLDS = [0.65, 0.70, 0.75, 0.80]
TEMPORAL_RULES = {
    "lenient": {
        "min_active_years_per_side": 3,
        "min_overlap_years": 3,
        "min_docs_per_side": 5,
        "min_chunks_per_side": 10,
    },
    "balanced": {
        "min_active_years_per_side": 4,
        "min_overlap_years": 4,
        "min_docs_per_side": 10,
        "min_chunks_per_side": 20,
    },
    "strict": {
        "min_active_years_per_side": 5,
        "min_overlap_years": 5,
        "min_docs_per_side": 20,
        "min_chunks_per_side": 40,
    },
}


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=False)


def threshold_suffix(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def parse_topic_terms(raw_value: Any) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip().strip("'").strip('"') for part in text.split(",")]
    return [part for part in parts if part]


def clean_microtopic_label(raw_label: Any) -> str:
    if raw_label is None:
        return ""
    text = str(raw_label).strip()
    if not text:
        return ""
    if "_" in text and text.split("_", 1)[0].lstrip("-").isdigit():
        return text.split("_", 1)[1]
    return text


def truncate_text(text: Any, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def build_profile_text(
    microtopic_label: str,
    top_terms: list[str],
    representative_chunks: list[str],
) -> str:
    lines = [
        f"Microtopic label: {microtopic_label or 'Unknown'}",
        "Top terms: " + ("; ".join(top_terms) if top_terms else "None available"),
        "Representative chunks:",
    ]
    if representative_chunks:
        for idx, chunk in enumerate(representative_chunks, start=1):
            lines.append(f"[{idx}] {truncate_text(chunk, 480)}")
    else:
        lines.append("[1] No representative chunk available.")
    return "\n".join(lines)


def build_pair_id(
    dyad: str,
    macro_topic: str,
    source_a: str,
    microtopic_id_a: int,
    source_b: str,
    microtopic_id_b: int,
) -> str:
    return f"{dyad}__{macro_topic}__{source_a}_{microtopic_id_a}__{source_b}_{microtopic_id_b}"


def dyad_code(source_a: str, source_b: str) -> str:
    return f"{source_a}__{source_b}"


def ordered_frame(frame: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    ordered = frame.copy()
    if "source" in ordered.columns:
        ordered["source"] = pd.Categorical(ordered["source"], categories=SOURCE_ORDER, ordered=True)
    if "assigned_label" in ordered.columns:
        ordered["assigned_label"] = pd.Categorical(
            ordered["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
        )
    return ordered.sort_values(sort_cols).reset_index(drop=True)


def list_subgroup_dirs(micro_root: Path = MICRO_ROOT) -> list[Path]:
    return sorted(path for path in micro_root.iterdir() if path.is_dir() and path.name != "summary")


def subgroup_macro_topic_name(document_topics: pd.DataFrame, manifest: dict[str, Any]) -> str:
    if "topic_name" in manifest and manifest.get("topic_name"):
        return str(manifest["topic_name"])
    if "topic_name" in document_topics.columns and document_topics["topic_name"].notna().any():
        return str(document_topics["topic_name"].dropna().iloc[0])
    return ""


def load_stage1_enrichment(path: Path = YEAR_SUMMARIES_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"subgroup", "micro_topic_id", "year"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["micro_topic_id"] = pd.to_numeric(frame["micro_topic_id"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["micro_topic_id", "year"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["micro_topic_id"] = frame["micro_topic_id"].astype(int)
    frame["year"] = frame["year"].astype(int)
    grouped = (
        frame.sort_values(["subgroup", "micro_topic_id", "year"])
        .groupby(["subgroup", "micro_topic_id"], dropna=False)
        .agg(
            stage1_summary_year_count=("year", "nunique"),
            stage1_summary_year_min=("year", "min"),
            stage1_summary_year_max=("year", "max"),
            stage1_focus_primary_examples=(
                "year_focus_primary",
                lambda s: [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()][:3],
            ),
            stage1_frame_examples=(
                "year_frame_or_angle",
                lambda s: [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()][:3],
            ),
        )
        .reset_index()
    )
    grouped["stage1_summary_available"] = grouped["stage1_summary_year_count"] > 0
    grouped["stage1_focus_primary_examples_json"] = grouped["stage1_focus_primary_examples"].map(json_dumps)
    grouped["stage1_frame_examples_json"] = grouped["stage1_frame_examples"].map(json_dumps)
    return grouped.drop(columns=["stage1_focus_primary_examples", "stage1_frame_examples"])


def variance_flag(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    arr = np.asarray(values, dtype="float64")
    if np.allclose(arr, arr[0]):
        return True
    return float(np.nanstd(arr)) == 0.0


def load_stage2_narratives(path: Path = EVOLUTION_NARRATIVES_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or not {"subgroup", "micro_topic_id"}.issubset(frame.columns):
        return pd.DataFrame()
    frame["micro_topic_id"] = pd.to_numeric(frame["micro_topic_id"], errors="coerce")
    frame = frame.dropna(subset=["micro_topic_id"]).copy()
    frame["micro_topic_id"] = frame["micro_topic_id"].astype(int)
    frame["is_narrative_ready"] = frame.apply(is_stage2_narrative_ready, axis=1)
    return frame


def stage2_narrative_key_sets(path: Path = EVOLUTION_NARRATIVES_PATH) -> tuple[set[tuple[str, int]], set[tuple[str, int]], pd.DataFrame]:
    frame = load_stage2_narratives(path)
    row_keys: set[tuple[str, int]] = set()
    if not frame.empty:
        row_keys = {(str(row.subgroup), int(row.micro_topic_id)) for row in frame.itertuples(index=False)}
    ready_keys = {
        (str(row.subgroup), int(row.micro_topic_id))
        for row in frame.loc[frame["is_narrative_ready"]].itertuples(index=False)
    } if not frame.empty else set()
    pending_path = path.parent / "repair_artifacts" / "stage2_pending_incomplete_rows.csv"
    if pending_path.exists() and pending_path.stat().st_size > 0:
        try:
            pending = pd.read_csv(pending_path)
        except pd.errors.EmptyDataError:
            pending = pd.DataFrame()
        if {"subgroup", "micro_topic_id"}.issubset(pending.columns):
            pending["micro_topic_id"] = pd.to_numeric(pending["micro_topic_id"], errors="coerce")
            pending = pending.dropna(subset=["micro_topic_id"]).copy()
            pending["micro_topic_id"] = pending["micro_topic_id"].astype(int)
            row_keys.update((str(row.subgroup), int(row.micro_topic_id)) for row in pending.itertuples(index=False))
    return row_keys, ready_keys, frame
