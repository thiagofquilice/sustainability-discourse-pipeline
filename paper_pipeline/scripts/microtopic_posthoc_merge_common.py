#!/usr/bin/env python3
"""Shared helpers for post-hoc microtopic merge review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.utils import get_column_letter

from cross_source_microtopic_common import DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT
from micro_topic_evolution_common import OUTPUT_ROOT as EVOLUTION_OUTPUT_ROOT


WORKFLOW_ROOT = Path("paper_pipeline")
RAW_MICRO_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_unsupervised"
RAW_MICRO_ROOT_MULTIASPECT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_unsupervised_multiaspect"
MERGE_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_posthoc_merge_review"
MERGE_FIRST_REVIEW_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_merge_first_review"
MERGE_FIRST_GROUP_REVIEW_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_merge_first_group_review"
MERGE_FIRST_REVIEW_MULTIASPECT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_merge_first_review_multiaspect"
MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT = (
    WORKFLOW_ROOT / "outputs" / "microtopic_merge_first_group_review_multiaspect"
)
MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_merge_first_hierarchical_review"
PAIR_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_cross_source_pairs_threshold_065"
DISCOURSE_OUTPUT_ROOT = DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT
MERGED_OUTPUT_ROOT = MERGE_OUTPUT_ROOT / "merged_outputs"
MERGED_MICRO_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_merged"
MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED = WORKFLOW_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
MERGE_FIRST_EVOLUTION_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_evolution_merge_first"
MERGE_FIRST_PAIR_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_cross_source_pairs_threshold_065_merge_first"
MERGE_FIRST_DISCOURSE_OUTPUT_ROOT = (
    WORKFLOW_ROOT / "outputs" / "microtopic_same_issue_discourse_function_top2_threshold055_with_academic_media_merge_first"
)
CORPORATE_FOCUS_PAIR_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "microtopic_cross_source_pairs_corporate_focus"
CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "corporate_focus_review"
CORPORATE_FOCUS_STAGE12_INPUT_ROOT = WORKFLOW_ROOT / "outputs" / "corporate_focus_stage12_input"
CORPORATE_FOCUS_AGGREGATED_TEMPORAL_OUTPUT_ROOT = (
    WORKFLOW_ROOT / "outputs" / "corporate_focus_temporal_aggregated_no_mnn"
)

SELECTED_TOPICS_PATH = EVOLUTION_OUTPUT_ROOT / "selected_micro_topics.csv"
YEAR_SUMMARIES_PATH = EVOLUTION_OUTPUT_ROOT / "year_summaries" / "micro_topic_year_summaries.csv"
YEAR_EVIDENCE_PATH = EVOLUTION_OUTPUT_ROOT / "micro_topic_year_evidence.csv"
STAGE2_NARRATIVES_PATH = EVOLUTION_OUTPUT_ROOT / "evolution_summaries" / "micro_topic_evolution_narratives.csv"
REPAIR_PENDING_PATH = (
    EVOLUTION_OUTPUT_ROOT / "evolution_summaries" / "repair_artifacts" / "stage2_pending_incomplete_rows.csv"
)
ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def strip_excel_illegal_chars(value: Any) -> Any:
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS_RE.sub("", value)
    return value


def sanitize_frame_for_excel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(strip_excel_illegal_chars)
    return cleaned


def parse_json_list(value: Any) -> list[Any]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def parse_json_dict(value: Any) -> dict[str, Any]:
    text = normalize_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def stringify_list(values: list[Any], sep: str = "; ") -> str:
    cleaned = [normalize_text(item) for item in values]
    cleaned = [item for item in cleaned if item]
    return sep.join(cleaned)


def load_stage2_rows(include_pending: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    ready = safe_read_csv(STAGE2_NARRATIVES_PATH)
    if not ready.empty:
        ready["is_narrative_row"] = True
        ready["is_narrative_ready"] = True
        frames.append(ready)
    if include_pending and REPAIR_PENDING_PATH.exists():
        pending = safe_read_csv(REPAIR_PENDING_PATH)
        if not pending.empty:
            pending["is_narrative_row"] = True
            pending["is_narrative_ready"] = False
            frames.append(pending)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["subgroup"] = combined["subgroup"].astype(str)
    combined["micro_topic_id"] = pd.to_numeric(combined["micro_topic_id"], errors="coerce")
    combined = combined.dropna(subset=["micro_topic_id"]).copy()
    combined["micro_topic_id"] = combined["micro_topic_id"].astype(int)
    combined = combined.sort_values(
        ["subgroup", "micro_topic_id", "is_narrative_ready"], ascending=[True, True, False]
    )
    combined = combined.drop_duplicates(["subgroup", "micro_topic_id"], keep="first").reset_index(drop=True)
    return combined


def workbook_autofit(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame, freeze_cell: str = "A2") -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = freeze_cell
    worksheet.auto_filter.ref = worksheet.dimensions
    for idx, column in enumerate(frame.columns, start=1):
        series = frame[column].astype(str).fillna("")
        max_len = max([len(str(column))] + [len(val) for val in series.head(200).tolist()]) if not frame.empty else len(str(column))
        worksheet.column_dimensions[get_column_letter(idx)].width = min(max(max_len + 2, 12), 60)
