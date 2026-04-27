#!/usr/bin/env python3
"""Shared helpers for the micro BERTopic temporal reading workflow."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger("micro_topic_evolution")
WORKFLOW_ROOT = Path("paper_pipeline")
MICRO_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_unsupervised"
OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "bertopic_micro_evolution"
SOURCE_ORDER = ["academic", "media", "corporate"]
OLLAMA_MODEL = "gemma4:e4b"
STAGE2_PHASE_FIELDS = [
    ("phase_1_years", "phase_1_summary"),
    ("phase_2_years", "phase_2_summary"),
    ("phase_3_years", "phase_3_summary"),
    ("phase_4_years", "phase_4_summary"),
]
STAGE2_CANONICAL_FIELDS = [
    "subgroup",
    "source",
    "assigned_label",
    "micro_topic_id",
    "topic_name_original",
    "topic_label_refined",
    "overall_summary",
    "phase_1_years",
    "phase_1_summary",
    "phase_2_years",
    "phase_2_summary",
    "phase_3_years",
    "phase_3_summary",
    "phase_4_years",
    "phase_4_summary",
    "evolution_pattern",
    "evidence_note",
]
NULLISH_STAGE2_VALUES = {"", "none", "null", "nan", "n/a", "na"}


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_topic_terms(raw_value: Any) -> list[str]:
    if pd.isna(raw_value):
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


def normalize_stage2_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple, set)):
        text = " ".join(str(value).split()).strip()
        return "" if text.lower() in NULLISH_STAGE2_VALUES else text
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = " ".join(str(value).split()).strip()
    if text.lower() in NULLISH_STAGE2_VALUES:
        return ""
    return text


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = normalize_stage2_text(value)
        if text:
            return text
    return ""


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _list_to_text(value: Any) -> str:
    if isinstance(value, list):
        cleaned = [normalize_stage2_text(item) for item in value]
        cleaned = [item for item in cleaned if item]
        return "; ".join(cleaned)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            key_text = normalize_stage2_text(key)
            item_text = normalize_stage2_text(item)
            if key_text and item_text:
                parts.append(f"{key_text}: {item_text}")
            elif item_text:
                parts.append(item_text)
            elif key_text:
                parts.append(key_text)
        return "; ".join(parts)
    return normalize_stage2_text(value)


def _clean_topic_name_as_label(value: Any) -> str:
    text = normalize_stage2_text(value)
    if not text:
        return ""
    if "_" in text and text.split("_", 1)[0].lstrip("-").isdigit():
        text = text.split("_", 1)[1]
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def _format_phase_summary(title: str, summary: str, keywords: Any = None) -> str:
    pieces: list[str] = []
    title = normalize_stage2_text(title)
    summary = normalize_stage2_text(summary)
    keyword_text = _list_to_text(keywords)
    if title:
        pieces.append(title)
    if summary:
        pieces.append(summary)
    if keyword_text:
        pieces.append(f"Keywords: {keyword_text}")
    return ": ".join(pieces[:2]) + (f" | {pieces[2]}" if len(pieces) > 2 else "") if pieces else ""


def _phase_years_from_item(item: dict[str, Any], default_year_span: str = "") -> str:
    years = _first_nonempty(
        item.get("years"),
        item.get("Years"),
        item.get("timeframe"),
        item.get("period"),
        item.get("Period"),
        item.get("Time_Period"),
        item.get("range"),
        item.get("date"),
        item.get("year_range"),
        item.get("year_group"),
    )
    if years:
        return years
    earliest = _safe_int(item.get("earliest_year"))
    latest = _safe_int(item.get("latest_year"))
    if earliest is not None and latest is not None:
        return str(earliest) if earliest == latest else f"{earliest}-{latest}"
    if earliest is not None:
        return str(earliest)
    year = _safe_int(item.get("year"))
    if year is None:
        year = _safe_int(item.get("Year"))
    if year is not None:
        return str(year)
    timeline = item.get("timeline")
    if isinstance(timeline, dict):
        earliest = _safe_int(timeline.get("earliest_year"))
        latest = _safe_int(timeline.get("latest_year"))
        if earliest is not None and latest is not None:
            return str(earliest) if earliest == latest else f"{earliest}-{latest}"
        if earliest is not None:
            return str(earliest)
    return default_year_span


def _phase_summary_from_item(item: dict[str, Any]) -> str:
    return _format_phase_summary(
        _first_nonempty(
            item.get("step_name"),
            item.get("phase"),
            item.get("stage_name"),
            item.get("title"),
            item.get("Topic"),
            item.get("Phase_Name"),
            item.get("name"),
            item.get("theme_name"),
            item.get("theme"),
        ),
        _first_nonempty(
            item.get("description"),
            item.get("characteristics"),
            item.get("summary"),
            item.get("Summary"),
            item.get("details"),
            item.get("focus"),
            item.get("Primary_Concerns"),
            item.get("concepts"),
            item.get("narrative"),
        ),
        item.get("keywords") or item.get("Keywords") or item.get("key_concepts") or item.get("themes"),
    )


def _extract_alt_phase_items(payload: dict[str, Any], default_year_span: str = "") -> list[tuple[str, str]]:
    phase_items: list[tuple[str, str]] = []
    for alt_key in [
        "steps",
        "phase_breakdown",
        "phases",
        "stages",
        "subgroup_history",
        "history",
        "evidence",
        "themes",
        "Key_Phases",
        "key_milestone_years",
        "chronological_evolution",
        "timeline_summary",
    ]:
        value = payload.get(alt_key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                years = _phase_years_from_item(item, default_year_span=default_year_span)
                summary = _phase_summary_from_item(item)
                if not years and summary:
                    years = default_year_span
                if years or summary:
                    phase_items.append((years, summary))
            if phase_items:
                return phase_items

    timeline = payload.get("timeline")
    if isinstance(timeline, dict):
        for key, value in timeline.items():
            years = normalize_stage2_text(key) or default_year_span
            if isinstance(value, dict):
                summary = _phase_summary_from_item(value)
                if not summary:
                    summary = _format_phase_summary("", _list_to_text(value))
            else:
                summary = _format_phase_summary("", _list_to_text(value))
            if years or summary:
                phase_items.append((years, summary))
        if phase_items:
            return phase_items
    if isinstance(timeline, list):
        for item in timeline:
            if isinstance(item, dict):
                years = _phase_years_from_item(item, default_year_span=default_year_span)
                summary = _phase_summary_from_item(item)
            else:
                years = default_year_span
                summary = _list_to_text(item)
            if years or summary:
                phase_items.append((years, summary))
        if phase_items:
            return phase_items
    nested_candidates = [
        payload.get("Timeline"),
        payload.get("timeline_overview"),
        payload.get("key_themes_timeline"),
        payload.get("evolution_of_focus"),
        payload.get("analysis", {}).get("key_policy_shifts") if isinstance(payload.get("analysis"), dict) else None,
        payload.get("thematic_evolution", {}).get("key_milestone_years") if isinstance(payload.get("thematic_evolution"), dict) else None,
        payload.get("key_milestone_years"),
        payload.get("chronological_evolution"),
    ]
    for nested in nested_candidates:
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                years = _phase_years_from_item(item, default_year_span=default_year_span)
                summary = _phase_summary_from_item(item)
                if years or summary:
                    phase_items.append((years, summary))
            if phase_items:
                return phase_items
        if isinstance(nested, dict):
            for key, value in nested.items():
                years = normalize_stage2_text(key) or default_year_span
                if isinstance(value, dict):
                    summary = _phase_summary_from_item(value)
                    if not summary:
                        summary = _format_phase_summary("", _list_to_text(value))
                else:
                    summary = _format_phase_summary("", _list_to_text(value))
                if years or summary:
                    phase_items.append((years, summary))
            if phase_items:
                return phase_items
    year_like_items: list[tuple[str, str]] = []
    for key, value in payload.items():
        years = normalize_stage2_text(key)
        if not re.fullmatch(r"\d{4}(?:\s*[-–]\s*\d{4})?", years):
            continue
        if isinstance(value, dict):
            summary = _phase_summary_from_item(value)
            if not summary:
                summary = _format_phase_summary("", _list_to_text(value))
        else:
            summary = _format_phase_summary("", _list_to_text(value))
        if years or summary:
            year_like_items.append((years, summary))
    if year_like_items:
        return year_like_items
    return phase_items


def canonicalize_stage2_payload(payload: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    canonical: dict[str, Any] = {
        "subgroup": defaults.get("subgroup", ""),
        "source": defaults.get("source", ""),
        "assigned_label": defaults.get("assigned_label", ""),
        "micro_topic_id": _safe_int(payload.get("micro_topic_id"), _safe_int(defaults.get("micro_topic_id"), None)),
        "topic_name_original": _first_nonempty(payload.get("topic_name_original"), defaults.get("topic_name_original")),
        "topic_label_refined": _first_nonempty(
            payload.get("topic_label_refined"),
            payload.get("topic_name_raw"),
            payload.get("title"),
            payload.get("topic"),
            payload.get("Topic"),
            payload.get("subgroup"),
            payload.get("subgroup_name"),
            payload.get("theme_name"),
            defaults.get("topic_label_refined"),
        ),
        "overall_summary": _first_nonempty(
            payload.get("overall_summary"),
            payload.get("summary"),
            payload.get("period_overview"),
            payload.get("historical_evolution"),
            payload.get("historical_summary"),
            payload.get("historical_trend"),
            payload.get("narrative_summary"),
            payload.get("subgroup_description"),
            payload.get("subgroup_details"),
            payload.get("subgroup_description"),
            payload.get("description"),
            payload.get("trend_summary"),
            payload.get("context"),
            payload.get("evolutionary_description"),
            payload.get("analysis_summary"),
            payload.get("Trend_Description"),
            payload.get("Summary"),
            payload.get("scope"),
            payload.get("dominant_topic"),
            payload.get("overall_theme"),
            payload.get("timeline_summary", {}).get("period_overview") if isinstance(payload.get("timeline_summary"), dict) else "",
            payload.get("synthesis", {}).get("summary") if isinstance(payload.get("synthesis"), dict) else "",
            payload.get("conclusion"),
            defaults.get("overall_summary"),
        ),
        "evolution_pattern": _first_nonempty(
            payload.get("evolution_pattern"),
            payload.get("timeline_description"),
            payload.get("synthesis", {}).get("trend_analysis") if isinstance(payload.get("synthesis"), dict) else "",
            payload.get("concept_evolution", {}).get("process_description") if isinstance(payload.get("concept_evolution"), dict) else "",
            defaults.get("evolution_pattern"),
        ),
        "evidence_note": _first_nonempty(
            payload.get("evidence_note"),
            _list_to_text(payload.get("key_themes")),
            _list_to_text(payload.get("key_trends")),
            _list_to_text(payload.get("key_drivers")),
            _list_to_text(payload.get("challenges")),
            _list_to_text(payload.get("impact_areas")),
            _list_to_text(payload.get("predictive_focus_areas")),
            _list_to_text(payload.get("focus_areas")),
            _list_to_text(payload.get("major_themes")),
            _list_to_text(payload.get("Major Trends")),
            _list_to_text(payload.get("themes")),
            defaults.get("evidence_note"),
        ),
    }

    for years_key, summary_key in STAGE2_PHASE_FIELDS:
        canonical[years_key] = normalize_stage2_text(payload.get(years_key))
        canonical[summary_key] = normalize_stage2_text(payload.get(summary_key))

    default_year_span = normalize_stage2_text(defaults.get("default_year_span"))
    phase_items = _extract_alt_phase_items(payload, default_year_span=default_year_span)
    phase_items_from_scalar = []
    if normalize_stage2_text(payload.get("phase_4")):
        phase_items_from_scalar.append((default_year_span, normalize_stage2_text(payload.get("phase_4"))))

    alt_phase_items = phase_items or phase_items_from_scalar
    if alt_phase_items:
        for idx, (years_key, summary_key) in enumerate(STAGE2_PHASE_FIELDS):
            if idx >= len(alt_phase_items):
                break
            if not canonical[years_key]:
                canonical[years_key] = normalize_stage2_text(alt_phase_items[idx][0])
            if not canonical[summary_key]:
                canonical[summary_key] = normalize_stage2_text(alt_phase_items[idx][1])

    if not canonical["overall_summary"]:
        fallback_summary = _first_nonempty(payload.get("summary"), payload.get("period_overview"), payload.get("conclusion"))
        if fallback_summary:
            canonical["overall_summary"] = fallback_summary
    if not canonical["overall_summary"]:
        theme_text = _first_nonempty(
            _list_to_text(payload.get("focus_areas")),
            _list_to_text(payload.get("key_trends")),
            _list_to_text(payload.get("themes")),
            _list_to_text(payload.get("Keywords")),
            _list_to_text(payload.get("keywords")),
        )
        if theme_text:
            canonical["overall_summary"] = f"Temporal synthesis centers on: {theme_text}."
    if not canonical["topic_label_refined"]:
        canonical["topic_label_refined"] = _clean_topic_name_as_label(
            _first_nonempty(canonical.get("topic_name_original"), defaults.get("topic_name_original"))
        )
    if not canonical["overall_summary"]:
        phase_summaries = [
            normalize_stage2_text(canonical.get(summary_key))
            for _, summary_key in STAGE2_PHASE_FIELDS
            if normalize_stage2_text(canonical.get(summary_key))
        ]
        if phase_summaries:
            canonical["overall_summary"] = "Temporal synthesis based on available phase evidence: " + " ".join(phase_summaries[:2])
    if stage2_complete_phase_count(canonical) == 0 and canonical["overall_summary"]:
        canonical["phase_1_years"] = canonical["phase_1_years"] or default_year_span or "full period"
        canonical["phase_1_summary"] = canonical["phase_1_summary"] or canonical["overall_summary"]

    if not canonical["evolution_pattern"]:
        titles = []
        for _, summary_key in STAGE2_PHASE_FIELDS:
            summary = normalize_stage2_text(canonical.get(summary_key))
            if summary:
                titles.append(summary.split(":", 1)[0][:80])
        if len(titles) >= 2:
            canonical["evolution_pattern"] = " -> ".join(titles[:4])
        elif len(titles) == 1:
            canonical["evolution_pattern"] = "Single phase observation"

    if canonical["micro_topic_id"] is None:
        canonical["micro_topic_id"] = _safe_int(defaults.get("micro_topic_id"), 0)

    for field in STAGE2_CANONICAL_FIELDS:
        if field == "micro_topic_id":
            continue
        canonical[field] = normalize_stage2_text(canonical.get(field))
    return canonical


def stage2_complete_phase_count(record: dict[str, Any] | pd.Series) -> int:
    return sum(
        1
        for years_key, summary_key in STAGE2_PHASE_FIELDS
        if normalize_stage2_text(record.get(years_key)) and normalize_stage2_text(record.get(summary_key))
    )


def stage2_missing_core_fields(record: dict[str, Any] | pd.Series) -> list[str]:
    missing: list[str] = []
    if not normalize_stage2_text(record.get("topic_label_refined")):
        missing.append("topic_label_refined")
    if not normalize_stage2_text(record.get("overall_summary")):
        missing.append("overall_summary")
    if stage2_complete_phase_count(record) == 0:
        missing.append("complete_phase")
    return missing


def is_stage2_narrative_ready(record: dict[str, Any] | pd.Series) -> bool:
    return len(stage2_missing_core_fields(record)) == 0


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = strip_ansi(raw_text)
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in model response.")
    depth = 0
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw_text[start : idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    sanitized = candidate.replace("\r", " ").replace("\n", " ").replace("\t", " ")
                    return json.loads(sanitized)
    raise ValueError("No complete JSON object found in model response.")


def call_ollama(model: str, prompt: str, timeout_seconds: int = 240) -> str:
    result = subprocess.run(
        ["ollama", "run", model, "--hidethinking", "--nowordwrap", "--think=false", prompt],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ollama exited with code {result.returncode}")
    return result.stdout.strip()


def resume_completed_keys(path: Path, key_columns: list[str]) -> set[tuple[Any, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_json(path, lines=True)
    if frame.empty:
        return set()
    return {tuple(row[column] for column in key_columns) for _, row in frame[key_columns].iterrows()}
