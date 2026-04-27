#!/usr/bin/env python3
"""Repair Stage 2 evolution narratives via raw-response salvage plus targeted rerun."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from micro_topic_evolution_common import (
    OLLAMA_MODEL,
    OUTPUT_ROOT,
    STAGE2_CANONICAL_FIELDS,
    canonicalize_stage2_payload,
    configure_logging,
    extract_json_object,
    is_stage2_narrative_ready,
    normalize_stage2_text,
    stage2_complete_phase_count,
    stage2_missing_core_fields,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", type=Path, default=OUTPUT_ROOT / "selected_micro_topics.csv")
    parser.add_argument("--annual-summaries", type=Path, default=OUTPUT_ROOT / "year_summaries" / "micro_topic_year_summaries.csv")
    parser.add_argument("--year-evidence", type=Path, default=OUTPUT_ROOT / "micro_topic_year_evidence.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "evolution_summaries")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--refresh-sample", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_year_span(annual: pd.DataFrame) -> str:
    if annual.empty:
        return ""
    years = pd.to_numeric(annual["year"], errors="coerce").dropna().astype(int).tolist()
    if not years:
        return ""
    year_min = min(years)
    year_max = max(years)
    return str(year_min) if year_min == year_max else f"{year_min}-{year_max}"


def record_score(record: dict[str, Any]) -> tuple[int, int, int, int]:
    present_count = sum(1 for field in STAGE2_CANONICAL_FIELDS if field != "micro_topic_id" and normalize_stage2_text(record.get(field)))
    return (
        int(is_stage2_narrative_ready(record)),
        stage2_complete_phase_count(record),
        present_count,
        len(normalize_stage2_text(record.get("overall_summary"))),
    )


def merge_records(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for field in STAGE2_CANONICAL_FIELDS:
        if field == "micro_topic_id":
            merged[field] = incoming.get(field, merged.get(field))
            continue
        if not normalize_stage2_text(merged.get(field)) and normalize_stage2_text(incoming.get(field)):
            merged[field] = incoming.get(field)
    return merged


def write_canonical_rows(path_csv: Path, path_jsonl: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=STAGE2_CANONICAL_FIELDS)
    frame.to_csv(path_csv, index=False)
    with path_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    narratives_csv = args.output_dir / "micro_topic_evolution_narratives.csv"
    narratives_jsonl = args.output_dir / "micro_topic_evolution_narratives.jsonl"
    raw_jsonl = args.output_dir / "raw_responses.jsonl"
    repair_root = args.output_dir / "repair_artifacts"
    repair_root.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_topics)
    annual = pd.read_csv(args.annual_summaries)

    defaults_map: dict[tuple[str, int], dict[str, Any]] = {}
    annual_grouped = annual.groupby(["subgroup", "micro_topic_id"], dropna=False)
    for row in selected.itertuples(index=False):
        key = (str(row.subgroup), int(row.micro_topic_id))
        annual_rows = annual_grouped.get_group(key) if key in annual_grouped.groups else pd.DataFrame()
        defaults_map[key] = {
            "subgroup": str(row.subgroup),
            "source": str(row.source),
            "assigned_label": str(row.assigned_label),
            "micro_topic_id": int(row.micro_topic_id),
            "topic_name_original": str(row.topic_name_original),
            "default_year_span": build_year_span(annual_rows),
        }

    current_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if narratives_csv.exists() and narratives_csv.stat().st_size > 0:
        current = pd.read_csv(narratives_csv)
        for _, row in current.iterrows():
            key = (str(row["subgroup"]), int(row["micro_topic_id"]))
            defaults = defaults_map.get(key, {"micro_topic_id": int(row["micro_topic_id"])})
            current_rows[key] = canonicalize_stage2_payload(dict(row), defaults=defaults)

    if narratives_csv.exists():
        shutil.copy2(narratives_csv, repair_root / "micro_topic_evolution_narratives_before_repair.csv")
    if narratives_jsonl.exists():
        shutil.copy2(narratives_jsonl, repair_root / "micro_topic_evolution_narratives_before_repair.jsonl")

    salvaged_rows: dict[tuple[str, int], dict[str, Any]] = {}
    salvage_meta: dict[tuple[str, int], dict[str, Any]] = {}
    if raw_jsonl.exists() and raw_jsonl.stat().st_size > 0:
        with raw_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                subgroup = payload.get("subgroup")
                micro_topic_id = payload.get("micro_topic_id")
                if subgroup is None or micro_topic_id is None or "raw_response" not in payload:
                    continue
                key = (str(subgroup), int(micro_topic_id))
                defaults = defaults_map.get(key, {"subgroup": subgroup, "micro_topic_id": int(micro_topic_id)})
                try:
                    parsed = extract_json_object(payload["raw_response"])
                except Exception:
                    continue
                canonical = canonicalize_stage2_payload(parsed, defaults=defaults)
                if key not in salvaged_rows or record_score(canonical) > record_score(salvaged_rows[key]):
                    salvaged_rows[key] = canonical
                    salvage_meta[key] = {
                        "parsed_keys": "|".join(sorted(parsed.keys())),
                        "ready_after_salvage": bool(is_stage2_narrative_ready(canonical)),
                    }

    ready_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for key, defaults in defaults_map.items():
        current_record = current_rows.get(key, canonicalize_stage2_payload({}, defaults=defaults))
        salvage_record = salvaged_rows.get(key)
        merged = merge_records(current_record, salvage_record) if salvage_record else current_record
        ready = is_stage2_narrative_ready(merged)
        missing_fields = stage2_missing_core_fields(merged)
        audit_row = {
            "subgroup": key[0],
            "micro_topic_id": key[1],
            "has_current_row": key in current_rows,
            "has_salvaged_row": key in salvaged_rows,
            "ready_after_merge": ready,
            "missing_core_fields": "|".join(missing_fields),
            "current_score": record_score(current_record),
            "merged_score": record_score(merged),
            "salvage_parsed_keys": salvage_meta.get(key, {}).get("parsed_keys", ""),
        }
        audit_rows.append(audit_row)
        if ready:
            ready_rows.append(merged)
        else:
            pending_row = {field: merged.get(field, "") for field in STAGE2_CANONICAL_FIELDS}
            pending_row["missing_core_fields"] = "|".join(missing_fields)
            pending_rows.append(pending_row)

    ready_rows = sorted(ready_rows, key=lambda row: (row["subgroup"], int(row["micro_topic_id"])))
    write_canonical_rows(narratives_csv, narratives_jsonl, ready_rows)

    audit_path = repair_root / "stage2_narrative_repair_audit.csv"
    pending_path = repair_root / "stage2_pending_incomplete_rows.csv"
    subset_selected_path = repair_root / "stage2_pending_selected_topics.csv"
    pd.DataFrame(audit_rows).sort_values(["subgroup", "micro_topic_id"]).to_csv(audit_path, index=False)
    pd.DataFrame(pending_rows).to_csv(pending_path, index=False)

    pending_keys = pd.DataFrame(
        [{"subgroup": row["subgroup"], "micro_topic_id": row["micro_topic_id"]} for row in pending_rows],
        columns=["subgroup", "micro_topic_id"],
    )
    if pending_keys.empty:
        subset_selected = selected.head(0).copy()
    else:
        subset_selected = selected.merge(
            pending_keys,
            on=["subgroup", "micro_topic_id"],
            how="inner",
        )
    subset_selected.to_csv(subset_selected_path, index=False)

    pre_manifest = {
        "selected_topics": str(args.selected_topics),
        "annual_summaries": str(args.annual_summaries),
        "raw_responses": str(raw_jsonl),
        "ready_rows_after_salvage": len(ready_rows),
        "pending_rows_after_salvage": len(pending_rows),
        "rerun_incomplete_requested": bool(args.rerun_incomplete),
        "outputs": {
            "repaired_narratives_csv": str(narratives_csv),
            "repaired_narratives_jsonl": str(narratives_jsonl),
            "repair_audit_csv": str(audit_path),
            "pending_incomplete_csv": str(pending_path),
            "pending_selected_topics_csv": str(subset_selected_path),
        },
    }
    write_json(repair_root / "stage2_repair_manifest_pre_rerun.json", pre_manifest)

    rerun_returncode = None
    if args.rerun_incomplete and not subset_selected.empty:
        script_path = Path(__file__).resolve().parent / "run_micro_topic_evolution_synthesis.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--selected-topics",
            str(subset_selected_path),
            "--annual-summaries",
            str(args.annual_summaries),
            "--year-evidence",
            str(args.year_evidence),
            "--output-dir",
            str(args.output_dir),
            "--model",
            args.model,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-attempts",
            str(args.max_attempts),
            "--log-level",
            args.log_level,
        ]
        rerun = subprocess.run(cmd, check=False)
        rerun_returncode = rerun.returncode

    if args.refresh_sample:
        sample_script = Path(__file__).resolve().parent / "build_stage2_evolution_manual_validation_sample.py"
        subprocess.run([sys.executable, str(sample_script)], check=False)

    final_ready_count = 0
    final_pending_count = 0
    if narratives_csv.exists() and narratives_csv.stat().st_size > 0:
        final_rows = pd.read_csv(narratives_csv)
        final_ready_count = int(final_rows.apply(is_stage2_narrative_ready, axis=1).sum())
    final_pending_count = int(len(selected) - final_ready_count)

    write_json(
        repair_root / "stage2_repair_manifest.json",
        {
            **pre_manifest,
            "rerun_returncode": rerun_returncode,
            "final_ready_rows": final_ready_count,
            "final_pending_rows": final_pending_count,
        },
    )

    print(args.output_dir)


if __name__ == "__main__":
    main()
