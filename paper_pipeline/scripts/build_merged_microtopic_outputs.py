#!/usr/bin/env python3
"""Build derived outputs for manually merged post-hoc microtopic groups."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cross_source_microtopic_common import configure_logging
from micro_topic_evolution_common import STAGE2_CANONICAL_FIELDS
from microtopic_posthoc_merge_common import (
    MERGE_OUTPUT_ROOT,
    MERGED_OUTPUT_ROOT,
    SELECTED_TOPICS_PATH,
    STAGE2_NARRATIVES_PATH,
    YEAR_EVIDENCE_PATH,
    YEAR_SUMMARIES_PATH,
    ensure_directory,
    json_dumps,
    load_stage2_rows,
    normalize_text,
    parse_json_list,
    sanitize_frame_for_excel,
    safe_read_csv,
    workbook_autofit,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGED_OUTPUT_ROOT)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=MERGE_OUTPUT_ROOT / "microtopic_to_merged_group.csv",
    )
    parser.add_argument("--selected-topics", type=Path, default=SELECTED_TOPICS_PATH)
    parser.add_argument(
        "--profiles-csv",
        type=Path,
        default=Path("paper_pipeline/outputs/microtopic_cross_source_pairs_threshold_065/microtopic_profiles.csv"),
    )
    parser.add_argument("--year-summaries-csv", type=Path, default=YEAR_SUMMARIES_PATH)
    parser.add_argument("--year-evidence-csv", type=Path, default=YEAR_EVIDENCE_PATH)
    parser.add_argument("--stage2-csv", type=Path, default=STAGE2_NARRATIVES_PATH)
    parser.add_argument("--run-stage2", action="store_true")
    parser.add_argument("--stage2-output-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def stable_unique(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def parse_chunk_records(value: Any) -> list[dict[str, Any]]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    records: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            records.append(item)
    return records


def build_group_metadata(
    merged_members: pd.DataFrame,
    stage2_rows: pd.DataFrame,
) -> pd.DataFrame:
    metadata_rows: list[dict[str, Any]] = []
    stage2_map = {
        (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
        for row in stage2_rows.itertuples(index=False)
    }
    for group_key, group in merged_members.groupby(
        ["subgroup", "source", "assigned_label", "macro_topic_name", "final_merge_group_id", "merged_micro_topic_id"],
        dropna=False,
    ):
        subgroup, source, assigned_label, macro_topic_name, final_merge_group_id, merged_micro_topic_id = group_key
        member_ids = sorted(group["micro_topic_id"].astype(int).tolist())
        years_union = sorted(
            {
                int(year)
                for value in group["years_present"].tolist()
                for year in parse_json_list(value)
                if str(year).strip()
            }
        )
        top_terms_union = stable_unique(
            [str(term) for value in group["top_terms"].tolist() for term in parse_json_list(value)],
            limit=15,
        )
        representative_chunks = stable_unique(
            [str(chunk) for value in group["representative_chunks"].tolist() for chunk in parse_json_list(value)],
            limit=5,
        )
        member_refined_labels = stable_unique(
            [
                normalize_text(stage2_map.get((str(subgroup), int(micro_topic_id)), {}).get("topic_label_refined"))
                for micro_topic_id in member_ids
            ]
        )
        member_overall_summaries = stable_unique(
            [
                normalize_text(stage2_map.get((str(subgroup), int(micro_topic_id)), {}).get("overall_summary"))
                for micro_topic_id in member_ids
            ],
            limit=4,
        )
        narrative_ready_count = int(
            sum(
                bool(stage2_map.get((str(subgroup), int(micro_topic_id)), {}).get("is_narrative_ready", False))
                for micro_topic_id in member_ids
            )
        )
        merged_topic_name_original = (
            str(group["topic_name_original"].iloc[0])
            if len(member_ids) == 1
            else f"merged_{final_merge_group_id}"
        )
        heterogeneity_note = (
            "Single original microtopic retained."
            if len(member_ids) == 1
            else (
                "Multiple original microtopics grouped with partially distinct refined labels."
                if len(member_refined_labels) > 1
                else "Multiple original microtopics grouped under a consistent refined label family."
            )
        )
        metadata_rows.append(
            {
                "subgroup": subgroup,
                "source": source,
                "assigned_label": assigned_label,
                "macro_topic_name": macro_topic_name,
                "final_merge_group_id": final_merge_group_id,
                "merged_micro_topic_id": int(merged_micro_topic_id),
                "merged_topic_name_original": merged_topic_name_original,
                "group_size": int(group["group_size"].iloc[0]),
                "is_singleton_group": bool(group["is_singleton_group"].iloc[0]),
                "member_micro_topic_ids_json": json_dumps(member_ids),
                "member_topic_names_json": json_dumps(group["topic_name_original"].astype(str).tolist()),
                "member_topic_label_refined_json": json_dumps(member_refined_labels),
                "member_overall_summaries_json": json_dumps(member_overall_summaries),
                "topic_size": int(group["topic_size"].sum()),
                "share_of_non_outlier": float(group["share_of_non_outlier"].sum()),
                "document_count": int(group["document_count"].sum()),
                "chunk_count": int(group["chunk_count"].sum()),
                "active_year_count": int(len(years_union)),
                "active_year_min": int(min(years_union)) if years_union else None,
                "active_year_max": int(max(years_union)) if years_union else None,
                "years_present_json": json_dumps(years_union),
                "overall_keywords": json_dumps(top_terms_union),
                "representative_chunks_json": json_dumps(representative_chunks),
                "narrative_ready_member_count": narrative_ready_count,
                "heterogeneity_note": heterogeneity_note,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    if metadata.empty:
        return metadata
    metadata = metadata.sort_values(["subgroup", "merged_micro_topic_id"]).reset_index(drop=True)
    metadata["cumulative_share"] = (
        metadata.sort_values(["subgroup", "topic_size"], ascending=[True, False])
        .groupby("subgroup", dropna=False)["share_of_non_outlier"]
        .cumsum()
        .sort_index()
    )
    return metadata


def build_merged_stage1_annual_summaries(
    mapping: pd.DataFrame,
    metadata: pd.DataFrame,
    year_summaries: pd.DataFrame,
) -> pd.DataFrame:
    annual = year_summaries.merge(
        mapping[
            ["subgroup", "micro_topic_id", "final_merge_group_id", "merged_micro_topic_id"]
        ],
        on=["subgroup", "micro_topic_id"],
        how="inner",
    )
    annual = annual.merge(
        metadata[
            ["subgroup", "final_merge_group_id", "merged_micro_topic_id", "merged_topic_name_original"]
        ],
        on=["subgroup", "final_merge_group_id", "merged_micro_topic_id"],
        how="left",
    )
    grouped = (
        annual.groupby(
            [
                "subgroup",
                "source",
                "assigned_label",
                "final_merge_group_id",
                "merged_micro_topic_id",
                "merged_topic_name_original",
                "year",
            ],
            dropna=False,
        )
        .agg(
            member_micro_topic_ids_json=("micro_topic_id", lambda values: json_dumps(sorted({int(v) for v in values}))),
            year_focus_primary=("year_focus_primary", lambda values: " || ".join(stable_unique(values.astype(str).tolist(), limit=6))),
            year_focus_secondary=("year_focus_secondary", lambda values: " || ".join(stable_unique(values.astype(str).tolist(), limit=6))),
            year_keywords_interpreted=("year_keywords_interpreted", lambda values: " || ".join(stable_unique(values.astype(str).tolist(), limit=6))),
            year_frame_or_angle=("year_frame_or_angle", lambda values: " || ".join(stable_unique(values.astype(str).tolist(), limit=6))),
            year_evidence_note=("year_evidence_note", lambda values: " || ".join(stable_unique(values.astype(str).tolist(), limit=6))),
        )
        .reset_index()
        .rename(
            columns={
                "merged_micro_topic_id": "micro_topic_id",
                "merged_topic_name_original": "topic_name_original",
            }
        )
        .sort_values(["subgroup", "micro_topic_id", "year"])
        .reset_index(drop=True)
    )
    return grouped


def build_merged_year_evidence(
    mapping: pd.DataFrame,
    metadata: pd.DataFrame,
    year_evidence: pd.DataFrame,
) -> pd.DataFrame:
    evidence = year_evidence.merge(
        mapping[["subgroup", "micro_topic_id", "final_merge_group_id", "merged_micro_topic_id"]],
        on=["subgroup", "micro_topic_id"],
        how="inner",
    )
    evidence = evidence.merge(
        metadata[
            [
                "subgroup",
                "final_merge_group_id",
                "merged_micro_topic_id",
                "macro_topic_name",
                "merged_topic_name_original",
                "topic_size",
                "share_of_non_outlier",
                "overall_keywords",
            ]
        ],
        on=["subgroup", "final_merge_group_id", "merged_micro_topic_id"],
        how="left",
        suffixes=("", "_meta"),
    )
    grouped_rows: list[dict[str, Any]] = []
    for group_key, group in evidence.groupby(
        ["subgroup", "source", "assigned_label", "final_merge_group_id", "merged_micro_topic_id", "year"],
        dropna=False,
    ):
        subgroup, source, assigned_label, final_merge_group_id, merged_micro_topic_id, year = group_key
        year_words = stable_unique(
            [str(word) for value in group["year_specific_words"].tolist() for word in parse_json_list(value)],
            limit=10,
        )
        chunk_records = [
            record
            for value in group["chunk_records_json"].tolist()
            for record in parse_chunk_records(value)
        ]
        chunk_records = sorted(
            chunk_records,
            key=lambda item: float(item.get("micro_topic_probability", 0.0)),
            reverse=True,
        )[:5]
        grouped_rows.append(
            {
                "subgroup": subgroup,
                "source": source,
                "assigned_label": assigned_label,
                "macro_topic_name": str(group["macro_topic_name_meta"].iloc[0]),
                "final_merge_group_id": final_merge_group_id,
                "micro_topic_id": int(merged_micro_topic_id),
                "topic_name_original": str(group["merged_topic_name_original"].iloc[0]),
                "year": int(year),
                "year_frequency": int(group["year_frequency"].sum()),
                "topic_size": int(group["topic_size_meta"].iloc[0]),
                "share_of_non_outlier": float(group["share_of_non_outlier_meta"].iloc[0]),
                "year_specific_words": json_dumps(year_words),
                "overall_keywords": str(group["overall_keywords"].iloc[0]),
                "available_chunk_count": int(group["available_chunk_count"].sum()),
                "is_sparse_year": bool(group["available_chunk_count"].sum() < 5),
                "chunk_records_json": json_dumps(chunk_records),
                "chunk_id_1": chunk_records[0]["chunk_id"] if len(chunk_records) > 0 else "",
                "chunk_id_2": chunk_records[1]["chunk_id"] if len(chunk_records) > 1 else "",
                "chunk_id_3": chunk_records[2]["chunk_id"] if len(chunk_records) > 2 else "",
                "chunk_id_4": chunk_records[3]["chunk_id"] if len(chunk_records) > 3 else "",
                "chunk_id_5": chunk_records[4]["chunk_id"] if len(chunk_records) > 4 else "",
                "chunk_text_1": chunk_records[0]["text"] if len(chunk_records) > 0 else "",
                "chunk_text_2": chunk_records[1]["text"] if len(chunk_records) > 1 else "",
                "chunk_text_3": chunk_records[2]["text"] if len(chunk_records) > 2 else "",
                "chunk_text_4": chunk_records[3]["text"] if len(chunk_records) > 3 else "",
                "chunk_text_5": chunk_records[4]["text"] if len(chunk_records) > 4 else "",
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values(["subgroup", "micro_topic_id", "year"]).reset_index(drop=True)
    return grouped


def run_stage2_for_merged_groups(
    args: argparse.Namespace,
    selected_inputs_csv: Path,
    annual_summaries_csv: Path,
    year_evidence_csv: Path,
    output_dir: Path,
) -> pd.DataFrame:
    ensure_directory(output_dir)
    script_path = Path("paper_pipeline/scripts/run_micro_topic_evolution_synthesis.py")
    command = [
        sys.executable,
        str(script_path),
        "--selected-topics",
        str(selected_inputs_csv),
        "--annual-summaries",
        str(annual_summaries_csv),
        "--year-evidence",
        str(year_evidence_csv),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
        "--log-level",
        args.log_level,
    ]
    subprocess.run(command, check=True)
    return safe_read_csv(output_dir / "micro_topic_evolution_narratives.csv")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    mapping = pd.read_csv(args.mapping_csv)
    selected = pd.read_csv(args.selected_topics)
    profiles = pd.read_csv(args.profiles_csv)
    year_summaries = pd.read_csv(args.year_summaries_csv)
    year_evidence = pd.read_csv(args.year_evidence_csv)
    stage2_rows = load_stage2_rows(include_pending=True)

    selected["micro_topic_id"] = pd.to_numeric(selected["micro_topic_id"], errors="coerce")
    selected = selected.dropna(subset=["micro_topic_id"]).copy()
    selected["micro_topic_id"] = selected["micro_topic_id"].astype(int)
    profiles = profiles.rename(columns={"microtopic_id": "micro_topic_id"})
    profiles["micro_topic_id"] = pd.to_numeric(profiles["micro_topic_id"], errors="coerce")
    profiles = profiles.dropna(subset=["micro_topic_id"]).copy()
    profiles["micro_topic_id"] = profiles["micro_topic_id"].astype(int)

    groups = (
        mapping[["subgroup", "source", "assigned_label", "final_merge_group_id"]]
        .drop_duplicates()
        .sort_values(["subgroup", "final_merge_group_id"])
        .reset_index(drop=True)
    )
    groups["merged_micro_topic_id"] = groups.groupby("subgroup", dropna=False).cumcount().astype(int)
    mapping = mapping.merge(groups, on=["subgroup", "source", "assigned_label", "final_merge_group_id"], how="left")

    merged_members = (
        mapping.merge(selected, on=["subgroup", "source", "assigned_label", "micro_topic_id"], how="left")
        .merge(
            profiles[
                [
                    "subgroup",
                    "source",
                    "assigned_label",
                    "micro_topic_id",
                    "top_terms",
                    "representative_chunks",
                    "document_count",
                    "chunk_count",
                    "years_present",
                ]
            ],
            on=["subgroup", "source", "assigned_label", "micro_topic_id"],
            how="left",
        )
    )

    metadata = build_group_metadata(merged_members=merged_members, stage2_rows=stage2_rows)
    metadata = metadata.sort_values(["subgroup", "topic_size"], ascending=[True, False]).reset_index(drop=True)
    metadata["cumulative_share"] = metadata.groupby("subgroup", dropna=False)["share_of_non_outlier"].cumsum()

    merged_stage2_inputs = metadata[
        [
            "subgroup",
            "source",
            "assigned_label",
            "macro_topic_name",
            "merged_micro_topic_id",
            "merged_topic_name_original",
            "topic_size",
            "share_of_non_outlier",
            "cumulative_share",
            "active_year_count",
            "active_year_min",
            "active_year_max",
            "overall_keywords",
            "final_merge_group_id",
            "group_size",
            "member_micro_topic_ids_json",
        ]
    ].rename(
        columns={
            "merged_micro_topic_id": "micro_topic_id",
            "merged_topic_name_original": "topic_name_original",
        }
    )
    merged_stage2_inputs = merged_stage2_inputs.sort_values(["subgroup", "micro_topic_id"]).reset_index(drop=True)

    merged_annual = build_merged_stage1_annual_summaries(mapping=mapping, metadata=metadata, year_summaries=year_summaries)
    merged_year_evidence = build_merged_year_evidence(mapping=mapping, metadata=metadata, year_evidence=year_evidence)

    metadata_csv = args.output_root / "merged_microtopic_metadata.csv"
    annual_csv = args.output_root / "merged_microtopic_annual_summaries.csv"
    stage2_inputs_csv = args.output_root / "merged_microtopic_stage2_inputs.csv"
    year_evidence_csv = args.output_root / "merged_microtopic_year_evidence.csv"
    narratives_csv = args.output_root / "merged_microtopic_narratives.csv"
    summary_xlsx = args.output_root / "merged_microtopic_summary.xlsx"
    manifest_path = args.output_root / "merged_microtopic_outputs_manifest.json"

    metadata.to_csv(metadata_csv, index=False)
    merged_annual.to_csv(annual_csv, index=False)
    merged_stage2_inputs.to_csv(stage2_inputs_csv, index=False)
    merged_year_evidence.to_csv(year_evidence_csv, index=False)

    stage2_dir = args.stage2_output_dir or (args.output_root / "merged_stage2_evolution_summaries")
    if args.run_stage2:
        merged_stage2_narratives = run_stage2_for_merged_groups(
            args=args,
            selected_inputs_csv=stage2_inputs_csv,
            annual_summaries_csv=annual_csv,
            year_evidence_csv=year_evidence_csv,
            output_dir=stage2_dir,
        )
    else:
        merged_stage2_narratives = pd.DataFrame(columns=["final_merge_group_id", *STAGE2_CANONICAL_FIELDS])

    if not merged_stage2_narratives.empty:
        merged_stage2_narratives = merged_stage2_narratives.merge(
            metadata[["subgroup", "merged_micro_topic_id", "final_merge_group_id"]].rename(columns={"merged_micro_topic_id": "micro_topic_id"}),
            on=["subgroup", "micro_topic_id"],
            how="left",
        )
    elif "final_merge_group_id" not in merged_stage2_narratives.columns:
        merged_stage2_narratives["final_merge_group_id"] = pd.Series(dtype=str)

    merged_stage2_narratives.to_csv(narratives_csv, index=False)

    metadata_excel = sanitize_frame_for_excel(metadata)
    merged_stage2_inputs_excel = sanitize_frame_for_excel(merged_stage2_inputs)
    merged_annual_excel = sanitize_frame_for_excel(merged_annual)
    merged_year_evidence_excel = sanitize_frame_for_excel(merged_year_evidence)
    merged_stage2_narratives_excel = sanitize_frame_for_excel(merged_stage2_narratives)
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        metadata_excel.to_excel(writer, sheet_name="metadata", index=False)
        merged_stage2_inputs_excel.to_excel(writer, sheet_name="stage2_inputs", index=False)
        merged_annual_excel.to_excel(writer, sheet_name="annual_summaries", index=False)
        merged_year_evidence_excel.to_excel(writer, sheet_name="year_evidence", index=False)
        merged_stage2_narratives_excel.to_excel(writer, sheet_name="narratives", index=False)
        workbook_autofit(writer, "metadata", metadata_excel)
        workbook_autofit(writer, "stage2_inputs", merged_stage2_inputs_excel)
        workbook_autofit(writer, "annual_summaries", merged_annual_excel)
        workbook_autofit(writer, "year_evidence", merged_year_evidence_excel)
        workbook_autofit(writer, "narratives", merged_stage2_narratives_excel)

    write_json(
        manifest_path,
        {
            "mapping_csv": str(args.mapping_csv),
            "run_stage2": bool(args.run_stage2),
            "stage2_generation_note": "Merged Stage 2 uses newly aggregated Stage 1 annual summaries and year evidence; it does not reuse original Stage 2 narratives.",
            "row_counts": {
                "merged_groups": int(metadata.shape[0]),
                "merged_annual_rows": int(merged_annual.shape[0]),
                "merged_year_evidence_rows": int(merged_year_evidence.shape[0]),
                "merged_stage2_narrative_rows": int(merged_stage2_narratives.shape[0]),
            },
            "outputs": {
                "merged_microtopic_metadata_csv": str(metadata_csv),
                "merged_microtopic_annual_summaries_csv": str(annual_csv),
                "merged_microtopic_stage2_inputs_csv": str(stage2_inputs_csv),
                "merged_microtopic_year_evidence_csv": str(year_evidence_csv),
                "merged_microtopic_narratives_csv": str(narratives_csv),
                "merged_microtopic_summary_xlsx": str(summary_xlsx),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
