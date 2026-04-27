#!/usr/bin/env python3
"""Build evolution inputs for all approved merged groups after merge-first review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_micro_topic_evolution_inputs import build_year_evidence
from micro_topic_evolution_common import SOURCE_ORDER, configure_logging, ensure_directory, read_json, write_json
from microtopic_posthoc_merge_common import (
    MERGED_MICRO_ROOT,
    MERGE_FIRST_EVOLUTION_OUTPUT_ROOT,
    MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=MERGED_MICRO_ROOT)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_EVOLUTION_OUTPUT_ROOT)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT / "microtopic_to_merged_group.csv",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--max-topics-per-subgroup", type=int, default=10)
    parser.add_argument("--min-topics-per-subgroup", type=int, default=1)
    parser.add_argument("--max-chunks-per-year", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def approve_all_topics(topic_info: pd.DataFrame) -> pd.DataFrame:
    valid = topic_info.loc[topic_info["Topic"] != -1].copy()
    valid = valid.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
    total = float(valid["Count"].sum())
    if total > 0:
        valid["share_of_non_outlier"] = valid["Count"] / total
        valid["cumulative_share"] = valid["share_of_non_outlier"].cumsum()
    else:
        valid["share_of_non_outlier"] = 0.0
        valid["cumulative_share"] = 0.0
    valid["approved_after_merge"] = True
    return valid


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    mapping = pd.read_csv(args.mapping_csv)
    selected_topic_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    yearly_rows: list[dict] = []
    audit_rows: list[dict] = []

    subgroup_dirs = sorted(path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary")
    for subgroup_dir in subgroup_dirs:
        manifest_path = subgroup_dir / "manifest.json"
        topic_info_path = subgroup_dir / "topic_info.csv"
        topics_over_time_path = subgroup_dir / "topics_over_time.csv"
        document_topics_path = subgroup_dir / "document_topics.parquet"
        if not all(path.exists() for path in [manifest_path, topic_info_path, topics_over_time_path, document_topics_path]):
            continue

        manifest = read_json(manifest_path)
        topic_info = pd.read_csv(topic_info_path)
        topics_over_time = pd.read_csv(topics_over_time_path)
        document_topics = pd.read_parquet(document_topics_path)

        approved = approve_all_topics(topic_info)
        topic_rows, group_year_rows = build_year_evidence(
            subgroup=subgroup_dir.name,
            manifest=manifest,
            selected_topics=approved,
            topics_over_time=topics_over_time,
            document_topics=document_topics,
            max_chunks_per_year=args.max_chunks_per_year,
        )

        approved_meta = approved[
            [
                "Topic",
                "Name",
                "Count",
                "share_of_non_outlier",
                "cumulative_share",
                "approved_after_merge",
                "final_merge_group_id",
                "group_size",
                "is_singleton_group",
                "member_micro_topic_ids_json",
            ]
        ].copy()
        approved_meta = approved_meta.rename(columns={"Topic": "micro_topic_id", "Name": "topic_name_original"})
        topic_rows_frame = pd.DataFrame(topic_rows).merge(
            approved_meta,
            on=["micro_topic_id", "topic_name_original"],
            how="left",
        )
        selected_topic_rows.extend(topic_rows_frame.to_dict(orient="records"))
        yearly_rows.extend(group_year_rows)

        subgroup_rows.append(
            {
                "subgroup": subgroup_dir.name,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "status": manifest["status"],
                "warning": manifest.get("warning", ""),
                "row_count": int(manifest["row_count"]),
                "selected_topic_count": int(len(approved)),
                "selected_topic_coverage": float(approved["share_of_non_outlier"].sum()) if not approved.empty else 0.0,
                "selected_topic_ids": json.dumps(approved["Topic"].astype(int).tolist()),
                "selected_topic_names": json.dumps(approved["Name"].astype(str).tolist(), ensure_ascii=False),
                "selected_final_merge_group_ids": json.dumps(approved["final_merge_group_id"].astype(str).tolist(), ensure_ascii=False),
                "approved_group_count": int(len(approved)),
                "approved_group_coverage": float(approved["share_of_non_outlier"].sum()) if not approved.empty else 0.0,
            }
        )

        selected_group_ids = set(approved["final_merge_group_id"].astype(str).tolist())
        subgroup_mapping = mapping.loc[mapping["subgroup"] == subgroup_dir.name].copy()
        subgroup_mapping["selected_after_merge"] = subgroup_mapping["final_merge_group_id"].astype(str).isin(selected_group_ids)
        subgroup_mapping["approved_after_merge"] = subgroup_mapping["selected_after_merge"]
        audit_rows.extend(subgroup_mapping.to_dict(orient="records"))

    selected_topics_frame = pd.DataFrame(selected_topic_rows)
    subgroup_frame = pd.DataFrame(subgroup_rows)
    yearly_frame = pd.DataFrame(yearly_rows)
    audit_frame = pd.DataFrame(audit_rows)

    if not selected_topics_frame.empty:
        selected_topics_frame["source"] = pd.Categorical(selected_topics_frame["source"], categories=SOURCE_ORDER, ordered=True)
        selected_topics_frame = selected_topics_frame.sort_values(["source", "assigned_label", "micro_topic_id"]).reset_index(drop=True)
    if not subgroup_frame.empty:
        subgroup_frame["source"] = pd.Categorical(subgroup_frame["source"], categories=SOURCE_ORDER, ordered=True)
        subgroup_frame = subgroup_frame.sort_values(["source", "assigned_label"]).reset_index(drop=True)
    if not yearly_frame.empty:
        yearly_frame["source"] = pd.Categorical(yearly_frame["source"], categories=SOURCE_ORDER, ordered=True)
        yearly_frame = yearly_frame.sort_values(["source", "assigned_label", "micro_topic_id", "year"]).reset_index(drop=True)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values(["subgroup", "final_merge_group_id", "micro_topic_id"]).reset_index(drop=True)

    selected_topics_csv = args.output_root / "selected_micro_topics.csv"
    subgroup_csv = args.output_root / "selected_micro_topics_by_subgroup.csv"
    year_evidence_csv = args.output_root / "micro_topic_year_evidence.csv"
    year_evidence_jsonl = args.output_root / "micro_topic_year_evidence.jsonl"
    selected_alias_csv = args.output_root / "selected_merged_microtopics.csv"
    subgroup_alias_csv = args.output_root / "selected_merged_microtopics_by_subgroup.csv"
    year_alias_csv = args.output_root / "merged_micro_topic_year_evidence.csv"
    audit_csv = args.output_root / "selected_merged_microtopic_audit.csv"
    approved_groups_csv = args.output_root / "approved_merged_groups.csv"
    approved_groups_by_subgroup_csv = args.output_root / "approved_merged_groups_by_subgroup.csv"
    approved_year_csv = args.output_root / "approved_merged_group_year_evidence.csv"
    approved_year_jsonl = args.output_root / "approved_merged_group_year_evidence.jsonl"
    approved_audit_csv = args.output_root / "approved_merged_group_audit.csv"

    selected_topics_frame.to_csv(selected_topics_csv, index=False)
    subgroup_frame.to_csv(subgroup_csv, index=False)
    yearly_frame.to_csv(year_evidence_csv, index=False)
    write_jsonl(year_evidence_jsonl, yearly_frame)

    selected_topics_frame.to_csv(selected_alias_csv, index=False)
    subgroup_frame.to_csv(subgroup_alias_csv, index=False)
    yearly_frame.to_csv(year_alias_csv, index=False)
    audit_frame.to_csv(audit_csv, index=False)
    selected_topics_frame.to_csv(approved_groups_csv, index=False)
    subgroup_frame.to_csv(approved_groups_by_subgroup_csv, index=False)
    yearly_frame.to_csv(approved_year_csv, index=False)
    write_jsonl(approved_year_jsonl, yearly_frame)
    audit_frame.to_csv(approved_audit_csv, index=False)

    write_json(
        args.output_root / "selection_manifest.json",
        {
            "micro_root": str(args.micro_root),
            "mapping_csv": str(args.mapping_csv),
            "selection_mode": "all_approved_merged_groups",
            "legacy_selection_parameters_ignored": {
                "coverage_threshold": args.coverage_threshold,
                "max_topics_per_subgroup": args.max_topics_per_subgroup,
                "min_topics_per_subgroup": args.min_topics_per_subgroup,
            },
            "max_chunks_per_year": args.max_chunks_per_year,
            "subgroup_count": int(subgroup_frame.shape[0]),
            "selected_merged_microtopic_count": int(selected_topics_frame.shape[0]),
            "approved_merged_group_count": int(selected_topics_frame.shape[0]),
            "annual_evidence_row_count": int(yearly_frame.shape[0]),
            "outputs": {
                "selected_micro_topics": str(selected_topics_csv),
                "selected_micro_topics_by_subgroup": str(subgroup_csv),
                "micro_topic_year_evidence": str(year_evidence_csv),
                "micro_topic_year_evidence_jsonl": str(year_evidence_jsonl),
                "selected_merged_microtopics": str(selected_alias_csv),
                "selected_merged_microtopics_by_subgroup": str(subgroup_alias_csv),
                "merged_micro_topic_year_evidence": str(year_alias_csv),
                "selected_merged_microtopic_audit": str(audit_csv),
                "approved_merged_groups": str(approved_groups_csv),
                "approved_merged_groups_by_subgroup": str(approved_groups_by_subgroup_csv),
                "approved_merged_group_year_evidence": str(approved_year_csv),
                "approved_merged_group_year_evidence_jsonl": str(approved_year_jsonl),
                "approved_merged_group_audit": str(approved_audit_csv),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
