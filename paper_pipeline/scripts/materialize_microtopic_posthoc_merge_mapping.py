#!/usr/bin/env python3
"""Materialize the official post-hoc microtopic merge mapping from manual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_OUTPUT_ROOT,
    ensure_directory,
    json_dumps,
    normalize_text,
    safe_read_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGE_OUTPUT_ROOT)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--roster-csv", type=Path, default=None)
    parser.add_argument("--sheet-name", type=str, default="microtopic_roster")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_roster(args: argparse.Namespace) -> pd.DataFrame:
    default_workbook = args.output_root / "microtopic_posthoc_merge_review.xlsx"
    merge_first_workbook = args.output_root / "microtopic_merge_first_review.xlsx"
    workbook = args.workbook or (default_workbook if default_workbook.exists() else merge_first_workbook)
    roster_csv = args.roster_csv or (args.output_root / "microtopic_roster.csv")
    if workbook.exists():
        frame = pd.read_excel(workbook, sheet_name=args.sheet_name)
    else:
        frame = safe_read_csv(roster_csv)
    if frame.empty:
        raise SystemExit("No microtopic roster found to materialize merge mapping.")
    return frame


def auto_singleton_group_id(subgroup: str, micro_topic_id: int) -> str:
    return f"{subgroup}_S{micro_topic_id:03d}"


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    roster = load_roster(args)
    required = {
        "subgroup",
        "source",
        "assigned_label",
        "micro_topic_id",
        "manual_merge_group_id",
        "manual_merge_confidence",
        "manual_merge_notes",
    }
    missing = [column for column in required if column not in roster.columns]
    if missing:
        raise SystemExit(f"Roster is missing required columns: {missing}")

    roster["subgroup"] = roster["subgroup"].astype(str)
    roster["source"] = roster["source"].astype(str)
    roster["assigned_label"] = roster["assigned_label"].astype(str)
    roster["micro_topic_id"] = pd.to_numeric(roster["micro_topic_id"], errors="coerce")
    roster = roster.dropna(subset=["micro_topic_id"]).copy()
    roster["micro_topic_id"] = roster["micro_topic_id"].astype(int)
    roster["manual_merge_group_id"] = roster["manual_merge_group_id"].map(normalize_text)
    roster["manual_merge_confidence"] = roster["manual_merge_confidence"].map(normalize_text)
    roster["manual_merge_notes"] = roster["manual_merge_notes"].map(normalize_text)
    if "topic_name_original" not in roster.columns:
        roster["topic_name_original"] = ""
    if "topic_label_refined" not in roster.columns:
        roster["topic_label_refined"] = ""

    manual_groups = roster.loc[roster["manual_merge_group_id"] != "", ["subgroup", "manual_merge_group_id"]].drop_duplicates()
    mixed = manual_groups.groupby("manual_merge_group_id", dropna=False)["subgroup"].nunique().reset_index(name="subgroup_count")
    mixed = mixed.loc[mixed["subgroup_count"] > 1]
    if not mixed.empty:
        raise SystemExit(
            "Found manual_merge_group_id values reused across subgroups: "
            + ", ".join(mixed["manual_merge_group_id"].astype(str).tolist())
        )

    roster["final_merge_group_id"] = roster.apply(
        lambda row: row["manual_merge_group_id"]
        if row["manual_merge_group_id"]
        else auto_singleton_group_id(str(row["subgroup"]), int(row["micro_topic_id"])),
        axis=1,
    )
    roster["decision_source"] = roster["manual_merge_group_id"].map(lambda value: "manual" if value else "singleton_auto")

    group_sizes = roster.groupby("final_merge_group_id", dropna=False).size().reset_index(name="group_size")
    members = (
        roster.groupby("final_merge_group_id", dropna=False)["micro_topic_id"]
        .agg(lambda values: sorted({int(value) for value in values}))
        .reset_index(name="group_members")
    )
    group_meta = group_sizes.merge(members, on="final_merge_group_id", how="left")
    roster = roster.merge(group_meta, on="final_merge_group_id", how="left")
    roster["is_singleton_group"] = roster["group_size"] == 1
    roster["group_members_json"] = roster["group_members"].map(json_dumps)

    mapping = roster[
        [
            "subgroup",
            "source",
            "assigned_label",
            "micro_topic_id",
            "final_merge_group_id",
            "is_singleton_group",
            "group_size",
            "group_members_json",
            "manual_merge_group_id",
            "manual_merge_confidence",
            "manual_merge_notes",
            "decision_source",
        ]
    ].sort_values(["subgroup", "final_merge_group_id", "micro_topic_id"]).reset_index(drop=True)

    group_roster = (
        roster.groupby(["subgroup", "source", "assigned_label", "final_merge_group_id"], dropna=False)
        .agg(
            group_size=("group_size", "max"),
            is_singleton_group=("is_singleton_group", "max"),
            group_members_json=("group_members_json", "first"),
            member_topic_names=("topic_name_original", lambda values: json_dumps(sorted({str(value) for value in values}))),
            member_refined_labels=("topic_label_refined", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            manual_merge_confidence_values=("manual_merge_confidence", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            manual_merge_notes_compiled=("manual_merge_notes", lambda values: " || ".join([normalize_text(value) for value in values if normalize_text(value)])),
            decision_source=("decision_source", "first"),
        )
        .reset_index()
        .sort_values(["subgroup", "final_merge_group_id"])
        .reset_index(drop=True)
    )

    mapping_csv = args.output_root / "microtopic_to_merged_group.csv"
    group_roster_csv = args.output_root / "merged_group_roster.csv"
    manifest_path = args.output_root / "microtopic_posthoc_merge_mapping_manifest.json"

    mapping.to_csv(mapping_csv, index=False)
    group_roster.to_csv(group_roster_csv, index=False)
    write_json(
        manifest_path,
        {
            "row_counts": {
                "microtopic_rows": int(mapping.shape[0]),
                "merged_groups": int(group_roster.shape[0]),
                "manual_groups": int(group_roster.loc[group_roster["decision_source"] == "manual"].shape[0]),
                "singleton_groups": int(group_roster.loc[group_roster["is_singleton_group"]].shape[0]),
            },
            "outputs": {
                "mapping_csv": str(mapping_csv),
                "group_roster_csv": str(group_roster_csv),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
