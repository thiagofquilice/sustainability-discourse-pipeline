#!/usr/bin/env python3
"""Materialize the merge mapping from grouped merge-review decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_GROUP_REVIEW_OUTPUT_ROOT,
    ensure_directory,
    json_dumps,
    normalize_text,
    safe_read_csv,
    write_json,
)


ALLOWED_DECISIONS = {"accept", "reject", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_GROUP_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--blank-group-decision", type=str, default="reject")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def auto_singleton_group_id(subgroup: str, micro_topic_id: int) -> str:
    return f"{subgroup}_S{micro_topic_id:03d}"


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = args.workbook or (args.output_root / "microtopic_merge_first_group_review.xlsx")
    if workbook.exists():
        group_summary = pd.read_excel(workbook, sheet_name="group_summary")
        group_members = pd.read_excel(workbook, sheet_name="group_members")
        singletons = pd.read_excel(workbook, sheet_name="singletons")
    else:
        group_summary = safe_read_csv(args.output_root / "proposed_group_summary.csv")
        group_members = safe_read_csv(args.output_root / "proposed_group_members.csv")
        singletons = safe_read_csv(args.output_root / "singleton_topics.csv")

    if group_summary.empty and group_members.empty and singletons.empty:
        raise SystemExit("No grouped merge review inputs found to materialize mapping.")
    return group_summary, group_members, singletons


def normalize_summary(frame: pd.DataFrame, blank_decision: str) -> pd.DataFrame:
    required = {"subgroup", "proposed_group_id", "group_decision"}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"group_summary is missing required columns: {missing}")

    summary = frame.copy()
    summary["subgroup"] = summary["subgroup"].astype(str)
    summary["proposed_group_id"] = summary["proposed_group_id"].astype(str)
    summary["group_decision_raw"] = summary["group_decision"].map(normalize_text)
    summary["group_review_notes"] = summary.get("group_review_notes", "").map(normalize_text) if "group_review_notes" in summary.columns else ""
    summary["group_decision"] = summary["group_decision_raw"].str.lower()
    summary.loc[summary["group_decision"] == "", "group_decision"] = blank_decision
    invalid = sorted({value for value in summary["group_decision"].tolist() if value not in ALLOWED_DECISIONS})
    if invalid:
        raise SystemExit(f"Invalid group_decision values found: {invalid}")
    return summary


def normalize_members(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subgroup",
        "source",
        "assigned_label",
        "proposed_group_id",
        "micro_topic_id",
        "manual_split_group_id",
    }
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"group_members is missing required columns: {missing}")

    members = frame.copy()
    members["subgroup"] = members["subgroup"].astype(str)
    members["source"] = members["source"].astype(str)
    members["assigned_label"] = members["assigned_label"].astype(str)
    members["proposed_group_id"] = members["proposed_group_id"].astype(str)
    members["micro_topic_id"] = pd.to_numeric(members["micro_topic_id"], errors="coerce")
    members = members.dropna(subset=["micro_topic_id"]).copy()
    members["micro_topic_id"] = members["micro_topic_id"].astype(int)
    members["manual_split_group_id"] = members["manual_split_group_id"].map(normalize_text)
    members["manual_member_notes"] = members.get("manual_member_notes", "").map(normalize_text) if "manual_member_notes" in members.columns else ""
    members["topic_name_original"] = members.get("topic_name_original", "").astype(str)
    return members


def normalize_singletons(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"subgroup", "source", "assigned_label", "micro_topic_id"}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"singletons is missing required columns: {missing}")

    singletons = frame.copy()
    singletons["subgroup"] = singletons["subgroup"].astype(str)
    singletons["source"] = singletons["source"].astype(str)
    singletons["assigned_label"] = singletons["assigned_label"].astype(str)
    singletons["micro_topic_id"] = pd.to_numeric(singletons["micro_topic_id"], errors="coerce")
    singletons = singletons.dropna(subset=["micro_topic_id"]).copy()
    singletons["micro_topic_id"] = singletons["micro_topic_id"].astype(int)
    singletons["topic_name_original"] = singletons.get("topic_name_original", "").astype(str)
    return singletons


def validate_inputs(summary: pd.DataFrame, members: pd.DataFrame, singletons: pd.DataFrame) -> None:
    summary_keys = set(zip(summary["subgroup"], summary["proposed_group_id"]))
    member_keys = set(zip(members["subgroup"], members["proposed_group_id"]))
    missing_summary = sorted(member_keys - summary_keys)
    if missing_summary:
        raise SystemExit(f"Found proposed groups in group_members with no group_summary row: {missing_summary[:10]}")

    member_topic_keys = set(zip(members["subgroup"], members["micro_topic_id"]))
    singleton_topic_keys = set(zip(singletons["subgroup"], singletons["micro_topic_id"]))
    overlap = sorted(member_topic_keys.intersection(singleton_topic_keys))
    if overlap:
        raise SystemExit(f"Found topics listed in both group_members and singletons: {overlap[:10]}")

    duplicates = (
        members.groupby(["subgroup", "micro_topic_id"], dropna=False)
        .size()
        .reset_index(name="n")
        .loc[lambda df: df["n"] > 1]
    )
    if not duplicates.empty:
        raise SystemExit("Found duplicate topic membership rows in group_members.")

    singleton_duplicates = (
        singletons.groupby(["subgroup", "micro_topic_id"], dropna=False)
        .size()
        .reset_index(name="n")
        .loc[lambda df: df["n"] > 1]
    )
    if not singleton_duplicates.empty:
        raise SystemExit("Found duplicate topic rows in singletons.")


def build_mapping(
    summary: pd.DataFrame,
    members: pd.DataFrame,
    singletons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = {
        (str(row.subgroup), str(row.proposed_group_id)): row._asdict()
        for row in summary.itertuples(index=False)
    }
    audit_rows: list[dict] = []

    split_rows = members.merge(
        summary[["subgroup", "proposed_group_id", "group_decision"]],
        on=["subgroup", "proposed_group_id"],
        how="left",
    )
    split_rows = split_rows.loc[split_rows["group_decision"] == "split"].copy()
    if not split_rows.empty:
        missing_split = split_rows.loc[split_rows["manual_split_group_id"].map(normalize_text) == ""]
        if not missing_split.empty:
            raise SystemExit("Found split groups with blank manual_split_group_id values.")
        reused = (
            split_rows.groupby(["subgroup", "manual_split_group_id"], dropna=False)["proposed_group_id"]
            .nunique()
            .reset_index(name="group_count")
            .loc[lambda df: df["group_count"] > 1]
        )
        if not reused.empty:
            raise SystemExit(
                "manual_split_group_id values may not be reused across proposed groups within the same subgroup: "
                + ", ".join(
                    f"{row.subgroup}:{row.manual_split_group_id}"
                    for row in reused.itertuples(index=False)
                )
            )

    for row in members.itertuples(index=False):
        summary_row = summary_lookup[(str(row.subgroup), str(row.proposed_group_id))]
        decision = str(summary_row["group_decision"])
        decision_raw = normalize_text(summary_row.get("group_decision_raw", ""))
        group_review_notes = normalize_text(summary_row.get("group_review_notes", ""))
        member_notes = normalize_text(getattr(row, "manual_member_notes", ""))
        compiled_notes = " || ".join([value for value in [group_review_notes, member_notes] if value])

        if decision == "accept":
            final_merge_group_id = str(row.proposed_group_id)
            manual_merge_group_id = final_merge_group_id
            decision_source = "group_accept"
        elif decision == "reject":
            final_merge_group_id = auto_singleton_group_id(str(row.subgroup), int(row.micro_topic_id))
            manual_merge_group_id = ""
            decision_source = "group_blank_default_reject" if decision_raw == "" else "group_reject_singleton"
        elif decision == "split":
            final_merge_group_id = normalize_text(row.manual_split_group_id)
            manual_merge_group_id = final_merge_group_id
            decision_source = "group_split"
        else:
            raise SystemExit(f"Unexpected group decision: {decision}")

        audit_rows.append(
            {
                "subgroup": str(row.subgroup),
                "source": str(row.source),
                "assigned_label": str(row.assigned_label),
                "micro_topic_id": int(row.micro_topic_id),
                "topic_name_original": str(getattr(row, "topic_name_original", "")),
                "proposed_group_id": str(row.proposed_group_id),
                "group_decision_raw": decision_raw,
                "effective_group_decision": decision,
                "manual_split_group_id": normalize_text(row.manual_split_group_id),
                "group_review_notes": group_review_notes,
                "manual_member_notes": member_notes,
                "final_merge_group_id": final_merge_group_id,
                "manual_merge_group_id": manual_merge_group_id,
                "manual_merge_confidence": "",
                "manual_merge_notes": compiled_notes,
                "decision_source": decision_source,
            }
        )

    for row in singletons.itertuples(index=False):
        audit_rows.append(
            {
                "subgroup": str(row.subgroup),
                "source": str(row.source),
                "assigned_label": str(row.assigned_label),
                "micro_topic_id": int(row.micro_topic_id),
                "topic_name_original": str(getattr(row, "topic_name_original", "")),
                "proposed_group_id": "",
                "group_decision_raw": "",
                "effective_group_decision": "singleton_auto",
                "manual_split_group_id": "",
                "group_review_notes": "",
                "manual_member_notes": "",
                "final_merge_group_id": auto_singleton_group_id(str(row.subgroup), int(row.micro_topic_id)),
                "manual_merge_group_id": "",
                "manual_merge_confidence": "",
                "manual_merge_notes": "",
                "decision_source": "no_proposed_group_singleton",
            }
        )

    audit = pd.DataFrame(audit_rows)
    key_counts = (
        audit.groupby(["subgroup", "micro_topic_id"], dropna=False)
        .size()
        .reset_index(name="n")
        .loc[lambda df: df["n"] > 1]
    )
    if not key_counts.empty:
        raise SystemExit("A topic was assigned more than once while materializing group review decisions.")

    mixed = (
        audit.groupby("final_merge_group_id", dropna=False)["subgroup"]
        .nunique()
        .reset_index(name="subgroup_count")
        .loc[lambda df: df["subgroup_count"] > 1]
    )
    if not mixed.empty:
        raise SystemExit(
            "Found final merge group IDs reused across subgroups: "
            + ", ".join(mixed["final_merge_group_id"].astype(str).tolist())
        )

    group_sizes = audit.groupby("final_merge_group_id", dropna=False).size().reset_index(name="group_size")
    members = (
        audit.groupby("final_merge_group_id", dropna=False)["micro_topic_id"]
        .agg(lambda values: sorted({int(value) for value in values}))
        .reset_index(name="group_members")
    )
    group_meta = group_sizes.merge(members, on="final_merge_group_id", how="left")
    audit = audit.merge(group_meta, on="final_merge_group_id", how="left")
    audit["is_singleton_group"] = audit["group_size"] == 1
    audit["group_members_json"] = audit["group_members"].map(json_dumps)

    mapping = audit[
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
            "proposed_group_id",
            "effective_group_decision",
            "group_decision_raw",
            "manual_split_group_id",
        ]
    ].sort_values(["subgroup", "final_merge_group_id", "micro_topic_id"]).reset_index(drop=True)

    group_roster = (
        audit.groupby(["subgroup", "source", "assigned_label", "final_merge_group_id"], dropna=False)
        .agg(
            group_size=("group_size", "max"),
            is_singleton_group=("is_singleton_group", "max"),
            group_members_json=("group_members_json", "first"),
            member_topic_names=("topic_name_original", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            proposed_group_ids=("proposed_group_id", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            effective_group_decisions=("effective_group_decision", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            decision_sources=("decision_source", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            manual_merge_notes_compiled=("manual_merge_notes", lambda values: " || ".join([normalize_text(value) for value in values if normalize_text(value)])),
        )
        .reset_index()
        .sort_values(["subgroup", "final_merge_group_id"])
        .reset_index(drop=True)
    )
    return mapping, audit, group_roster


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    blank_decision = normalize_text(args.blank_group_decision).lower() or "reject"
    if blank_decision not in ALLOWED_DECISIONS:
        raise SystemExit(f"blank-group-decision must be one of {sorted(ALLOWED_DECISIONS)}")

    group_summary, group_members, singletons = load_inputs(args)
    group_summary = normalize_summary(group_summary, blank_decision=blank_decision)
    group_members = normalize_members(group_members)
    singletons = normalize_singletons(singletons)
    validate_inputs(group_summary, group_members, singletons)
    mapping, audit, group_roster = build_mapping(group_summary, group_members, singletons)

    mapping_csv = args.output_root / "microtopic_to_merged_group.csv"
    group_roster_csv = args.output_root / "merged_group_roster.csv"
    audit_csv = args.output_root / "group_review_decision_audit.csv"
    manifest_path = args.output_root / "group_review_materialization_manifest.json"

    mapping.to_csv(mapping_csv, index=False)
    group_roster.to_csv(group_roster_csv, index=False)
    audit.to_csv(audit_csv, index=False)

    write_json(
        manifest_path,
        {
            "blank_group_decision": blank_decision,
            "row_counts": {
                "microtopic_rows": int(mapping.shape[0]),
                "merged_groups": int(group_roster.shape[0]),
                "final_singleton_groups": int(group_roster.loc[group_roster["is_singleton_group"]].shape[0]),
                "accepted_groups": int((audit["decision_source"] == "group_accept").sum()),
                "split_member_rows": int((audit["decision_source"] == "group_split").sum()),
                "rejected_member_rows": int(audit["decision_source"].isin({"group_reject_singleton", "group_blank_default_reject"}).sum()),
                "blank_default_reject_member_rows": int((audit["decision_source"] == "group_blank_default_reject").sum()),
                "singleton_no_proposed_group_rows": int((audit["decision_source"] == "no_proposed_group_singleton").sum()),
            },
            "outputs": {
                "mapping_csv": str(mapping_csv),
                "group_roster_csv": str(group_roster_csv),
                "group_review_decision_audit_csv": str(audit_csv),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
