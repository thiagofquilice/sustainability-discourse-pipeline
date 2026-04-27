#!/usr/bin/env python3
"""Materialize the merge mapping from hierarchical BERTopic review decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
    ensure_directory,
    json_dumps,
    normalize_text,
    safe_read_csv,
    write_json,
)


ALLOWED_DECISIONS = {"accept", "reject", "review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--blank-group-decision", type=str, default="reject")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def auto_singleton_group_id(subgroup: str, micro_topic_id: int) -> str:
    return f"{subgroup}_S{micro_topic_id:03d}"


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = args.workbook or (args.output_root / "microtopic_merge_first_hierarchical_review.xlsx")
    if workbook.exists():
        summary_groups = pd.read_excel(workbook, sheet_name="summary_groups")
        group_members = pd.read_excel(workbook, sheet_name="group_members")
        singletons = pd.read_excel(workbook, sheet_name="singletons")
    else:
        summary_groups = safe_read_csv(args.output_root / "proposed_branchlet_summary.csv")
        group_members = safe_read_csv(args.output_root / "proposed_branchlet_members.csv")
        singletons = safe_read_csv(args.output_root / "singleton_topics.csv")

    if summary_groups.empty and group_members.empty and singletons.empty:
        raise SystemExit("No hierarchical review inputs found to materialize mapping.")
    return summary_groups, group_members, singletons


def normalize_summary(frame: pd.DataFrame, blank_decision: str) -> pd.DataFrame:
    required = {"subgroup", "proposed_branchlet_id", "group_decision"}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"summary_groups is missing required columns: {missing}")

    summary = frame.copy()
    summary["subgroup"] = summary["subgroup"].astype(str)
    summary["source"] = summary.get("source", "").astype(str)
    summary["assigned_label"] = summary.get("assigned_label", "").astype(str)
    summary["proposed_branchlet_id"] = summary["proposed_branchlet_id"].astype(str)
    summary["group_decision_raw"] = summary["group_decision"].map(normalize_text)
    summary["group_review_notes"] = summary.get("group_review_notes", "").map(normalize_text)
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
        "proposed_branchlet_id",
        "micro_topic_id",
        "manual_review_group_id",
    }
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"group_members is missing required columns: {missing}")

    members = frame.copy()
    members["subgroup"] = members["subgroup"].astype(str)
    members["source"] = members["source"].astype(str)
    members["assigned_label"] = members["assigned_label"].astype(str)
    members["proposed_branchlet_id"] = members["proposed_branchlet_id"].astype(str)
    members["micro_topic_id"] = pd.to_numeric(members["micro_topic_id"], errors="coerce")
    members = members.dropna(subset=["micro_topic_id"]).copy()
    members["micro_topic_id"] = members["micro_topic_id"].astype(int)
    members["manual_review_group_id"] = members["manual_review_group_id"].map(normalize_text)
    members["manual_member_notes"] = members.get("manual_member_notes", "").map(normalize_text)
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
    summary_keys = set(zip(summary["subgroup"], summary["proposed_branchlet_id"]))
    member_keys = set(zip(members["subgroup"], members["proposed_branchlet_id"]))
    missing_summary = sorted(member_keys - summary_keys)
    if missing_summary:
        raise SystemExit(f"Found proposed branchlets in group_members with no summary_groups row: {missing_summary[:10]}")

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_lookup = {
        (str(row.subgroup), str(row.proposed_branchlet_id)): row._asdict()
        for row in summary.itertuples(index=False)
    }
    audit_rows: list[dict] = []

    review_rows = members.merge(
        summary[["subgroup", "proposed_branchlet_id", "group_decision"]],
        on=["subgroup", "proposed_branchlet_id"],
        how="left",
    )
    review_rows = review_rows.loc[review_rows["group_decision"] == "review"].copy()
    if not review_rows.empty:
        missing_review = review_rows.loc[review_rows["manual_review_group_id"].map(normalize_text) == ""]
        if not missing_review.empty:
            raise SystemExit("Found review branchlets with blank manual_review_group_id values.")
        reused = (
            review_rows.groupby(["subgroup", "manual_review_group_id"], dropna=False)["proposed_branchlet_id"]
            .nunique()
            .reset_index(name="branchlet_count")
            .loc[lambda df: df["branchlet_count"] > 1]
        )
        if not reused.empty:
            raise SystemExit(
                "manual_review_group_id values may not be reused across proposed branchlets within the same subgroup: "
                + ", ".join(f"{row.subgroup}:{row.manual_review_group_id}" for row in reused.itertuples(index=False))
            )

    for row in members.itertuples(index=False):
        summary_row = summary_lookup[(str(row.subgroup), str(row.proposed_branchlet_id))]
        decision = str(summary_row["group_decision"])
        decision_raw = normalize_text(summary_row.get("group_decision_raw", ""))
        review_notes = normalize_text(summary_row.get("group_review_notes", ""))
        member_notes = normalize_text(getattr(row, "manual_member_notes", ""))
        compiled_notes = " || ".join([value for value in [review_notes, member_notes] if value])

        if decision == "accept":
            final_merge_group_id = str(row.proposed_branchlet_id)
            manual_merge_group_id = final_merge_group_id
            decision_source = "hierarchical_accept"
        elif decision == "reject":
            final_merge_group_id = auto_singleton_group_id(str(row.subgroup), int(row.micro_topic_id))
            manual_merge_group_id = ""
            decision_source = (
                "hierarchical_blank_default_reject" if decision_raw == "" else "hierarchical_reject_singleton"
            )
        elif decision == "review":
            final_merge_group_id = normalize_text(row.manual_review_group_id)
            manual_merge_group_id = final_merge_group_id
            decision_source = "hierarchical_review_manual"
        else:
            raise SystemExit(f"Unexpected group decision: {decision}")

        audit_rows.append(
            {
                "subgroup": str(row.subgroup),
                "source": str(row.source),
                "assigned_label": str(row.assigned_label),
                "micro_topic_id": int(row.micro_topic_id),
                "topic_name_original": str(getattr(row, "topic_name_original", "")),
                "proposed_branchlet_id": str(row.proposed_branchlet_id),
                "group_decision_raw": decision_raw,
                "effective_group_decision": decision,
                "manual_review_group_id": normalize_text(row.manual_review_group_id),
                "group_review_notes": review_notes,
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
                "proposed_branchlet_id": "",
                "group_decision_raw": "",
                "effective_group_decision": "singleton_auto",
                "manual_review_group_id": "",
                "group_review_notes": "",
                "manual_member_notes": "",
                "final_merge_group_id": auto_singleton_group_id(str(row.subgroup), int(row.micro_topic_id)),
                "manual_merge_group_id": "",
                "manual_merge_confidence": "",
                "manual_merge_notes": "",
                "decision_source": "no_branchlet_singleton",
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
        raise SystemExit("A topic was assigned more than once while materializing hierarchical review decisions.")

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
    member_lists = (
        audit.groupby("final_merge_group_id", dropna=False)["micro_topic_id"]
        .agg(lambda values: sorted({int(value) for value in values}))
        .reset_index(name="group_members")
    )
    group_meta = group_sizes.merge(member_lists, on="final_merge_group_id", how="left")
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
            "proposed_branchlet_id",
            "effective_group_decision",
            "group_decision_raw",
            "manual_review_group_id",
        ]
    ].sort_values(["subgroup", "final_merge_group_id", "micro_topic_id"]).reset_index(drop=True)

    group_roster = (
        audit.groupby(["subgroup", "source", "assigned_label", "final_merge_group_id"], dropna=False)
        .agg(
            group_size=("group_size", "max"),
            is_singleton_group=("is_singleton_group", "max"),
            group_members_json=("group_members_json", "first"),
            member_topic_names=("topic_name_original", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
            proposed_branchlet_ids=("proposed_branchlet_id", lambda values: json_dumps(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
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

    summary_groups, group_members, singletons = load_inputs(args)
    summary_groups = normalize_summary(summary_groups, blank_decision=blank_decision)
    group_members = normalize_members(group_members)
    singletons = normalize_singletons(singletons)
    validate_inputs(summary_groups, group_members, singletons)
    mapping, audit, group_roster = build_mapping(summary_groups, group_members, singletons)

    mapping_csv = args.output_root / "microtopic_to_merged_group.csv"
    roster_csv = args.output_root / "merged_group_roster.csv"
    audit_csv = args.output_root / "hierarchical_review_decision_audit.csv"

    mapping.to_csv(mapping_csv, index=False)
    group_roster.to_csv(roster_csv, index=False)
    audit.to_csv(audit_csv, index=False)

    write_json(
        args.output_root / "hierarchical_review_materialization_manifest.json",
        {
            "output_root": str(args.output_root),
            "blank_group_decision": blank_decision,
            "raw_topic_count": int(mapping.shape[0]),
            "final_group_count": int(group_roster.shape[0]),
            "singleton_group_count": int(group_roster["is_singleton_group"].sum()) if not group_roster.empty else 0,
            "accepted_branchlet_count": int(
                summary_groups.loc[summary_groups["group_decision"] == "accept", "proposed_branchlet_id"].nunique()
            )
            if not summary_groups.empty
            else 0,
            "rejected_branchlet_count": int(
                summary_groups.loc[summary_groups["group_decision"] == "reject", "proposed_branchlet_id"].nunique()
            )
            if not summary_groups.empty
            else 0,
            "review_branchlet_count": int(
                summary_groups.loc[summary_groups["group_decision"] == "review", "proposed_branchlet_id"].nunique()
            )
            if not summary_groups.empty
            else 0,
            "outputs": {
                "microtopic_to_merged_group": str(mapping_csv),
                "merged_group_roster": str(roster_csv),
                "hierarchical_review_decision_audit": str(audit_csv),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
