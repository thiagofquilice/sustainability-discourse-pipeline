#!/usr/bin/env python3
"""Materialize marked corporate/external group exclusions and final filter counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PIPELINE_ROOT = Path("/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline")
DEFAULT_MARKED_WORKBOOK = Path("/home/thiago/corporate_external_group_exclusion_review_marked.xlsx")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PIPELINE_ROOT
    / "outputs"
    / "corporate_external_group_exclusion_review"
)

ALL_GROUPS = (
    Path("outputs")
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_all_groups_optional.csv"
)
PREVIOUS_REVIEW_DECISIONS = (
    Path("outputs")
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_commented_decision_rows.csv"
)
MASTER_REVIEW = (
    Path("outputs")
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_master_review.csv"
)
MERGED_ROOT = Path("outputs") / "bertopic_micro_merged_multiaspect_reviewed"

SOURCE_ORDER = ["academic", "media", "corporate"]
REVIEW_SHEETS = ["Corporate groups", "External groups"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--marked-workbook", type=Path, default=DEFAULT_MARKED_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def make_topic_id(subgroup: object, group_id: object) -> str:
    return f"{clean_text(subgroup)}::{clean_text(group_id)}"


def detect_document_id_column(columns: pd.Index) -> str:
    for name in ["source_doc_id", "document_id", "doc_id"]:
        if name in columns:
            return name
    raise KeyError("Could not find source_doc_id, document_id, or doc_id in document_topics.csv")


def normalize_decision(value: object) -> str:
    text = clean_text(value).lower()
    return "exclude" if text == "exclude" else "keep"


def normalize_previous_review_decision(value: object) -> str:
    text = clean_text(value).lower()
    return "delete" if text == "delete" else ""


def load_marked_decisions(marked_workbook: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sheet_name in REVIEW_SHEETS:
        frame = pd.read_excel(marked_workbook, sheet_name=sheet_name).fillna("")
        frame["marked_sheet"] = sheet_name
        frame["_normalized_review_decision"] = frame["review_decision"].map(normalize_decision)
        rows.append(frame)
    decisions = pd.concat(rows, ignore_index=True)
    decisions = decisions.loc[decisions["_normalized_review_decision"] == "exclude"].copy()
    if decisions.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "macro_topic",
                "source",
                "subgroup",
                "final_merge_group_id",
                "topic_label",
                "exclusion_reason_category",
                "review_notes",
                "marked_sheet",
                "_normalized_review_decision",
            ]
        )
    decisions["topic_id"] = decisions["topic_id"].map(clean_text)
    decisions = decisions.drop_duplicates("topic_id", keep="last")
    columns = [
        "topic_id",
        "macro_topic",
        "source",
        "subgroup",
        "final_merge_group_id",
        "topic_label",
        "exclusion_reason_category",
        "review_notes",
        "marked_sheet",
        "_normalized_review_decision",
    ]
    for column in columns:
        if column not in decisions.columns:
            decisions[column] = ""
    return decisions[columns].sort_values(["source", "macro_topic", "topic_id"], kind="mergesort")


def load_selected_groups(pipeline_root: Path) -> pd.DataFrame:
    groups = pd.read_csv(pipeline_root / ALL_GROUPS).fillna("")
    groups["final_merge_group_id"] = groups["final_merge_group_id"].astype(str)
    groups["topic_id"] = [make_topic_id(row.subgroup, row.final_merge_group_id) for row in groups.itertuples(index=False)]
    groups["topic_label"] = groups["topic_label_refined"].map(clean_text)
    fallback = groups["topic_name_original"].map(clean_text)
    groups.loc[groups["topic_label"] == "", "topic_label"] = fallback
    return groups


def load_cascaded_external_decisions(
    pipeline_root: Path,
    marked_decisions: pd.DataFrame,
    selected_groups: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "topic_id",
        "macro_topic",
        "source",
        "subgroup",
        "final_merge_group_id",
        "topic_label",
        "exclusion_reason_category",
        "review_notes",
        "marked_sheet",
        "_normalized_review_decision",
    ]
    corporate_decisions = marked_decisions.loc[marked_decisions["source"].eq("corporate")].copy()
    if corporate_decisions.empty:
        return pd.DataFrame(columns=columns)

    path = pipeline_root / MASTER_REVIEW
    if not path.exists():
        raise FileNotFoundError(f"Missing corporate-external master review table: {path}")

    review = pd.read_csv(path).fillna("")
    review["corporate_topic_id"] = [
        make_topic_id(row.subgroup_corporate, row.final_merge_group_id_corporate)
        for row in review.itertuples(index=False)
    ]
    review["external_topic_id"] = [
        make_topic_id(row.subgroup_noncorporate, row.final_merge_group_id_noncorporate)
        for row in review.itertuples(index=False)
    ]

    corporate_topic_ids = set(corporate_decisions["topic_id"])
    linked = review.loc[review["corporate_topic_id"].isin(corporate_topic_ids)].copy()
    if linked.empty:
        return pd.DataFrame(columns=columns)

    selected_lookup = selected_groups.set_index("topic_id", drop=False)
    rows: list[dict[str, Any]] = []
    for external_topic_id, group in linked.groupby("external_topic_id", sort=True):
        if external_topic_id in set(marked_decisions["topic_id"]):
            continue
        if external_topic_id not in selected_lookup.index:
            continue
        selected = selected_lookup.loc[external_topic_id]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[0]
        anchors = (
            group[["corporate_topic_id", "topic_label_refined_corporate"]]
            .drop_duplicates()
            .sort_values(["corporate_topic_id", "topic_label_refined_corporate"], kind="mergesort")
        )
        anchor_notes = "; ".join(
            f"{row.corporate_topic_id} ({clean_text(row.topic_label_refined_corporate)})"
            for row in anchors.itertuples(index=False)
        )
        rows.append(
            {
                "topic_id": external_topic_id,
                "macro_topic": selected["macro_topic"],
                "source": selected["source"],
                "subgroup": selected["subgroup"],
                "final_merge_group_id": selected["final_merge_group_id"],
                "topic_label": selected["topic_label"],
                "exclusion_reason_category": "counterpart_of_excluded_corporate_anchor",
                "review_notes": f"Excluded as external counterpart of excluded corporate anchor(s): {anchor_notes}",
                "marked_sheet": "Cascaded external counterparts",
                "_normalized_review_decision": "exclude",
            }
        )

    return pd.DataFrame(rows, columns=columns)


def combine_direct_and_cascaded_decisions(
    direct_decisions: pd.DataFrame,
    cascaded_decisions: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "topic_id",
        "macro_topic",
        "source",
        "subgroup",
        "final_merge_group_id",
        "topic_label",
        "exclusion_reason_category",
        "review_notes",
        "marked_sheet",
        "_normalized_review_decision",
    ]
    combined = pd.concat([direct_decisions, cascaded_decisions], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=columns)
    combined["_decision_priority"] = combined["marked_sheet"].ne("Cascaded external counterparts").astype(int)
    combined = combined.sort_values(
        ["_decision_priority", "source", "macro_topic", "topic_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    combined = combined.drop_duplicates("topic_id", keep="first").drop(columns=["_decision_priority"])
    return combined[columns].sort_values(["source", "macro_topic", "topic_id"], kind="mergesort")


def load_previous_delete_topic_ids(pipeline_root: Path) -> set[str]:
    path = pipeline_root / PREVIOUS_REVIEW_DECISIONS
    if not path.exists():
        return set()
    decisions = pd.read_csv(path).fillna("")
    decision_column = "_normalized_review_decision" if "_normalized_review_decision" in decisions.columns else "review_decision"
    decisions["_previous_review_decision"] = decisions[decision_column].map(normalize_previous_review_decision)
    deleted = decisions.loc[decisions["_previous_review_decision"] == "delete"].copy()
    if deleted.empty:
        return set()
    return {
        make_topic_id(row.subgroup_noncorporate, row.final_merge_group_id_noncorporate)
        for row in deleted.itertuples(index=False)
    }


def load_group_documents(pipeline_root: Path, selected_groups: pd.DataFrame) -> pd.DataFrame:
    merged_root = pipeline_root / MERGED_ROOT
    frames: list[pd.DataFrame] = []
    selected_key = selected_groups[["topic_id", "source", "subgroup", "final_merge_group_id"]].copy()
    selected_key["final_merge_group_id"] = selected_key["final_merge_group_id"].astype(str)

    for subgroup, requested in selected_key.groupby("subgroup", sort=True):
        path = merged_root / str(subgroup) / "document_topics.csv"
        header = pd.read_csv(path, nrows=0)
        doc_col = detect_document_id_column(header.columns)
        doc_topics = pd.read_csv(path, usecols=[doc_col, "source", "final_merge_group_id"])
        doc_topics = doc_topics.rename(columns={doc_col: "stable_doc_id"})
        doc_topics = doc_topics.dropna(subset=["stable_doc_id", "source", "final_merge_group_id"]).copy()
        doc_topics["stable_doc_id"] = doc_topics["stable_doc_id"].astype(str)
        doc_topics["final_merge_group_id"] = doc_topics["final_merge_group_id"].astype(str)
        doc_topics["subgroup"] = str(subgroup)
        merged = doc_topics.merge(
            requested[["topic_id", "subgroup", "final_merge_group_id"]],
            on=["subgroup", "final_merge_group_id"],
            how="inner",
        )
        frames.append(merged[["topic_id", "source", "subgroup", "final_merge_group_id", "stable_doc_id"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def count_unique_documents(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {source: 0 for source in SOURCE_ORDER}
    return (
        frame.groupby("source")["stable_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )


def count_groups(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {source: 0 for source in SOURCE_ORDER}
    return (
        frame.groupby("source")["topic_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )


def source_summary_rows(
    selected_groups: pd.DataFrame,
    group_docs: pd.DataFrame,
    excluded_topic_ids: set[str],
) -> list[dict[str, Any]]:
    retained_groups = selected_groups.loc[~selected_groups["topic_id"].isin(excluded_topic_ids)].copy()
    excluded_groups = selected_groups.loc[selected_groups["topic_id"].isin(excluded_topic_ids)].copy()
    retained_docs = group_docs.loc[~group_docs["topic_id"].isin(excluded_topic_ids)].copy()
    excluded_group_docs = group_docs.loc[group_docs["topic_id"].isin(excluded_topic_ids)].copy()

    removed_doc_rows = []
    for source in SOURCE_ORDER:
        pre_docs = set(group_docs.loc[group_docs["source"] == source, "stable_doc_id"])
        post_docs = set(retained_docs.loc[retained_docs["source"] == source, "stable_doc_id"])
        for doc_id in sorted(pre_docs - post_docs):
            removed_doc_rows.append({"source": source, "stable_doc_id": doc_id})
    removed_docs = pd.DataFrame(removed_doc_rows)

    pre_doc_counts = count_unique_documents(group_docs)
    retained_doc_counts = count_unique_documents(retained_docs)
    excluded_group_doc_counts = count_unique_documents(excluded_group_docs)
    removed_doc_counts = count_unique_documents(removed_docs)
    pre_group_counts = count_groups(selected_groups)
    retained_group_counts = count_groups(retained_groups)
    excluded_group_counts = count_groups(excluded_groups)

    rows = []
    for source in SOURCE_ORDER:
        rows.append(
            {
                "source": source,
                "pre_final_filter_groups": pre_group_counts[source],
                "excluded_groups": excluded_group_counts[source],
                "retained_groups": retained_group_counts[source],
                "pre_final_filter_unique_documents": pre_doc_counts[source],
                "excluded_group_unique_documents": excluded_group_doc_counts[source],
                "removed_from_final_unique_documents": removed_doc_counts[source],
                "retained_unique_documents": retained_doc_counts[source],
            }
        )
    return rows


def macro_source_summary(
    selected_groups: pd.DataFrame,
    group_docs: pd.DataFrame,
    excluded_topic_ids: set[str],
) -> pd.DataFrame:
    retained_groups = selected_groups.loc[~selected_groups["topic_id"].isin(excluded_topic_ids)].copy()
    rows: list[dict[str, Any]] = []
    for (macro_topic, source), subset in selected_groups.groupby(["macro_topic", "source"], sort=True):
        retained_subset = retained_groups.loc[
            (retained_groups["macro_topic"] == macro_topic) & (retained_groups["source"] == source)
        ]
        pre_topic_ids = set(subset["topic_id"])
        retained_topic_ids = set(retained_subset["topic_id"])
        pre_docs = group_docs.loc[
            (group_docs["source"] == source) & group_docs["topic_id"].isin(pre_topic_ids),
            "stable_doc_id",
        ]
        retained_docs = group_docs.loc[
            (group_docs["source"] == source) & group_docs["topic_id"].isin(retained_topic_ids),
            "stable_doc_id",
        ]
        rows.append(
            {
                "macro_topic": macro_topic,
                "source": source,
                "pre_final_filter_groups": int(len(pre_topic_ids)),
                "excluded_groups": int(len(pre_topic_ids - retained_topic_ids)),
                "retained_groups": int(len(retained_topic_ids)),
                "pre_final_filter_unique_documents": int(pre_docs.nunique()),
                "retained_unique_documents": int(retained_docs.nunique()),
                "removed_from_final_unique_documents": int(len(set(pre_docs) - set(retained_docs))),
            }
        )
    return pd.DataFrame(rows)


def group_impact_table(
    selected_groups: pd.DataFrame,
    group_docs: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    doc_counts = group_docs.groupby("topic_id")["stable_doc_id"].nunique().astype(int).rename("unique_document_count")
    impact = selected_groups[
        [
            "topic_id",
            "macro_topic",
            "source",
            "subgroup",
            "final_merge_group_id",
            "topic_label",
            "selection_bucket",
            "best_matched_corporate_group_id",
            "topic_size",
        ]
    ].merge(doc_counts, left_on="topic_id", right_index=True, how="left")
    impact["unique_document_count"] = impact["unique_document_count"].fillna(0).astype(int)
    decision_info = decisions[
        ["topic_id", "exclusion_reason_category", "review_notes", "marked_sheet"]
    ].copy()
    impact = impact.merge(decision_info, on="topic_id", how="left")
    impact["is_excluded_by_final_filter"] = impact["topic_id"].isin(set(decisions["topic_id"]))
    impact[["exclusion_reason_category", "review_notes", "marked_sheet"]] = impact[
        ["exclusion_reason_category", "review_notes", "marked_sheet"]
    ].fillna("")
    return impact.sort_values(["is_excluded_by_final_filter", "source", "macro_topic", "topic_id"], ascending=[False, True, True, True])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = load_selected_groups(args.pipeline_root)
    direct_decisions = load_marked_decisions(args.marked_workbook)
    cascaded_decisions = load_cascaded_external_decisions(args.pipeline_root, direct_decisions, selected_groups)
    decisions = combine_direct_and_cascaded_decisions(direct_decisions, cascaded_decisions)
    missing = sorted(set(decisions["topic_id"]) - set(selected_groups["topic_id"]))
    if missing:
        raise SystemExit(f"Marked topic_id values not found in selected groups: {missing}")

    group_docs = load_group_documents(args.pipeline_root, selected_groups)
    excluded_topic_ids = set(decisions["topic_id"])
    previous_delete_topic_ids = load_previous_delete_topic_ids(args.pipeline_root)
    base_groups = selected_groups.loc[~selected_groups["topic_id"].isin(previous_delete_topic_ids)].copy()
    base_docs = group_docs.loc[~group_docs["topic_id"].isin(previous_delete_topic_ids)].copy()
    source_counts = pd.DataFrame(source_summary_rows(base_groups, base_docs, excluded_topic_ids))
    macro_counts = macro_source_summary(base_groups, base_docs, excluded_topic_ids)
    impact = group_impact_table(selected_groups, group_docs, decisions)

    decisions_path = args.output_dir / "group_exclusion_decisions.csv"
    impact_path = args.output_dir / "group_exclusion_document_counts.csv"
    source_counts_path = args.output_dir / "final_filter_source_counts.csv"
    macro_counts_path = args.output_dir / "final_filter_macro_source_counts.csv"
    manifest_path = args.output_dir / "group_exclusion_materialization_manifest.json"

    decisions.to_csv(decisions_path, index=False)
    impact.to_csv(impact_path, index=False)
    source_counts.to_csv(source_counts_path, index=False)
    macro_counts.to_csv(macro_counts_path, index=False)

    manifest = {
        "marked_workbook": str(args.marked_workbook),
        "pipeline_root": str(args.pipeline_root),
        "outputs": {
            "group_exclusion_decisions": str(decisions_path),
            "group_exclusion_document_counts": str(impact_path),
            "final_filter_source_counts": str(source_counts_path),
            "final_filter_macro_source_counts": str(macro_counts_path),
        },
        "row_counts": {
            "excluded_groups": int(len(decisions)),
            "direct_marked_groups": int(len(direct_decisions)),
            "cascaded_external_groups": int(len(cascaded_decisions)),
            "previous_review_excluded_groups": int(len(previous_delete_topic_ids)),
            "selected_groups_pre_filter": int(len(base_groups)),
            "selected_groups_retained": int(len(base_groups) - len(set(base_groups["topic_id"]) & excluded_topic_ids)),
            "group_document_rows": int(len(group_docs)),
        },
        "source_counts": source_counts.to_dict(orient="records"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"outputs": manifest["outputs"], "row_counts": manifest["row_counts"]}, indent=2))
    print(source_counts.to_string(index=False))


if __name__ == "__main__":
    main()
