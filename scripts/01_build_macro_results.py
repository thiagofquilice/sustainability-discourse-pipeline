#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    MACRO_TOPIC_ORDER,
    STATUS_ORDER,
    TOPIC_KEY,
    complete_status_grid,
    ensure_dir,
    expand_aggregate_pairs,
    included_corporate_by_macro,
    infer_temporal_pattern,
    load_review_frame,
    macro_name,
    read_csv,
    unique_noncorporate_topics,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing macro result CSVs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_sample_construction(topics: pd.DataFrame, corporate: pd.DataFrame) -> pd.DataFrame:
    status_counts = topics["paper_status"].value_counts().to_dict()
    reason_counts = topics["inclusion_reason"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "sample_step": "Reviewed non-corporate topics in master review",
                "topic_count": len(topics),
                "note": "Topics carried forward into the final corporate-focus review workbook.",
            },
            {
                "step_order": 2,
                "sample_step": "Direct pairs above corporate threshold before comment round",
                "topic_count": reason_counts.get("direct_pair_with_corporate_above_threshold", 0),
                "note": "Automatically retained as corporate-linked comparisons before final comments.",
            },
            {
                "step_order": 3,
                "sample_step": "Manual-override inclusions before comment round",
                "topic_count": reason_counts.get("manual_override_from_excluded_review", 0),
                "note": "Recovered for substantive review despite weak or absent direct dyads.",
            },
            {
                "step_order": 4,
                "sample_step": "Final aligned external topics kept for paired discussion",
                "topic_count": status_counts.get("aligned", 0),
                "note": "Retained as the corporate-linked external set after final review.",
            },
            {
                "step_order": 5,
                "sample_step": "Final external topics retained without pair",
                "topic_count": status_counts.get("external_relevant_unpaired", 0),
                "note": "Kept as substantively relevant external signals without a direct corporate counterpart.",
            },
            {
                "step_order": 6,
                "sample_step": "Final excluded topics",
                "topic_count": status_counts.get("excluded", 0),
                "note": "Removed from substantive interpretation.",
            },
            {
                "step_order": 7,
                "sample_step": "Corporate groups available in the retained corporate set",
                "topic_count": len(corporate),
                "note": "Used as the corporate anchor inventory across the six macro topics.",
            },
        ]
    )


def build_coverage(topics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        macro_subset = topics[topics["macro_topic"] == macro_topic]
        for status in STATUS_ORDER:
            subset = macro_subset[macro_subset["paper_status"] == status]
            rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": macro_name(macro_topic),
                    "paper_status": status,
                    "topic_count": subset["topic_id"].nunique(),
                    "academic_topic_count": subset.loc[
                        subset["source_noncorporate"] == "academic", "topic_id"
                    ].nunique(),
                    "media_topic_count": subset.loc[
                        subset["source_noncorporate"] == "media", "topic_id"
                    ].nunique(),
                }
            )
    return complete_status_grid(
        pd.DataFrame(rows),
        ["topic_count", "academic_topic_count", "media_topic_count"],
    )


def build_matched_temporal_pairs(review: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    aggregate_pairs = read_csv(input_dir, "aggregate_pair_precedence_classification.csv")
    expanded = expand_aggregate_pairs(aggregate_pairs)
    aligned = (
        review.loc[
            review["paper_status"] == "aligned",
            [
                "macro_topic",
                "final_merge_group_id_noncorporate",
                "final_merge_group_id_corporate",
            ],
        ]
        .drop_duplicates()
        .rename(
            columns={
                "final_merge_group_id_noncorporate": "noncorp_group_id",
                "final_merge_group_id_corporate": "matched_corporate_group_id",
            }
        )
    )
    aligned["noncorp_group_id"] = aligned["noncorp_group_id"].astype(str)
    aligned["matched_corporate_group_id"] = aligned["matched_corporate_group_id"].astype(str)

    matched = expanded.merge(
        aligned,
        on=["macro_topic", "noncorp_group_id", "matched_corporate_group_id"],
        how="inner",
    )
    return matched.drop_duplicates(
        ["aggregate_pair_id", "macro_topic", "dyad", "precedence_type_label"]
    )


def build_temporal_summary(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    figure = (
        matched.groupby(["macro_topic", "macro_topic_name", "dyad", "precedence_type_label"])
        .size()
        .reset_index(name="aggregate_pair_count")
    )

    rows: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = matched[matched["macro_topic"] == macro_topic]
        counts = subset.groupby("precedence_type_label")["aggregate_pair_id"].nunique().to_dict()
        stable_counts = (
            subset[subset["is_stable_leading_pair"]]
            .groupby("precedence_type_label")["aggregate_pair_id"]
            .nunique()
            .to_dict()
        )
        row = {
            "macro_topic": macro_topic,
            "macro_topic_name": macro_name(macro_topic),
            "aligned_aggregate_pair_count": subset["aggregate_pair_id"].nunique(),
            "stable_leading_pair_count": sum(stable_counts.values()),
            "dominant_temporal_pattern": infer_temporal_pattern(counts, stable_counts),
        }
        for label in [
            "academic_leads_corporate",
            "media_leads_corporate",
            "corporate_leads_academic",
            "corporate_leads_media",
            "synchronous_or_unclear",
        ]:
            row[f"{label}_count"] = counts.get(label, 0)
        rows.append(row)

    return figure.sort_values(["macro_topic", "dyad", "precedence_type_label"]), pd.DataFrame(rows)


def coverage_label(ratio: float, unpaired: int) -> str:
    if ratio >= 0.99 and unpaired >= 5:
        return "broad anchor set with substantial external spillovers"
    if ratio >= 0.99:
        return "broad coverage with residual external gaps"
    if ratio >= 0.70:
        return "partial but substantive coverage"
    if ratio >= 0.50:
        return "partial coverage with notable external spillovers"
    return "selective coverage concentrated in a few corporate anchors"


def dominant_alignment_type(aligned_academic: int, aligned_media: int) -> str:
    if aligned_academic and aligned_media:
        if aligned_academic >= aligned_media * 2:
            return "academic-dominated aligned set with a limited media layer"
        return "mixed academic-media aligned set"
    if aligned_academic:
        return "academic-only aligned set"
    if aligned_media:
        return "media-only aligned set"
    return "no aligned external set"


def strategic_implication(ratio: float, unpaired: int) -> str:
    if unpaired >= 5 and ratio < 0.99:
        return "External discourse exceeds what corporate discourse absorbs, indicating a visible blind-spot problem."
    if unpaired:
        return "Corporate discourse covers the core of this domain while leaving some external strands outside its frame."
    if ratio >= 0.70:
        return "Corporate disclosure incorporates most retained external signals in this domain."
    return "Corporate disclosure remains concentrated in a limited subset of possible anchors."


def build_macro_summary(
    review: pd.DataFrame,
    topics: pd.DataFrame,
    corporate: pd.DataFrame,
    temporal_macro: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        macro_topics = topics[topics["macro_topic"] == macro_topic]
        aligned = macro_topics[macro_topics["paper_status"] == "aligned"]
        unpaired = macro_topics[macro_topics["paper_status"] == "external_relevant_unpaired"]
        excluded = macro_topics[macro_topics["paper_status"] == "excluded"]
        aligned_review = review[(review["macro_topic"] == macro_topic) & (review["paper_status"] == "aligned")]
        corporate_total = corporate[corporate["macro_topic"] == macro_topic]
        matched_corporate = aligned_review.drop_duplicates(
            ["subgroup_corporate", "final_merge_group_id_corporate"]
        )
        total_count = len(corporate_total)
        matched_count = len(matched_corporate)
        ratio = matched_count / total_count if total_count else 0
        temporal = temporal_macro[temporal_macro["macro_topic"] == macro_topic]
        temporal_row = temporal.iloc[0].to_dict() if not temporal.empty else {}
        aligned_academic = aligned.loc[aligned["source_noncorporate"] == "academic", "topic_id"].nunique()
        aligned_media = aligned.loc[aligned["source_noncorporate"] == "media", "topic_id"].nunique()
        rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_name(macro_topic),
                "aligned_external_topic_count": aligned["topic_id"].nunique(),
                "external_relevant_unpaired_count": unpaired["topic_id"].nunique(),
                "excluded_topic_count": excluded["topic_id"].nunique(),
                "aligned_academic_count": aligned_academic,
                "aligned_media_count": aligned_media,
                "corporate_groups_matched_aligned": matched_count,
                "corporate_groups_total": total_count,
                "corporate_coverage_ratio": round(ratio, 4),
                "corporate_coverage_label": coverage_label(ratio, unpaired["topic_id"].nunique()),
                "dominant_alignment_type": dominant_alignment_type(aligned_academic, aligned_media),
                "dominant_temporal_pattern": temporal_row.get(
                    "dominant_temporal_pattern",
                    "No robust temporal signal remains after final review.",
                ),
                "aligned_aggregate_pair_count": int(temporal_row.get("aligned_aggregate_pair_count", 0)),
                "stable_leading_pair_count": int(temporal_row.get("stable_leading_pair_count", 0)),
                "strategic_implication": strategic_implication(ratio, unpaired["topic_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    review = load_review_frame(args.input_dir)
    topics = unique_noncorporate_topics(review)
    corporate = included_corporate_by_macro(args.input_dir)
    matched = build_matched_temporal_pairs(review, args.input_dir)
    temporal_figure, temporal_macro = build_temporal_summary(matched)

    outputs = [
        write_csv(build_sample_construction(topics, corporate), args.output_dir, "table_01_sample_construction.csv"),
        write_csv(build_coverage(topics), args.output_dir, "figure_01_macro_topic_coverage.csv"),
        write_csv(temporal_figure, args.output_dir, "figure_02_macro_topic_temporal_summary.csv"),
        write_csv(
            build_macro_summary(review, topics, corporate, temporal_macro),
            args.output_dir,
            "table_02_macro_topic_summary.csv",
        ),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
