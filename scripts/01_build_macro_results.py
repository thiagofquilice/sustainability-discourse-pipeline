#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    MACRO_TOPIC_ORDER,
    STATUS_ORDER,
    complete_status_grid,
    ensure_dir,
    included_corporate_by_macro,
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


def build_macro_document_counts(input_dir: Path) -> pd.DataFrame:
    counts = read_csv(input_dir, "macro_topic_document_counts_final_by_source.csv")
    counts["macro_topic"] = pd.Categorical(counts["macro_topic"], MACRO_TOPIC_ORDER, ordered=True)
    return counts.sort_values("macro_topic").reset_index(drop=True)


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


def load_timing_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = read_csv(input_dir, "aggregate_spearman_peak_summary_table.csv")
    individual = read_csv(input_dir, "individual_spearman_peak_summary_table.csv")
    aggregate["relation_level"] = "aggregate"
    individual["relation_level"] = "individual"
    return aggregate, individual


def summarize_timing(frame: pd.DataFrame, relation_level: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = frame[frame["macro_topic"] == macro_topic].copy()
        spearman = pd.to_numeric(subset.get("spearman_r"), errors="coerce")
        spearman_p = pd.to_numeric(subset.get("spearman_p"), errors="coerce")
        first_gap = pd.to_numeric(subset.get("first_active_year_gap"), errors="coerce")
        peak_gap = pd.to_numeric(subset.get("peak_year_gap"), errors="coerce")
        rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_name(
                    macro_topic,
                    subset["macro_topic_name"].iloc[0] if not subset.empty and "macro_topic_name" in subset else "",
                ),
                "relation_level": relation_level,
                "relation_count": int(len(subset)),
                "valid_spearman_count": int(spearman.notna().sum()),
                "spearman_positive_count": int((spearman > 0).sum()),
                "spearman_negative_count": int((spearman < 0).sum()),
                "spearman_p_lt_10_count": int((spearman_p < 0.10).sum()),
                "spearman_p_lt_05_count": int((spearman_p < 0.05).sum()),
                "external_first_count": int((first_gap > 0).sum()),
                "corporate_first_count": int((first_gap < 0).sum()),
                "same_first_year_count": int((first_gap == 0).sum()),
                "external_peak_first_count": int((peak_gap > 0).sum()),
                "corporate_peak_first_count": int((peak_gap < 0).sum()),
                "same_peak_year_count": int((peak_gap == 0).sum()),
                "median_first_active_year_gap": round(float(first_gap.median()), 3) if first_gap.notna().any() else "",
                "median_peak_year_gap": round(float(peak_gap.median()), 3) if peak_gap.notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def build_timing_summary(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate, individual = load_timing_tables(input_dir)
    summary = pd.concat(
        [
            summarize_timing(aggregate, "aggregate"),
            summarize_timing(individual, "individual"),
        ],
        ignore_index=True,
    )
    aggregate_macro = summary[summary["relation_level"] == "aggregate"].copy()
    return summary, aggregate_macro


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
    aggregate_timing: pd.DataFrame,
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
        timing = aggregate_timing[aggregate_timing["macro_topic"] == macro_topic]
        timing_row = timing.iloc[0].to_dict() if not timing.empty else {}
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
                "aggregate_relation_count": int(timing_row.get("relation_count", 0) or 0),
                "aggregate_valid_spearman_count": int(timing_row.get("valid_spearman_count", 0) or 0),
                "aggregate_spearman_p_lt_10_count": int(timing_row.get("spearman_p_lt_10_count", 0) or 0),
                "aggregate_median_first_active_year_gap": timing_row.get("median_first_active_year_gap", ""),
                "aggregate_median_peak_year_gap": timing_row.get("median_peak_year_gap", ""),
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
    timing_summary, aggregate_timing = build_timing_summary(args.input_dir)

    outputs = [
        write_csv(build_sample_construction(topics, corporate), args.output_dir, "table_01_sample_construction.csv"),
        write_csv(build_macro_document_counts(args.input_dir), args.output_dir, "figure_00_macro_document_counts.csv"),
        write_csv(build_coverage(topics), args.output_dir, "figure_01_macro_topic_coverage.csv"),
        write_csv(timing_summary, args.output_dir, "figure_02_source_timing_diagnostics.csv"),
        write_csv(
            build_macro_summary(review, topics, corporate, aggregate_timing),
            args.output_dir,
            "table_02_macro_topic_summary.csv",
        ),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
