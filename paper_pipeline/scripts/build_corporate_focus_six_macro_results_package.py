#!/usr/bin/env python3
"""Build a six-macro-topic paper results package for the reviewed corporate-focus set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_REVIEW_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_review_with_overrides"
DEFAULT_TEMPORAL_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_temporal_aggregated_no_mnn_lag5"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_six_macro_results"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
MACRO_TOPIC_NAME_MAP = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution and environmental stewardship",
}
STATUS_ORDER = ["aligned", "external_relevant_unpaired", "excluded"]
PRECEDENCE_ORDER = [
    "academic_leads_corporate",
    "media_leads_corporate",
    "corporate_leads_academic",
    "corporate_leads_media",
    "synchronous_or_unclear",
]
SOURCE_ORDER = {"academic": 0, "media": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--temporal-root", type=Path, default=DEFAULT_TEMPORAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-aligned", type=int, default=3)
    parser.add_argument("--top-unpaired", type=int, default=5)
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_review_decision(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text == "delete":
        return "delete"
    if text == "not related":
        return "not related"
    return ""


def decision_to_status(value: str) -> str:
    if value == "delete":
        return "excluded"
    if value == "not related":
        return "external_relevant_unpaired"
    return "aligned"


def fill_text_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].fillna("")
    return result


def load_master_with_status(review_root: Path) -> pd.DataFrame:
    master = read_csv_required(review_root / "corporate_focus_master_review.csv")
    decisions = read_csv_required(review_root / "corporate_focus_commented_decision_rows.csv")
    key = ["macro_topic", "subgroup_noncorporate", "final_merge_group_id_noncorporate"]

    decision_column = (
        "_normalized_review_decision"
        if "_normalized_review_decision" in decisions.columns
        else "review_decision"
    )
    decisions = decisions[key + [decision_column]].copy()
    decisions["_normalized_review_decision"] = decisions[decision_column].map(normalize_review_decision)
    decisions = decisions[key + ["_normalized_review_decision"]].drop_duplicates(subset=key, keep="last")

    merged = master.merge(decisions, on=key, how="left")
    merged["_normalized_review_decision"] = merged["_normalized_review_decision"].fillna("")
    merged["paper_status"] = merged["_normalized_review_decision"].map(decision_to_status)
    merged["paper_status"] = pd.Categorical(merged["paper_status"], categories=STATUS_ORDER, ordered=True)
    merged = fill_text_fields(merged)
    return merged


def expand_aggregate_pairs(aggregate_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in aggregate_pairs.to_dict(orient="records"):
        group_ids = json.loads(record["constituent_noncorporate_group_ids_json"])
        for group_id in group_ids:
            rows.append(
                {
                    "aggregate_pair_id": record["aggregate_pair_id"],
                    "macro_topic": record["macro_topic"],
                    "macro_topic_name": record["macro_topic_name"],
                    "dyad": record["dyad"],
                    "matched_corporate_group_id": record["matched_corporate_group_id"],
                    "noncorp_group_id": group_id,
                    "precedence_type_label": record["precedence_type_label"],
                    "is_stable_leading_pair": bool(record["is_stable_leading_pair"]),
                }
            )
    return pd.DataFrame(rows)


def build_temporal_outputs(master: pd.DataFrame, temporal_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_pairs = read_csv_required(temporal_root / "aggregate_pair_precedence_classification.csv")
    expanded = expand_aggregate_pairs(aggregate_pairs)

    aligned_status = (
        master.loc[master["paper_status"] == "aligned", [
            "macro_topic",
            "final_merge_group_id_noncorporate",
            "final_merge_group_id_corporate",
        ]]
        .drop_duplicates()
        .rename(
            columns={
                "final_merge_group_id_noncorporate": "noncorp_group_id",
                "final_merge_group_id_corporate": "matched_corporate_group_id",
            }
        )
    )

    matched = expanded.merge(
        aligned_status,
        on=["macro_topic", "noncorp_group_id", "matched_corporate_group_id"],
        how="inner",
    )
    matched = matched.drop_duplicates(
        subset=["aggregate_pair_id", "macro_topic", "dyad", "precedence_type_label"]
    )

    figure_2 = (
        matched.groupby(["macro_topic", "macro_topic_name", "dyad", "precedence_type_label"])
        .size()
        .reset_index(name="aggregate_pair_count")
    )
    figure_2["precedence_type_label"] = pd.Categorical(
        figure_2["precedence_type_label"], categories=PRECEDENCE_ORDER, ordered=True
    )
    figure_2 = figure_2.sort_values(["macro_topic", "dyad", "precedence_type_label"]).reset_index(drop=True)

    macro_summary_rows: list[dict[str, Any]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = matched.loc[matched["macro_topic"] == macro_topic].copy()
        macro_name = MACRO_TOPIC_NAME_MAP.get(macro_topic, "")
        if not subset.empty and pd.notna(subset["macro_topic_name"].iloc[0]) and str(subset["macro_topic_name"].iloc[0]).strip():
            macro_name = subset["macro_topic_name"].iloc[0]
        counts = subset.groupby("precedence_type_label")["aggregate_pair_id"].nunique().to_dict()
        stable_counts = (
            subset.loc[subset["is_stable_leading_pair"]]
            .groupby("precedence_type_label")["aggregate_pair_id"]
            .nunique()
            .to_dict()
        )
        total_pairs = subset["aggregate_pair_id"].nunique()
        dominant_pattern = infer_temporal_pattern(counts, stable_counts)
        macro_summary_rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_name,
                "aligned_aggregate_pair_count": total_pairs,
                "academic_leads_corporate_count": counts.get("academic_leads_corporate", 0),
                "media_leads_corporate_count": counts.get("media_leads_corporate", 0),
                "corporate_leads_academic_count": counts.get("corporate_leads_academic", 0),
                "corporate_leads_media_count": counts.get("corporate_leads_media", 0),
                "synchronous_or_unclear_count": counts.get("synchronous_or_unclear", 0),
                "stable_leading_pair_count": sum(stable_counts.values()),
                "dominant_temporal_pattern": dominant_pattern,
            }
        )
    return figure_2, pd.DataFrame(macro_summary_rows)


def infer_temporal_pattern(counts: dict[str, int], stable_counts: dict[str, int]) -> str:
    non_zero = {key: value for key, value in counts.items() if value}
    if not non_zero:
        return "No robust temporal signal remains after post-comment filtering."

    dominant_label, dominant_count = max(
        non_zero.items(),
        key=lambda item: (item[1], -PRECEDENCE_ORDER.index(item[0]) if item[0] in PRECEDENCE_ORDER else 0),
    )
    synchronous_count = counts.get("synchronous_or_unclear", 0)
    stable_total = sum(stable_counts.values())

    if dominant_label == "synchronous_or_unclear":
        leading_non_zero = [
            label for label in PRECEDENCE_ORDER if label != "synchronous_or_unclear" and counts.get(label, 0) > 0
        ]
        if not leading_non_zero:
            return "Mostly synchronous or inconclusive temporal ordering."
        lead_phrase = precedence_label_to_phrase(leading_non_zero[0], compact=True)
        if stable_total:
            return (
                f"Mostly synchronous/co-evolving, with isolated {lead_phrase} episodes "
                f"({stable_total} stable leading case{'s' if stable_total != 1 else ''})."
            )
        return f"Mostly synchronous/co-evolving, with isolated {lead_phrase} episodes."

    dominant_phrase = precedence_label_to_phrase(dominant_label, compact=False)
    if synchronous_count:
        return f"{dominant_phrase}, but with residual synchronous or unclear cases."
    return f"{dominant_phrase}."


def precedence_label_to_phrase(label: str, *, compact: bool) -> str:
    if label == "academic_leads_corporate":
        return "academic-leading" if compact else "Academic discourse appears to lead corporate discourse"
    if label == "media_leads_corporate":
        return "media-leading" if compact else "Media discourse appears to lead corporate discourse"
    if label == "corporate_leads_academic":
        return "corporate-leading-academic" if compact else "Corporate discourse appears to lead academic discourse"
    if label == "corporate_leads_media":
        return "corporate-leading-media" if compact else "Corporate discourse appears to lead media discourse"
    return "synchronous-or-unclear" if compact else "Temporal ordering is mostly synchronous or unclear"


def build_sample_construction(master: pd.DataFrame, review_root: Path) -> pd.DataFrame:
    included_corporate = read_csv_required(review_root / "included_corporate_groups.csv")
    rows = [
        {
            "step_order": 1,
            "sample_step": "Reviewed non-corporate topics in master review",
            "topic_count": int(len(master)),
            "note": "Topics carried forward into the hybrid corporate-focus review workbook.",
        },
        {
            "step_order": 2,
            "sample_step": "Direct pairs above corporate threshold before comment round",
            "topic_count": int((master["inclusion_reason"] == "direct_pair_with_corporate_above_threshold").sum()),
            "note": "Automatically retained as corporate-linked comparisons before manual comments.",
        },
        {
            "step_order": 3,
            "sample_step": "Manual-override inclusions before comment round",
            "topic_count": int((master["inclusion_reason"] == "manual_override_from_excluded_review").sum()),
            "note": "Recovered for substantive review despite weak or absent direct dyads.",
        },
        {
            "step_order": 4,
            "sample_step": "Final aligned external topics kept for paired discussion",
            "topic_count": int((master["paper_status"] == "aligned").sum()),
            "note": "Retained as the corporate-linked set in the paper after comments.",
        },
        {
            "step_order": 5,
            "sample_step": "Final external topics retained without pair",
            "topic_count": int((master["paper_status"] == "external_relevant_unpaired").sum()),
            "note": "Marked 'not related' but kept as substantively relevant external signals.",
        },
        {
            "step_order": 6,
            "sample_step": "Final excluded topics",
            "topic_count": int((master["paper_status"] == "excluded").sum()),
            "note": "Marked 'delete' and removed from substantive interpretation.",
        },
        {
            "step_order": 7,
            "sample_step": "Corporate groups available in the retained corporate set",
            "topic_count": int(included_corporate["final_merge_group_id"].nunique()),
            "note": "Used as the corporate anchor inventory across the six macro topics.",
        },
    ]
    return pd.DataFrame(rows)


def build_coverage_figure_data(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = master.loc[master["macro_topic"] == macro_topic].copy()
        macro_name = MACRO_TOPIC_NAME_MAP.get(macro_topic, "")
        if not subset.empty and pd.notna(subset["macro_topic_name"].iloc[0]) and str(subset["macro_topic_name"].iloc[0]).strip():
            macro_name = subset["macro_topic_name"].iloc[0]
        for status in STATUS_ORDER:
            status_subset = subset.loc[subset["paper_status"] == status]
            rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": macro_name,
                    "paper_status": status,
                    "topic_count": int(len(status_subset)),
                    "academic_topic_count": int((status_subset["source_noncorporate"] == "academic").sum()),
                    "media_topic_count": int((status_subset["source_noncorporate"] == "media").sum()),
                }
            )
    return pd.DataFrame(rows)


def infer_corporate_coverage_label(
    coverage_ratio: float,
    aligned_count: int,
    unpaired_count: int,
) -> str:
    if coverage_ratio >= 0.8 and unpaired_count == 0:
        return "broad corporate coverage"
    if coverage_ratio >= 0.8 and unpaired_count >= max(3, aligned_count // 2):
        return "broad anchor set with substantial external spillovers"
    if coverage_ratio >= 0.8:
        return "broad coverage with residual external gaps"
    if coverage_ratio >= 0.5 and unpaired_count >= max(3, aligned_count // 2):
        return "partial coverage with notable external spillovers"
    if coverage_ratio >= 0.5:
        return "partial but substantive coverage"
    return "selective coverage concentrated in a few corporate anchors"


def infer_alignment_type(aligned_subset: pd.DataFrame) -> str:
    academic_count = int((aligned_subset["source_noncorporate"] == "academic").sum())
    media_count = int((aligned_subset["source_noncorporate"] == "media").sum())
    total = int(len(aligned_subset))
    if total == 0:
        return "no aligned external set"
    academic_share = academic_count / total
    media_share = media_count / total
    if media_count == 0:
        return "academic-only aligned set"
    if academic_share >= 0.8:
        return "academic-dominated aligned set with a limited media layer"
    if media_share >= 0.4:
        return "mixed academic-media aligned set"
    return "academic-dominated aligned set with visible media support"


def infer_strategic_implication(
    coverage_label: str,
    aligned_count: int,
    unpaired_count: int,
    temporal_pattern: str,
) -> str:
    if unpaired_count == 0:
        return (
            "Corporate discourse absorbs most external signals in this domain, but it does so through "
            f"{coverage_label}; temporally, the retained aligned set looks {temporal_pattern.lower()}"
        )
    if unpaired_count >= aligned_count:
        return (
            "External discourse in this domain exceeds what corporate discourse absorbs, suggesting a pronounced "
            "blind-spot problem and a likely need for broader external sensing."
        )
    return (
        "Corporate discourse covers the core of this domain but leaves strategically relevant external strands outside "
        "its frame, pointing to selective attention rather than full issue absorption."
    )


def build_macro_summary(
    master: pd.DataFrame,
    review_root: Path,
    temporal_summary: pd.DataFrame,
) -> pd.DataFrame:
    included_corporate = read_csv_required(review_root / "included_corporate_groups.csv")
    corporate_counts = (
        included_corporate.groupby("assigned_label")["final_merge_group_id"].nunique().to_dict()
    )
    temporal_lookup = temporal_summary.set_index("macro_topic").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = master.loc[master["macro_topic"] == macro_topic].copy()
        aligned_subset = subset.loc[subset["paper_status"] == "aligned"]
        unpaired_subset = subset.loc[subset["paper_status"] == "external_relevant_unpaired"]
        excluded_subset = subset.loc[subset["paper_status"] == "excluded"]
        macro_name = MACRO_TOPIC_NAME_MAP.get(macro_topic, "")
        if not subset.empty and pd.notna(subset["macro_topic_name"].iloc[0]) and str(subset["macro_topic_name"].iloc[0]).strip():
            macro_name = subset["macro_topic_name"].iloc[0]
        corporate_total = int(corporate_counts.get(macro_topic, 0))
        corporate_matched = int(aligned_subset["final_merge_group_id_corporate"].nunique())
        coverage_ratio = (corporate_matched / corporate_total) if corporate_total else 0.0
        temporal = temporal_lookup.get(macro_topic, {})
        coverage_label = infer_corporate_coverage_label(
            coverage_ratio=coverage_ratio,
            aligned_count=int(len(aligned_subset)),
            unpaired_count=int(len(unpaired_subset)),
        )
        alignment_type = infer_alignment_type(aligned_subset)
        implication = infer_strategic_implication(
            coverage_label=coverage_label,
            aligned_count=int(len(aligned_subset)),
            unpaired_count=int(len(unpaired_subset)),
            temporal_pattern=str(temporal.get("dominant_temporal_pattern") or "unclear"),
        )
        rows.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": macro_name,
                "aligned_external_topic_count": int(len(aligned_subset)),
                "external_relevant_unpaired_count": int(len(unpaired_subset)),
                "excluded_topic_count": int(len(excluded_subset)),
                "aligned_academic_count": int((aligned_subset["source_noncorporate"] == "academic").sum()),
                "aligned_media_count": int((aligned_subset["source_noncorporate"] == "media").sum()),
                "corporate_groups_matched_aligned": corporate_matched,
                "corporate_groups_total": corporate_total,
                "corporate_coverage_ratio": round(coverage_ratio, 3),
                "corporate_coverage_label": coverage_label,
                "dominant_alignment_type": alignment_type,
                "dominant_temporal_pattern": str(temporal.get("dominant_temporal_pattern") or ""),
                "aligned_aggregate_pair_count": int(temporal.get("aligned_aggregate_pair_count") or 0),
                "stable_leading_pair_count": int(temporal.get("stable_leading_pair_count") or 0),
                "strategic_implication": implication,
            }
        )
    return pd.DataFrame(rows)


def choose_top_aligned(aligned_subset: pd.DataFrame, limit: int) -> pd.DataFrame:
    if aligned_subset.empty:
        return aligned_subset.copy()
    working = aligned_subset.copy()
    working["_source_order"] = working["source_noncorporate"].map(SOURCE_ORDER).fillna(99)
    working = working.sort_values(
        ["best_cosine_similarity", "direct_pair_count", "_source_order", "topic_label_refined_noncorporate"],
        ascending=[False, False, True, True],
    )
    selected_indices: list[int] = []
    seen_corporate: set[str] = set()
    for index, row in working.iterrows():
        corporate_group = str(row["final_merge_group_id_corporate"])
        if corporate_group in seen_corporate:
            continue
        selected_indices.append(index)
        seen_corporate.add(corporate_group)
        if len(selected_indices) == limit:
            break
    if len(selected_indices) < min(limit, len(working)):
        for index in working.index:
            if index in selected_indices:
                continue
            selected_indices.append(index)
            if len(selected_indices) == limit:
                break
    return working.loc[selected_indices].drop(columns="_source_order")


def choose_top_unpaired(unpaired_subset: pd.DataFrame, limit: int) -> pd.DataFrame:
    if unpaired_subset.empty:
        return unpaired_subset.copy()
    working = unpaired_subset.copy()
    working["_source_order"] = working["source_noncorporate"].map(SOURCE_ORDER).fillna(99)
    working = working.sort_values(
        ["best_cosine_similarity", "_source_order", "topic_label_refined_noncorporate"],
        ascending=[False, True, True],
    )
    return working.head(limit).drop(columns="_source_order")


def snippet(text: object, limit: int = 220) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def build_overview_paragraph(summary_row: pd.Series) -> str:
    aligned = int(summary_row["aligned_external_topic_count"])
    unpaired = int(summary_row["external_relevant_unpaired_count"])
    excluded = int(summary_row["excluded_topic_count"])
    academic = int(summary_row["aligned_academic_count"])
    media = int(summary_row["aligned_media_count"])
    matched = int(summary_row["corporate_groups_matched_aligned"])
    total = int(summary_row["corporate_groups_total"])
    return (
        f"{summary_row['macro_topic']} contains {aligned} retained aligned external topics "
        f"({academic} academic; {media} media), {unpaired} retained external topics without pair, and {excluded} "
        f"excluded topics. In the retained aligned set, corporate discourse connects to {matched} of {total} "
        f"available corporate groups, which reads as {summary_row['corporate_coverage_label']}."
    )


def build_temporal_paragraph(summary_row: pd.Series) -> str:
    pattern = str(summary_row["dominant_temporal_pattern"]).strip()
    pair_count = int(summary_row["aligned_aggregate_pair_count"])
    if not pair_count:
        return "No robust macro-level temporal summary remains after filtering to the retained aligned set."
    return f"Across {pair_count} retained aggregate pair(s), the most plausible temporal reading is: {pattern}"


def build_interpretive_close(summary_row: pd.Series) -> str:
    return (
        f"Interpretive close: {summary_row['strategic_implication']} "
        f"This makes {summary_row['macro_topic']} a useful section for discussing selective attention, "
        f"blind spots, and the limits of corporate issue absorption."
    )


def write_section_packet(
    packet_path: Path,
    summary_row: pd.Series,
    aligned_examples: pd.DataFrame,
    unpaired_examples: pd.DataFrame,
) -> None:
    lines = [
        f"# {summary_row['macro_topic']} — {summary_row['macro_topic_name']}",
        "",
        "## Overview",
        build_overview_paragraph(summary_row),
        "",
        "## Aligned Pairs To Foreground",
    ]
    if aligned_examples.empty:
        lines.extend(["No retained aligned pairs remain for this macro topic.", ""])
    else:
        lines.extend(
            [
                "| Source | Non-corporate topic | Corporate counterpart | Cosine | Temporal cue | Inclusion reason |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for _, row in aligned_examples.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["source_noncorporate"]),
                        str(row["topic_label_refined_noncorporate"]),
                        str(row["topic_label_refined_corporate"]),
                        f"{float(row['best_cosine_similarity']):.3f}" if str(row["best_cosine_similarity"]) else "",
                        str(row["precedence_type_label"] or ""),
                        str(row["inclusion_reason"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("Suggested reading note:")
        for _, row in aligned_examples.iterrows():
            lines.append(
                f"- {row['topic_label_refined_noncorporate']} ↔ {row['topic_label_refined_corporate']}: "
                f"{snippet(row['overall_summary_noncorporate'])}"
            )
        lines.append("")

    lines.extend(["## External Topics To Keep Without Pair"])
    if unpaired_examples.empty:
        lines.extend(["No `not related` topics remain in this macro topic.", ""])
    else:
        for _, row in unpaired_examples.iterrows():
            lines.append(
                f"- {row['topic_label_refined_noncorporate']} ({row['source_noncorporate']}): "
                f"{snippet(row['overall_summary_noncorporate'])}"
            )
        lines.append("")

    lines.extend(
        [
            "## Temporal Anchor",
            build_temporal_paragraph(summary_row),
            "",
            "## Interpretive Close",
            build_interpretive_close(summary_row),
            "",
        ]
    )
    packet_path.write_text("\n".join(lines), encoding="utf-8")


def build_results_outline(sample_table: pd.DataFrame, macro_summary: pd.DataFrame) -> str:
    aligned_total = int(macro_summary["aligned_external_topic_count"].sum())
    unpaired_total = int(macro_summary["external_relevant_unpaired_count"].sum())
    excluded_total = int(macro_summary["excluded_topic_count"].sum())
    reviewed_total = aligned_total + unpaired_total + excluded_total
    lines = [
        "# Results Outline — Six Macro Topics As The Narrative Spine",
        "",
        "## Short Opening To Results",
        (
            f"The reviewed corporate-focus set contains {reviewed_total} non-corporate topics in `master_review`. "
            f"After the manual comment round, {aligned_total} remain as aligned external topics, {unpaired_total} "
            f"remain in the paper as substantively relevant but unpaired external topics, and {excluded_total} are "
            f"removed from substantive interpretation."
        ),
        "",
        "Use `table_01_sample_construction.csv` for the opening sample-construction paragraph and "
        "`figure_01_macro_topic_coverage.csv` for the coverage/blind-spot overview figure.",
        "",
        "## Section Sequence",
    ]

    for _, row in macro_summary.iterrows():
        lines.extend(
            [
                f"### {row['macro_topic']} — {row['macro_topic_name']}",
                build_overview_paragraph(row),
                f"- Best structure: overview -> aligned pairs -> unpaired external topics -> temporal anchor -> interpretive close.",
                f"- Temporal anchor: {row['dominant_temporal_pattern']}",
                f"- Strategic payoff: {row['strategic_implication']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Closing Cross-Sectional Synthesis",
            "- Attention selectivity: show where corporate discourse concentrates external signals into a narrow anchor set.",
            "- External sensing: use the retained unpaired topics as evidence that strategy-relevant signals can remain outside the corporate frame.",
            "- Late or partial incorporation: use the macro-level temporal summary to show where corporate discourse co-evolves, lags, or selectively leads.",
            "",
        ]
    )
    return "\n".join(lines)


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_directory(args.output_dir)
    section_dir = ensure_directory(output_dir / "section_packets")
    supplement_dir = ensure_directory(output_dir / "supplement")

    master = load_master_with_status(args.review_root)
    sample_construction = build_sample_construction(master, args.review_root)
    coverage_figure = build_coverage_figure_data(master)
    temporal_figure, temporal_summary = build_temporal_outputs(master, args.temporal_root)
    macro_summary = build_macro_summary(master, args.review_root, temporal_summary)

    sample_construction.to_csv(output_dir / "table_01_sample_construction.csv", index=False)
    coverage_figure.to_csv(output_dir / "figure_01_macro_topic_coverage.csv", index=False)
    temporal_figure.to_csv(output_dir / "figure_02_macro_topic_temporal_summary.csv", index=False)
    macro_summary.to_csv(output_dir / "table_02_macro_topic_summary.csv", index=False)

    aligned_full = master.loc[master["paper_status"] == "aligned"].copy()
    unpaired_full = master.loc[master["paper_status"] == "external_relevant_unpaired"].copy()
    excluded_full = master.loc[master["paper_status"] == "excluded"].copy()

    aligned_full.to_csv(supplement_dir / "supplement_retained_aligned_topics.csv", index=False)
    unpaired_full.to_csv(supplement_dir / "supplement_not_related_topics.csv", index=False)
    excluded_full.to_csv(supplement_dir / "supplement_excluded_topics.csv", index=False)

    for macro_topic in MACRO_TOPIC_ORDER:
        summary_row = macro_summary.loc[macro_summary["macro_topic"] == macro_topic].iloc[0]
        aligned_subset = aligned_full.loc[aligned_full["macro_topic"] == macro_topic].copy()
        unpaired_subset = unpaired_full.loc[unpaired_full["macro_topic"] == macro_topic].copy()
        aligned_examples = choose_top_aligned(aligned_subset, limit=args.top_aligned)
        unpaired_examples = choose_top_unpaired(unpaired_subset, limit=args.top_unpaired)

        aligned_examples.to_csv(section_dir / f"{macro_topic}_aligned_examples.csv", index=False)
        unpaired_examples.to_csv(section_dir / f"{macro_topic}_unpaired_examples.csv", index=False)
        write_section_packet(
            section_dir / f"{macro_topic}_section_packet.md",
            summary_row,
            aligned_examples,
            unpaired_examples,
        )

    results_outline = build_results_outline(sample_construction, macro_summary)
    (output_dir / "results_six_macro_outline.md").write_text(results_outline, encoding="utf-8")

    readme_text = "\n".join(
        [
            "# Corporate Focus Six-Macro-Topic Results Package",
            "",
            "This package implements the post-comment paper structure centered on six macro topics.",
            "",
            "Core files:",
            "- `table_01_sample_construction.csv`",
            "- `figure_01_macro_topic_coverage.csv`",
            "- `figure_02_macro_topic_temporal_summary.csv`",
            "- `table_02_macro_topic_summary.csv`",
            "- `results_six_macro_outline.md`",
            "- `section_packets/T1_section_packet.md` ... `section_packets/T6_section_packet.md`",
            "",
            "Supplementary lists live under `supplement/`.",
        ]
    )
    (output_dir / "README_results_package.md").write_text(readme_text, encoding="utf-8")

    manifest = {
        "review_root": str(args.review_root),
        "temporal_root": str(args.temporal_root),
        "output_dir": str(output_dir),
        "reviewed_noncorporate_topics": int(len(master)),
        "aligned_external_topics": int(len(aligned_full)),
        "external_relevant_unpaired_topics": int(len(unpaired_full)),
        "excluded_topics": int(len(excluded_full)),
        "top_aligned_examples_per_macro": int(args.top_aligned),
        "top_unpaired_examples_per_macro": int(args.top_unpaired),
    }
    (output_dir / "results_package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_package(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
