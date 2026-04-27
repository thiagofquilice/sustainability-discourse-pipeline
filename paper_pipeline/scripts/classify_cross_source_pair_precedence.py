#!/usr/bin/env python3
"""Classify cross-source matched pairs by precedence type and prepare join-ready outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import (
    EVOLUTION_NARRATIVES_PATH,
    OUTPUT_ROOT,
    configure_logging,
    stage2_narrative_key_sets,
    write_json,
)


CORRELATION_FLOOR = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--lag-summary-csv", type=Path, default=None)
    parser.add_argument("--narratives-csv", type=Path, default=EVOLUTION_NARRATIVES_PATH)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def precedence_type_label(source_a: str, source_b: str, classification: str) -> str:
    if classification == "synchronous_or_unclear":
        return classification
    if classification == "source_a_leads":
        return f"{source_a}_leads_{source_b}"
    return f"{source_b}_leads_{source_a}"


def classify_row(row: pd.Series) -> tuple[str, str]:
    best_lag = row.get("best_lag_docs")
    best_corr = row.get("best_correlation_docs")
    low_info = bool(row.get("low_information_flag_docs", True))
    unstable = bool(row.get("unstable_series_flag_docs", True))
    if pd.isna(best_lag) or pd.isna(best_corr) or low_info or unstable or abs(float(best_corr)) < CORRELATION_FLOOR:
        classification = "synchronous_or_unclear"
        reason = "low_information_or_unclear"
    elif int(best_lag) > 0:
        classification = "source_a_leads"
        reason = "positive_best_lag_docs"
    elif int(best_lag) < 0:
        classification = "source_b_leads"
        reason = "negative_best_lag_docs"
    else:
        classification = "synchronous_or_unclear"
        reason = "zero_best_lag_docs"
    return classification, reason


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            pair_count=("pair_id", "nunique"),
            share_of_pairs=("pair_id", lambda s: float(pd.Series(s).nunique()) / frame["pair_id"].nunique()),
            mean_best_correlation_docs=("best_correlation_docs", "mean"),
            median_best_correlation_docs=("best_correlation_docs", "median"),
            mean_abs_best_lag_docs=("best_lag_docs", lambda s: float(pd.Series(s).abs().mean())),
            zero_lag_count=("best_lag_docs", lambda s: int((pd.Series(s).fillna(0) == 0).sum())),
            synchronous_or_unclear_count=(
                "precedence_type_label",
                lambda s: int((pd.Series(s) == "synchronous_or_unclear").sum()),
            ),
            stable_leading_pair_count=("is_stable_leading_pair", lambda s: int(pd.Series(s).fillna(False).sum())),
            unique_microtopics_a=("microtopic_id_a", "nunique"),
            unique_microtopics_b=("microtopic_id_b", "nunique"),
        )
        .reset_index()
    )
    return grouped


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    lag_summary_csv = args.lag_summary_csv or (args.output_root / "pair_lag_analysis_summary.csv")
    lag_summary = safe_read_csv(lag_summary_csv)
    narrative_row_keys, narrative_ready_keys, narratives = stage2_narrative_key_sets(args.narratives_csv)

    classification_path = args.output_root / "pair_precedence_classification.csv"
    join_ready_path = args.output_root / "pair_join_ready_table.csv"
    by_type_path = args.output_root / "pair_precedence_summary_by_type.csv"
    by_dyad_path = args.output_root / "pair_precedence_summary_by_dyad.csv"
    by_macro_path = args.output_root / "pair_precedence_summary_by_macro_topic.csv"
    by_type_macro_path = args.output_root / "pair_precedence_summary_by_type_macro_topic.csv"

    if lag_summary.empty:
        for path in [
            classification_path,
            join_ready_path,
            by_type_path,
            by_dyad_path,
            by_macro_path,
            by_type_macro_path,
        ]:
            pd.DataFrame().to_csv(path, index=False)
        write_json(
            args.output_root / "precedence_manifest.json",
            {
                "lag_summary_csv": str(lag_summary_csv),
                "narratives_csv": str(args.narratives_csv),
                "correlation_floor": CORRELATION_FLOOR,
                "row_counts": {
                    "pair_precedence_classification": 0,
                    "pair_join_ready_table": 0,
                },
                "outputs": {
                    "pair_precedence_classification": str(classification_path),
                    "pair_join_ready_table": str(join_ready_path),
                    "pair_precedence_summary_by_type": str(by_type_path),
                    "pair_precedence_summary_by_dyad": str(by_dyad_path),
                    "pair_precedence_summary_by_macro_topic": str(by_macro_path),
                    "pair_precedence_summary_by_type_macro_topic": str(by_type_macro_path),
                },
            },
        )
        print(args.output_root)
        return

    lag_summary = lag_summary.copy()
    classified = lag_summary.apply(lambda row: classify_row(row), axis=1, result_type="expand")
    lag_summary["precedence_class"] = classified[0]
    lag_summary["precedence_reason"] = classified[1]
    lag_summary["precedence_type_label"] = lag_summary.apply(
        lambda row: precedence_type_label(str(row["source_a"]), str(row["source_b"]), str(row["precedence_class"])),
        axis=1,
    )
    lag_summary["is_stable_leading_pair"] = lag_summary["precedence_type_label"] != "synchronous_or_unclear"

    lag_summary["has_narrative_row_a"] = lag_summary.apply(
        lambda row: (str(row["subgroup_a"]), int(row["microtopic_id_a"])) in narrative_row_keys,
        axis=1,
    )
    lag_summary["has_narrative_row_b"] = lag_summary.apply(
        lambda row: (str(row["subgroup_b"]), int(row["microtopic_id_b"])) in narrative_row_keys,
        axis=1,
    )
    lag_summary["both_have_narrative_rows"] = lag_summary["has_narrative_row_a"] & lag_summary["has_narrative_row_b"]
    lag_summary["has_narrative_ready_a"] = lag_summary.apply(
        lambda row: (str(row["subgroup_a"]), int(row["microtopic_id_a"])) in narrative_ready_keys,
        axis=1,
    )
    lag_summary["has_narrative_ready_b"] = lag_summary.apply(
        lambda row: (str(row["subgroup_b"]), int(row["microtopic_id_b"])) in narrative_ready_keys,
        axis=1,
    )
    lag_summary["has_narrative_a"] = lag_summary["has_narrative_ready_a"]
    lag_summary["has_narrative_b"] = lag_summary["has_narrative_ready_b"]
    lag_summary["both_have_narratives"] = lag_summary["has_narrative_ready_a"] & lag_summary["has_narrative_ready_b"]

    classification_cols = [
        "pair_id",
        "macro_topic",
        "macro_topic_name",
        "dyad",
        "dyad_label",
        "source_a",
        "subgroup_a",
        "microtopic_id_a",
        "microtopic_label_a",
        "source_b",
        "subgroup_b",
        "microtopic_id_b",
        "microtopic_label_b",
        "cosine_similarity",
        "best_lag_docs",
        "best_correlation_docs",
        "best_overlap_docs",
        "low_information_flag_docs",
        "unstable_series_flag_docs",
        "precedence_class",
        "precedence_reason",
        "precedence_type_label",
        "is_stable_leading_pair",
    ]
    join_ready_cols = classification_cols + [
        "lenient_is_viable",
        "balanced_is_viable",
        "strict_is_viable",
        "has_narrative_row_a",
        "has_narrative_row_b",
        "both_have_narrative_rows",
        "has_narrative_ready_a",
        "has_narrative_ready_b",
        "has_narrative_a",
        "has_narrative_b",
        "both_have_narratives",
    ]

    classification = lag_summary[[col for col in classification_cols if col in lag_summary.columns]].copy()
    join_ready = lag_summary[[col for col in join_ready_cols if col in lag_summary.columns]].copy()

    classification.to_csv(classification_path, index=False)
    join_ready.to_csv(join_ready_path, index=False)
    summarize_groups(classification, ["precedence_type_label"]).to_csv(by_type_path, index=False)
    summarize_groups(classification, ["dyad", "dyad_label"]).to_csv(by_dyad_path, index=False)
    summarize_groups(classification, ["macro_topic", "macro_topic_name"]).to_csv(by_macro_path, index=False)
    summarize_groups(classification, ["precedence_type_label", "macro_topic", "macro_topic_name"]).to_csv(
        by_type_macro_path, index=False
    )

    write_json(
        args.output_root / "precedence_manifest.json",
        {
            "lag_summary_csv": str(lag_summary_csv),
            "narratives_csv": str(args.narratives_csv),
            "correlation_floor": CORRELATION_FLOOR,
            "row_counts": {
                "pair_precedence_classification": int(classification.shape[0]),
                "pair_join_ready_table": int(join_ready.shape[0]),
                "pairs_with_both_narrative_rows": int(join_ready["both_have_narrative_rows"].fillna(False).sum()),
                "pairs_with_both_narratives": int(join_ready["both_have_narratives"].fillna(False).sum()),
            },
            "outputs": {
                "pair_precedence_classification": str(classification_path),
                "pair_join_ready_table": str(join_ready_path),
                "pair_precedence_summary_by_type": str(by_type_path),
                "pair_precedence_summary_by_dyad": str(by_dyad_path),
                "pair_precedence_summary_by_macro_topic": str(by_macro_path),
                "pair_precedence_summary_by_type_macro_topic": str(by_type_macro_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
