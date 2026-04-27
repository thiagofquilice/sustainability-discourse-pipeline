#!/usr/bin/env python3
"""Run the merge-first redesign pipeline with isolated output roots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGED_MICRO_ROOT,
    MERGE_FIRST_DISCOURSE_OUTPUT_ROOT,
    MERGE_FIRST_EVOLUTION_OUTPUT_ROOT,
    MERGE_FIRST_GROUP_REVIEW_OUTPUT_ROOT,
    MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
    MERGE_FIRST_PAIR_OUTPUT_ROOT,
    MERGE_FIRST_REVIEW_OUTPUT_ROOT,
    RAW_MICRO_ROOT,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-micro-root", type=Path, default=RAW_MICRO_ROOT)
    parser.add_argument("--review-output-root", type=Path, default=MERGE_FIRST_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--group-review-output-root", type=Path, default=MERGE_FIRST_GROUP_REVIEW_OUTPUT_ROOT)
    parser.add_argument(
        "--hierarchical-review-output-root",
        type=Path,
        default=MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
    )
    parser.add_argument("--merged-micro-root", type=Path, default=MERGED_MICRO_ROOT)
    parser.add_argument("--evolution-output-root", type=Path, default=MERGE_FIRST_EVOLUTION_OUTPUT_ROOT)
    parser.add_argument("--pair-output-root", type=Path, default=MERGE_FIRST_PAIR_OUTPUT_ROOT)
    parser.add_argument("--discourse-output-root", type=Path, default=MERGE_FIRST_DISCOURSE_OUTPUT_ROOT)
    parser.add_argument("--main-threshold", type=float, default=0.65)
    parser.add_argument("--include-llm-enrichment", action="store_true")
    parser.add_argument("--run-review", action="store_true")
    parser.add_argument("--run-group-review", action="store_true")
    parser.add_argument("--run-hierarchical-review", action="store_true")
    parser.add_argument("--materialize-mapping", action="store_true")
    parser.add_argument("--materialize-group-mapping", action="store_true")
    parser.add_argument("--materialize-hierarchical-mapping", action="store_true")
    parser.add_argument("--use-legacy-review-mapping", action="store_true")
    parser.add_argument("--use-group-review-mapping", action="store_true")
    parser.add_argument("--build-merged-root", action="store_true")
    parser.add_argument("--build-evolution-inputs", action="store_true")
    parser.add_argument("--run-evolution", action="store_true")
    parser.add_argument("--run-cross-source", action="store_true")
    parser.add_argument("--run-discourse", action="store_true")
    parser.add_argument("--run-temporal-grouping", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_script(script_name: str, args: list[str]) -> None:
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path), *args]
    print(f"[merge-first] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not any(
        [
            args.run_review,
            args.run_group_review,
            args.run_hierarchical_review,
            args.materialize_mapping,
            args.materialize_group_mapping,
            args.materialize_hierarchical_mapping,
            args.build_merged_root,
            args.build_evolution_inputs,
            args.run_evolution,
            args.run_cross_source,
            args.run_discourse,
            args.run_temporal_grouping,
        ]
    ):
        args.run_hierarchical_review = True

    if args.use_legacy_review_mapping:
        mapping_csv = args.review_output_root / "microtopic_to_merged_group.csv"
    elif args.use_group_review_mapping:
        mapping_csv = args.group_review_output_root / "microtopic_to_merged_group.csv"
    else:
        mapping_csv = args.hierarchical_review_output_root / "microtopic_to_merged_group.csv"
    narratives_csv = args.evolution_output_root / "evolution_summaries" / "micro_topic_evolution_narratives.csv"

    if args.run_review:
        run_script(
            "build_microtopic_merge_first_review.py",
            [
                "--micro-root",
                str(args.raw_micro_root),
                "--output-root",
                str(args.review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_group_review:
        run_script(
            "build_microtopic_merge_first_group_review.py",
            [
                "--input-root",
                str(args.review_output_root),
                "--output-root",
                str(args.group_review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_hierarchical_review:
        run_script(
            "build_microtopic_merge_first_hierarchical_review.py",
            [
                "--micro-root",
                str(args.raw_micro_root),
                "--output-root",
                str(args.hierarchical_review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.materialize_mapping:
        run_script(
            "materialize_microtopic_posthoc_merge_mapping.py",
            [
                "--output-root",
                str(args.review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.materialize_group_mapping:
        run_script(
            "materialize_microtopic_group_review_mapping.py",
            [
                "--output-root",
                str(args.group_review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.materialize_hierarchical_mapping:
        run_script(
            "materialize_microtopic_hierarchical_review_mapping.py",
            [
                "--output-root",
                str(args.hierarchical_review_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.build_merged_root:
        run_script(
            "build_merged_microtopic_root.py",
            [
                "--micro-root",
                str(args.raw_micro_root),
                "--mapping-csv",
                str(mapping_csv),
                "--output-root",
                str(args.merged_micro_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.build_evolution_inputs:
        run_script(
            "build_merge_first_evolution_inputs.py",
            [
                "--micro-root",
                str(args.merged_micro_root),
                "--mapping-csv",
                str(mapping_csv),
                "--output-root",
                str(args.evolution_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_evolution:
        run_script(
            "run_micro_topic_evolution_pipeline.py",
            [
                "--micro-root",
                str(args.merged_micro_root),
                "--output-root",
                str(args.evolution_output_root),
                "--skip-inputs",
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_cross_source:
        cross_source_args = [
            "--micro-root",
            str(args.merged_micro_root),
            "--output-root",
            str(args.pair_output_root),
            "--main-threshold",
            str(args.main_threshold),
            "--narratives-csv",
            str(narratives_csv),
            "--log-level",
            args.log_level,
        ]
        if args.include_llm_enrichment:
            cross_source_args.append("--include-llm-enrichment")
        run_script("run_cross_source_microtopic_pipeline.py", cross_source_args)

    if args.run_discourse:
        run_script(
            "run_same_issue_discourse_pipeline.py",
            [
                "--pair-root",
                str(args.pair_output_root),
                "--output-root",
                str(args.discourse_output_root),
                "--narratives-csv",
                str(narratives_csv),
                "--dyads",
                "academic__corporate",
                "media__corporate",
                "academic__media",
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_temporal_grouping:
        run_script(
            "analyze_same_issue_discourse_temporal_grouping.py",
            [
                "--pair-root",
                str(args.pair_output_root),
                "--output-root",
                str(args.discourse_output_root),
                "--log-level",
                args.log_level,
            ],
        )

    if args.run_hierarchical_review or args.materialize_hierarchical_mapping:
        print(args.hierarchical_review_output_root)
    elif args.run_group_review or args.materialize_group_mapping:
        print(args.group_review_output_root)
    else:
        print(args.review_output_root)


if __name__ == "__main__":
    main()
