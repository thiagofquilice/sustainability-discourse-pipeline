#!/usr/bin/env python3
"""Run the cross-source matched microtopic temporal analysis pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cross_source_microtopic_common import DEFAULT_THRESHOLD_065_OUTPUT_ROOT, OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--main-threshold", type=float, default=0.75)
    parser.add_argument("--include-llm-enrichment", action="store_true")
    parser.add_argument("--narratives-csv", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_step(script_name: str, output_root: Path, log_level: str, extra_args: list[str] | None = None) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script_path), "--output-root", str(output_root), "--log-level", log_level]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[cross-source] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    selected_output_root = args.output_root
    if selected_output_root == OUTPUT_ROOT and abs(args.main_threshold - 0.65) < 1e-9:
        selected_output_root = DEFAULT_THRESHOLD_065_OUTPUT_ROOT

    for script_name, extra_args in [
        (
            "build_cross_source_microtopic_profiles.py",
            ["--micro-root", str(args.micro_root)] if args.micro_root is not None else None,
        ),
        ("embed_cross_source_microtopic_profiles.py", None),
        ("match_cross_source_microtopics.py", None),
        (
            "filter_cross_source_microtopic_pairs_temporal.py",
            ["--main-threshold", str(args.main_threshold)],
        ),
        ("build_cross_source_microtopic_pair_series.py", None),
        ("summarize_cross_source_microtopic_pairs.py", None),
        (
            "analyze_cross_source_microtopic_precedence.py",
            ["--main-threshold", str(args.main_threshold)],
        ),
        ("classify_cross_source_pair_precedence.py", None),
    ]:
        run_step(script_name, selected_output_root, args.log_level, extra_args=extra_args)
    if args.include_llm_enrichment:
        enrichment_args: list[str] = []
        if args.narratives_csv is not None:
            enrichment_args.extend(["--narratives-csv", str(args.narratives_csv)])
        run_step(
            "enrich_cross_source_pairs_with_llm_evolution.py",
            selected_output_root,
            args.log_level,
            extra_args=enrichment_args or None,
        )
    print(selected_output_root)


if __name__ == "__main__":
    main()
