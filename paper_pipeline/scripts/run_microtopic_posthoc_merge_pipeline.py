#!/usr/bin/env python3
"""Convenience runner for the post-hoc microtopic merge workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import MERGE_OUTPUT_ROOT, MERGED_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGE_OUTPUT_ROOT)
    parser.add_argument("--merged-output-root", type=Path, default=MERGED_OUTPUT_ROOT)
    parser.add_argument("--build-review", action="store_true")
    parser.add_argument("--materialize-mapping", action="store_true")
    parser.add_argument("--build-merged-outputs", action="store_true")
    parser.add_argument("--run-stage2", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_script(script_name: str, args: list[str]) -> None:
    script_path = Path("paper_pipeline/scripts") / script_name
    subprocess.run([sys.executable, str(script_path), *args], check=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not any([args.build_review, args.materialize_mapping, args.build_merged_outputs]):
        args.build_review = True

    if args.build_review:
        run_script(
            "build_microtopic_posthoc_merge_review.py",
            ["--output-root", str(args.output_root), "--log-level", args.log_level],
        )
    if args.materialize_mapping:
        run_script(
            "materialize_microtopic_posthoc_merge_mapping.py",
            ["--output-root", str(args.output_root), "--log-level", args.log_level],
        )
    if args.build_merged_outputs:
        extra = ["--output-root", str(args.merged_output_root), "--log-level", args.log_level]
        if args.run_stage2:
            extra.append("--run-stage2")
        run_script("build_merged_microtopic_outputs.py", extra)


if __name__ == "__main__":
    main()
