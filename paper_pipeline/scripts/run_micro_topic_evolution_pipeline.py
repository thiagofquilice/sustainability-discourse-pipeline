#!/usr/bin/env python3
"""Run the end-to-end micro BERTopic temporal reading workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from micro_topic_evolution_common import OUTPUT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-inputs", action="store_true")
    parser.add_argument("--skip-year-summaries", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_step(script_name: str, *extra_args: str) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *extra_args]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_inputs:
        input_args = ["--output-root", str(args.output_root), "--log-level", args.log_level]
        if args.micro_root is not None:
            input_args.extend(["--micro-root", str(args.micro_root)])
        run_step("build_micro_topic_evolution_inputs.py", *input_args)
    if not args.skip_year_summaries:
        run_step("run_micro_topic_year_summaries.py", "--output-dir", str(args.output_root / "year_summaries"), "--log-level", args.log_level)
    if not args.skip_synthesis:
        run_step("run_micro_topic_evolution_synthesis.py", "--output-dir", str(args.output_root / "evolution_summaries"), "--log-level", args.log_level)
    if not args.skip_summary:
        run_step("build_micro_topic_evolution_summary.py", "--output-root", str(args.output_root), "--log-level", args.log_level)


if __name__ == "__main__":
    main()
