#!/usr/bin/env python3
"""Run the same-issue / different-discourse-function comparison pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cross_source_microtopic_common import (
    DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT,
    DEFAULT_THRESHOLD_065_OUTPUT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_THRESHOLD_065_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT)
    parser.add_argument("--narratives-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--dyads", type=str, nargs="*", default=None)
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_step(script_name: str, extra_args: list[str]) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"[same-issue] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    common = ["--output-root", str(args.output_root)]

    run_step(
        "build_same_issue_discourse_candidates.py",
        [
            "--pair-root",
            str(args.pair_root),
            "--output-root",
            str(args.output_root),
            *(["--narratives-csv", str(args.narratives_csv)] if args.narratives_csv else []),
            "--top-k",
            str(args.top_k),
            "--min-similarity",
            str(args.min_similarity),
            *(["--dyads", *args.dyads] if args.dyads else []),
            "--log-level",
            args.log_level,
        ],
    )
    run_step(
        "compare_same_issue_discourse_pairs.py",
        [
            *common,
            *(["--narratives-csv", str(args.narratives_csv)] if args.narratives_csv else []),
            "--model",
            args.model,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-attempts",
            str(args.max_attempts),
            "--log-level",
            args.log_level,
        ],
    )
    run_step(
        "summarize_same_issue_discourse_pairs.py",
        [*common, *(["--narratives-csv", str(args.narratives_csv)] if args.narratives_csv else [])],
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
