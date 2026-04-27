#!/usr/bin/env python3
"""Run the BERTopic continuation phase for the unified 6-topic pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKFLOW_ROOT = Path("paper_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--skip-supervised", action="store_true")
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_script(script_name: str, args: list[str]) -> None:
    command = [sys.executable, str(WORKFLOW_ROOT / "scripts" / script_name), *args]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_materialize:
        run_script("materialize_adjusted_full_validation_output.py", [])
    if not args.skip_supervised:
        run_script("run_6topic_supervised_pipeline.py", ["--log-level", args.log_level])
    if not args.skip_micro:
        run_script("run_6topic_micro_unsupervised.py", ["--log-level", args.log_level])
    print(WORKFLOW_ROOT / "outputs")


if __name__ == "__main__":
    main()
