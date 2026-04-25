#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-facing reproduction pipeline.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(script_dir / "01_build_macro_results.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "02_render_macro_figures.py"),
            "--results-dir",
            str(args.output_dir),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "03_render_longitudinal_panels.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )


if __name__ == "__main__":
    main()
