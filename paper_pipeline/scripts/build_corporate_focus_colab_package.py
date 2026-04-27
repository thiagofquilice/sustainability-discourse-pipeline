#!/usr/bin/env python3
"""Prepare a Colab upload package for corporate-focused Stage 1/2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from microtopic_posthoc_merge_common import CORPORATE_FOCUS_STAGE12_INPUT_ROOT, ensure_directory


README_TEXT = """# Corporate-Focused Stage 1/2 Package

This package mirrors the output-root layout expected by the micro-topic evolution scripts.

Recommended Colab flow after unzipping into a working directory:

1. Stage 1
python scripts/run_micro_topic_year_summaries.py \\
  --input-csv /content/corporate_focus_stage12_input/micro_topic_year_evidence.csv \\
  --output-dir /content/corporate_focus_stage12_input/year_summaries

2. Stage 2
python scripts/run_micro_topic_evolution_synthesis.py \\
  --selected-topics /content/corporate_focus_stage12_input/selected_micro_topics.csv \\
  --annual-summaries /content/corporate_focus_stage12_input/year_summaries/micro_topic_year_summaries.csv \\
  --year-evidence /content/corporate_focus_stage12_input/micro_topic_year_evidence.csv \\
  --output-dir /content/corporate_focus_stage12_input/evolution_summaries

3. Optional summary build
python scripts/build_micro_topic_evolution_summary.py \\
  --output-root /content/corporate_focus_stage12_input

The package intentionally contains only the corporate-focused subset prepared locally.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=CORPORATE_FOCUS_STAGE12_INPUT_ROOT)
    parser.add_argument("--zip-name", type=str, default="corporate_focus_stage12_input_upload.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(args.input_root)

    required = [
        args.input_root / "selected_micro_topics.csv",
        args.input_root / "selected_micro_topics_by_subgroup.csv",
        args.input_root / "micro_topic_year_evidence.csv",
        args.input_root / "micro_topic_year_evidence.jsonl",
        args.input_root / "selection_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing Stage 1/2 input files for packaging: " + ", ".join(missing))

    readme_path = args.input_root / "README_colab_package.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    manifest_path = args.input_root / "colab_package_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "required_files": [str(path.name) for path in required],
                "readme": readme_path.name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zip_path = args.input_root.parent / args.zip_name
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(args.input_root.glob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{args.input_root.name}/{path.name}")

    print(zip_path)


if __name__ == "__main__":
    main()
