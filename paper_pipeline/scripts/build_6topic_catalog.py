#!/usr/bin/env python3
"""Materialize the canonical 6-topic discourse catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from six_topic_discourse_catalog import TOPIC_GROUPS, bullet_list_text, catalog_rows


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = WORKFLOW_ROOT / "catalog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog = pd.DataFrame(catalog_rows()).sort_values("topic_id").reset_index(drop=True)
    catalog_csv = args.output_dir / "six_topic_discourse_catalog.csv"
    catalog_json = args.output_dir / "six_topic_discourse_catalog.json"
    manifest_json = args.output_dir / "six_topic_discourse_catalog_manifest.json"

    catalog.to_csv(catalog_csv, index=False)
    catalog.to_json(catalog_json, orient="records", indent=2, force_ascii=False)

    manifest = {
        "topic_count": len(TOPIC_GROUPS),
        "topic_codes": [topic["topic_code"] for topic in TOPIC_GROUPS],
        "topics": [
            {
                "topic_code": topic["topic_code"],
                "topic_name": topic["topic_name"],
                "topic_definition": topic["topic_definition"],
                "topic_elements_bullets": bullet_list_text(topic["elements"]),
                "sdg_crosswalk": topic["sdg_crosswalk"],
                "boundary_notes": topic["boundary_notes"],
                "false_positive_notes": topic["false_positive_notes"],
            }
            for topic in TOPIC_GROUPS
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(catalog_csv)


if __name__ == "__main__":
    main()
