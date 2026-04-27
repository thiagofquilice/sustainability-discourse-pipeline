#!/usr/bin/env python3
"""Build the current missing Stage 2 selected-topics subset from ready narratives."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from micro_topic_evolution_common import OUTPUT_ROOT, configure_logging, is_stage2_narrative_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", type=Path, default=OUTPUT_ROOT / "selected_micro_topics.csv")
    parser.add_argument(
        "--narratives-csv",
        type=Path,
        default=OUTPUT_ROOT / "evolution_summaries" / "micro_topic_evolution_narratives.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_ROOT / "evolution_summaries" / "repair_artifacts" / "stage2_current_missing_selected_topics.csv",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    selected = pd.read_csv(args.selected_topics)
    selected["micro_topic_id"] = pd.to_numeric(selected["micro_topic_id"], errors="coerce")
    selected = selected.dropna(subset=["micro_topic_id"]).copy()
    selected["micro_topic_id"] = selected["micro_topic_id"].astype(int)

    if args.narratives_csv.exists() and args.narratives_csv.stat().st_size > 0:
        narratives = pd.read_csv(args.narratives_csv)
        if not narratives.empty:
            narratives["micro_topic_id"] = pd.to_numeric(narratives["micro_topic_id"], errors="coerce")
            narratives = narratives.dropna(subset=["micro_topic_id"]).copy()
            narratives["micro_topic_id"] = narratives["micro_topic_id"].astype(int)
            narratives = narratives.loc[narratives.apply(is_stage2_narrative_ready, axis=1)].copy()
        else:
            narratives = narratives.head(0).copy()
    else:
        narratives = pd.DataFrame(columns=["subgroup", "micro_topic_id"])

    ready_keys = narratives[["subgroup", "micro_topic_id"]].drop_duplicates()
    missing = selected.merge(ready_keys, on=["subgroup", "micro_topic_id"], how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only"].drop(columns="_merge")
    missing = missing.sort_values(["subgroup", "micro_topic_id"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
