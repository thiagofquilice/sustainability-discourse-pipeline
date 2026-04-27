#!/usr/bin/env python3
"""Run BERTopic Topics over Time for the unified 6-topic pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from workflow_common import configure_logging, load_config, select_time_field, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--document-topics", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nr-bins", type=int, default=None)
    parser.add_argument("--min-rows-per-subset", type=int, default=25)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run_single_topics_over_time(
    topic_model: BERTopic,
    frame: pd.DataFrame,
    output_path: Path,
    nr_bins: int | None,
) -> None:
    topics_over_time = topic_model.topics_over_time(
        docs=frame["text"].fillna("").astype(str).tolist(),
        timestamps=frame["year"].tolist(),
        topics=frame["bertopic_topic_id"].astype(int).tolist(),
        nr_bins=nr_bins,
    )
    topics_over_time.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["paths"]["full_output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    document_topics_path = args.document_topics or Path(config["paths"]["document_topics"])
    frame = pd.read_parquet(document_topics_path)
    time_field = select_time_field(frame)
    if time_field != "year":
        frame["year"] = pd.to_datetime(frame[time_field], errors="coerce").dt.year
    frame = frame.loc[frame["year"].notna()].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    if frame.empty:
        raise SystemExit("No rows with a valid year are available for Topics over Time.")

    model_dir = args.model_dir or Path(config["paths"]["model_dir"])
    topic_model = BERTopic.load(str(model_dir))
    run_single_topics_over_time(
        topic_model,
        frame,
        output_dir / "topics_over_time_global.csv",
        args.nr_bins,
    )

    source_output_dir = output_dir / "topics_over_time_by_source"
    source_output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for source_value, subset in frame.groupby("source", dropna=False):
        if len(subset) < args.min_rows_per_subset:
            statuses.append(
                {
                    "source": source_value,
                    "status": "skipped",
                    "reason": f"Too few rows ({len(subset)}) for a stable subset analysis.",
                }
            )
            continue
        output_path = source_output_dir / f"{str(source_value)}.csv"
        run_single_topics_over_time(topic_model, subset, output_path, args.nr_bins)
        statuses.append({"source": source_value, "status": "completed", "rows": int(len(subset))})

    write_json(output_dir / "topics_over_time_status.json", {"time_field_used": "year", "statuses": statuses})
    print(f"Saved Topics over Time outputs to {output_dir}")


if __name__ == "__main__":
    main()
