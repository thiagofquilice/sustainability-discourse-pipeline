#!/usr/bin/env python3
"""Run BERTopic Topics per Class for the unified 6-topic pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from workflow_common import configure_logging, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--document-topics", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--class-column", type=str, default="source")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["paths"]["full_output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    document_topics_path = args.document_topics or Path(config["paths"]["document_topics"])
    frame = pd.read_parquet(document_topics_path)
    class_values = frame[args.class_column].dropna().astype(str).unique().tolist()
    prevalence = (
        frame.groupby([args.class_column, "assigned_label"], dropna=False)
        .size()
        .rename("document_count")
        .reset_index()
    )
    prevalence.to_csv(output_dir / "label_prevalence_by_class.csv", index=False)

    if len(class_values) <= 1:
        write_json(
            output_dir / "topics_per_class_status.json",
            {
                "status": "skipped",
                "reason": f"Only one unique value found in {args.class_column}.",
                "class_values": class_values,
            },
        )
        prevalence.to_csv(output_dir / "topics_per_class.csv", index=False)
        print(f"Skipped BERTopic Topics per Class because {args.class_column} has a single value.")
        return

    model_dir = args.model_dir or Path(config["paths"]["model_dir"])
    topic_model = BERTopic.load(str(model_dir))
    topics_per_class = topic_model.topics_per_class(
        docs=frame["text"].fillna("").astype(str).tolist(),
        classes=frame[args.class_column].astype(str).tolist(),
    )
    topics_per_class.to_csv(output_dir / "topics_per_class.csv", index=False)

    try:
        figure = topic_model.visualize_topics_per_class(topics_per_class=topics_per_class)
        figure.write_html(str(output_dir / "topics_per_class.html"))
    except Exception as exc:  # noqa: BLE001
        write_json(
            output_dir / "topics_per_class_visualization_status.json",
            {"status": "failed", "reason": str(exc)},
        )

    row_totals = prevalence.groupby(args.class_column)["document_count"].transform("sum")
    prevalence["share_within_class"] = prevalence["document_count"] / row_totals
    prevalence.to_csv(output_dir / "label_prevalence_by_class_row_pct.csv", index=False)
    topic_prevalence = (
        frame.groupby([args.class_column, "bertopic_topic_id"], dropna=False)
        .size()
        .rename("document_count")
        .reset_index()
    )
    topic_totals = topic_prevalence.groupby(args.class_column)["document_count"].transform("sum")
    topic_prevalence["share_within_class"] = topic_prevalence["document_count"] / topic_totals
    topic_prevalence.to_csv(output_dir / "topic_prevalence_by_class.csv", index=False)
    print(f"Saved Topics per Class outputs to {output_dir}")


if __name__ == "__main__":
    main()
