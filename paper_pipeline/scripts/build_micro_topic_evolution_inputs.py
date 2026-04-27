#!/usr/bin/env python3
"""Build selection and annual evidence inputs for the micro BERTopic temporal reading workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from micro_topic_evolution_common import (
    MICRO_ROOT,
    OUTPUT_ROOT,
    SOURCE_ORDER,
    configure_logging,
    ensure_directory,
    parse_topic_terms,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=MICRO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--max-topics-per-subgroup", type=int, default=10)
    parser.add_argument("--min-topics-per-subgroup", type=int, default=1)
    parser.add_argument("--max-chunks-per-year", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def select_topics(topic_info: pd.DataFrame, threshold: float, max_topics: int, min_topics: int) -> pd.DataFrame:
    valid = topic_info.loc[topic_info["Topic"] != -1].copy()
    valid = valid.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
    total = float(valid["Count"].sum())
    if total <= 0:
        return valid.head(min_topics).copy()
    valid["share_of_non_outlier"] = valid["Count"] / total
    valid["cumulative_share"] = valid["share_of_non_outlier"].cumsum()
    if valid.empty:
        return valid
    cutoff = valid.index[valid["cumulative_share"] >= threshold]
    keep_count = min_topics if len(cutoff) == 0 else int(cutoff[0]) + 1
    keep_count = max(min_topics, min(max_topics, keep_count))
    return valid.head(keep_count).copy()


def build_year_evidence(
    subgroup: str,
    manifest: dict,
    selected_topics: pd.DataFrame,
    topics_over_time: pd.DataFrame,
    document_topics: pd.DataFrame,
    max_chunks_per_year: int,
) -> tuple[list[dict], list[dict]]:
    yearly_rows: list[dict] = []
    topic_rows: list[dict] = []

    for _, topic_row in selected_topics.iterrows():
        topic_id = int(topic_row["Topic"])
        overall_keywords = parse_topic_terms(topic_row.get("Representation"))
        topic_years = topics_over_time.loc[topics_over_time["Topic"] == topic_id].copy()
        topic_years = topic_years.sort_values("Timestamp").reset_index(drop=True)
        active_years = topic_years["Timestamp"].astype(int).tolist()
        topic_rows.append(
            {
                "subgroup": subgroup,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "macro_topic_name": manifest.get("topic_name"),
                "micro_topic_id": topic_id,
                "topic_name_original": str(topic_row["Name"]),
                "topic_size": int(topic_row["Count"]),
                "share_of_non_outlier": float(topic_row["share_of_non_outlier"]),
                "cumulative_share": float(topic_row["cumulative_share"]),
                "active_year_count": int(len(active_years)),
                "active_year_min": int(min(active_years)) if active_years else None,
                "active_year_max": int(max(active_years)) if active_years else None,
                "overall_keywords": json.dumps(overall_keywords, ensure_ascii=False),
            }
        )

        for _, year_row in topic_years.iterrows():
            year = int(year_row["Timestamp"])
            year_subset = document_topics.loc[
                (document_topics["micro_topic_id"] == topic_id) & (document_topics["year"].astype(int) == year)
            ].copy()
            year_subset = year_subset.sort_values(
                ["micro_topic_probability", "chunk_id"], ascending=[False, True]
            ).head(max_chunks_per_year)
            chunk_records = []
            for _, chunk_row in year_subset.iterrows():
                chunk_records.append(
                    {
                        "chunk_id": str(chunk_row["chunk_id"]),
                        "source_doc_id": str(chunk_row["source_doc_id"]),
                        "year": int(chunk_row["year"]),
                        "micro_topic_probability": float(chunk_row["micro_topic_probability"]),
                        "text": str(chunk_row["text"]),
                    }
                )

            yearly_rows.append(
                {
                    "subgroup": subgroup,
                    "source": manifest["source"],
                    "assigned_label": manifest["assigned_label"],
                    "macro_topic_name": manifest.get("topic_name"),
                    "micro_topic_id": topic_id,
                    "topic_name_original": str(topic_row["Name"]),
                    "year": year,
                    "year_frequency": int(year_row["Frequency"]),
                    "topic_size": int(topic_row["Count"]),
                    "share_of_non_outlier": float(topic_row["share_of_non_outlier"]),
                    "year_specific_words": json.dumps(parse_topic_terms(year_row["Words"]), ensure_ascii=False),
                    "overall_keywords": json.dumps(overall_keywords, ensure_ascii=False),
                    "available_chunk_count": int(year_subset.shape[0]),
                    "is_sparse_year": bool(year_subset.shape[0] < max_chunks_per_year),
                    "chunk_records_json": json.dumps(chunk_records, ensure_ascii=False),
                    "chunk_id_1": chunk_records[0]["chunk_id"] if len(chunk_records) > 0 else "",
                    "chunk_id_2": chunk_records[1]["chunk_id"] if len(chunk_records) > 1 else "",
                    "chunk_id_3": chunk_records[2]["chunk_id"] if len(chunk_records) > 2 else "",
                    "chunk_id_4": chunk_records[3]["chunk_id"] if len(chunk_records) > 3 else "",
                    "chunk_id_5": chunk_records[4]["chunk_id"] if len(chunk_records) > 4 else "",
                    "chunk_text_1": chunk_records[0]["text"] if len(chunk_records) > 0 else "",
                    "chunk_text_2": chunk_records[1]["text"] if len(chunk_records) > 1 else "",
                    "chunk_text_3": chunk_records[2]["text"] if len(chunk_records) > 2 else "",
                    "chunk_text_4": chunk_records[3]["text"] if len(chunk_records) > 3 else "",
                    "chunk_text_5": chunk_records[4]["text"] if len(chunk_records) > 4 else "",
                }
            )
    return topic_rows, yearly_rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_root = args.output_root
    ensure_directory(output_root)

    selected_topic_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    yearly_rows: list[dict] = []

    subgroup_dirs = sorted(
        path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary"
    )
    for subgroup_dir in subgroup_dirs:
        manifest_path = subgroup_dir / "manifest.json"
        topic_info_path = subgroup_dir / "topic_info.csv"
        topics_over_time_path = subgroup_dir / "topics_over_time.csv"
        document_topics_path = subgroup_dir / "document_topics.parquet"
        if not all(path.exists() for path in [manifest_path, topic_info_path, topics_over_time_path, document_topics_path]):
            continue

        manifest = read_json(manifest_path)
        topic_info = pd.read_csv(topic_info_path)
        topics_over_time = pd.read_csv(topics_over_time_path)
        document_topics = pd.read_parquet(document_topics_path)
        selected = select_topics(
            topic_info=topic_info,
            threshold=args.coverage_threshold,
            max_topics=args.max_topics_per_subgroup,
            min_topics=args.min_topics_per_subgroup,
        )
        topic_rows, group_year_rows = build_year_evidence(
            subgroup=subgroup_dir.name,
            manifest=manifest,
            selected_topics=selected,
            topics_over_time=topics_over_time,
            document_topics=document_topics,
            max_chunks_per_year=args.max_chunks_per_year,
        )
        selected_topic_rows.extend(topic_rows)
        yearly_rows.extend(group_year_rows)
        subgroup_rows.append(
            {
                "subgroup": subgroup_dir.name,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "status": manifest["status"],
                "warning": manifest.get("warning", ""),
                "row_count": int(manifest["row_count"]),
                "selected_topic_count": int(len(selected)),
                "selected_topic_coverage": float(selected["share_of_non_outlier"].sum()) if not selected.empty else 0.0,
                "selected_topic_ids": json.dumps(selected["Topic"].astype(int).tolist()),
                "selected_topic_names": json.dumps(selected["Name"].astype(str).tolist(), ensure_ascii=False),
            }
        )

    selected_topics_frame = pd.DataFrame(selected_topic_rows)
    subgroup_frame = pd.DataFrame(subgroup_rows)
    yearly_frame = pd.DataFrame(yearly_rows)

    if not selected_topics_frame.empty:
        selected_topics_frame["source"] = pd.Categorical(selected_topics_frame["source"], categories=SOURCE_ORDER, ordered=True)
        selected_topics_frame = selected_topics_frame.sort_values(["source", "assigned_label", "micro_topic_id"]).reset_index(drop=True)
    if not subgroup_frame.empty:
        subgroup_frame["source"] = pd.Categorical(subgroup_frame["source"], categories=SOURCE_ORDER, ordered=True)
        subgroup_frame = subgroup_frame.sort_values(["source", "assigned_label"]).reset_index(drop=True)
    if not yearly_frame.empty:
        yearly_frame["source"] = pd.Categorical(yearly_frame["source"], categories=SOURCE_ORDER, ordered=True)
        yearly_frame = yearly_frame.sort_values(["source", "assigned_label", "micro_topic_id", "year"]).reset_index(drop=True)

    selected_topics_frame.to_csv(output_root / "selected_micro_topics.csv", index=False)
    subgroup_frame.to_csv(output_root / "selected_micro_topics_by_subgroup.csv", index=False)
    yearly_frame.to_csv(output_root / "micro_topic_year_evidence.csv", index=False)

    jsonl_path = output_root / "micro_topic_year_evidence.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in yearly_frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(
        output_root / "selection_manifest.json",
        {
            "micro_root": str(args.micro_root),
            "coverage_threshold": args.coverage_threshold,
            "max_topics_per_subgroup": args.max_topics_per_subgroup,
            "min_topics_per_subgroup": args.min_topics_per_subgroup,
            "max_chunks_per_year": args.max_chunks_per_year,
            "subgroup_count": int(subgroup_frame.shape[0]),
            "selected_micro_topic_count": int(selected_topics_frame.shape[0]),
            "annual_evidence_row_count": int(yearly_frame.shape[0]),
            "outputs": {
                "selected_micro_topics": str(output_root / "selected_micro_topics.csv"),
                "selected_micro_topics_by_subgroup": str(output_root / "selected_micro_topics_by_subgroup.csv"),
                "micro_topic_year_evidence": str(output_root / "micro_topic_year_evidence.csv"),
                "micro_topic_year_evidence_jsonl": str(jsonl_path),
            },
        },
    )

    print(output_root)


if __name__ == "__main__":
    main()
