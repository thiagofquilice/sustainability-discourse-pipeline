#!/usr/bin/env python3
"""Materialize a merged canonical microtopic root from approved within-subgroup mappings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGED_MICRO_ROOT,
    MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
    RAW_MICRO_ROOT,
    ensure_directory,
    json_dumps,
    normalize_text,
    parse_json_list,
    safe_read_csv,
    write_json,
)


TOPIC_INFO_COLUMNS = ["Topic", "Count", "Name", "Representation", "Representative_Docs"]
HIERARCHY_COLUMNS = ["Parent_ID", "Parent_Name", "Topics", "Distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=RAW_MICRO_ROOT)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT / "microtopic_to_merged_group.csv",
    )
    parser.add_argument("--output-root", type=Path, default=MERGED_MICRO_ROOT)
    parser.add_argument("--max-representative-chunks", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def parse_topic_terms(raw_value: Any) -> list[str]:
    values = parse_json_list(raw_value)
    if values:
        return [normalize_text(item) for item in values if normalize_text(item)]
    text = normalize_text(raw_value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]


def stable_unique(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def slugify(parts: list[str], fallback: str) -> str:
    text = "_".join(part.lower() for part in parts if normalize_text(part))
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback.lower()


def ensure_mapping_columns(mapping: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subgroup",
        "source",
        "assigned_label",
        "micro_topic_id",
        "final_merge_group_id",
        "is_singleton_group",
        "group_size",
        "group_members_json",
    }
    missing = [column for column in required if column not in mapping.columns]
    if missing:
        raise SystemExit(f"Mapping CSV is missing required columns: {missing}")
    mapping = mapping.copy()
    mapping["subgroup"] = mapping["subgroup"].astype(str)
    mapping["source"] = mapping["source"].astype(str)
    mapping["assigned_label"] = mapping["assigned_label"].astype(str)
    mapping["micro_topic_id"] = pd.to_numeric(mapping["micro_topic_id"], errors="coerce")
    mapping = mapping.dropna(subset=["micro_topic_id"]).copy()
    mapping["micro_topic_id"] = mapping["micro_topic_id"].astype(int)
    mapping["group_size"] = pd.to_numeric(mapping["group_size"], errors="coerce").fillna(1).astype(int)
    mapping["is_singleton_group"] = mapping["is_singleton_group"].fillna(False).astype(bool)
    return mapping


def build_group_id_map(mapping_subgroup: pd.DataFrame) -> pd.DataFrame:
    groups = (
        mapping_subgroup.groupby("final_merge_group_id", dropna=False)
        .agg(
            min_member_id=("micro_topic_id", "min"),
            group_size=("group_size", "max"),
            is_singleton_group=("is_singleton_group", "max"),
            group_members_json=("group_members_json", "first"),
        )
        .reset_index()
        .sort_values(["min_member_id", "final_merge_group_id"])
        .reset_index(drop=True)
    )
    groups["merged_micro_topic_id"] = range(len(groups))
    return groups


def build_topic_info(
    group_map: pd.DataFrame,
    mapping_subgroup: pd.DataFrame,
    raw_topic_info: pd.DataFrame,
    merged_document_topics: pd.DataFrame,
    max_representative_chunks: int,
) -> pd.DataFrame:
    topic_meta = raw_topic_info.set_index("Topic").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for group_row in group_map.itertuples(index=False):
        member_ids = sorted(mapping_subgroup.loc[
            mapping_subgroup["final_merge_group_id"] == group_row.final_merge_group_id, "micro_topic_id"
        ].astype(int).tolist())
        member_meta = [
            topic_meta[int(member_id)]
            for member_id in member_ids
            if int(member_id) in topic_meta
        ]
        member_names = [normalize_text(meta.get("Name")) for meta in member_meta if normalize_text(meta.get("Name"))]
        term_pool: list[str] = []
        for meta in sorted(member_meta, key=lambda item: int(item.get("Count", 0)), reverse=True):
            term_pool.extend(parse_topic_terms(meta.get("Representation")))
        top_terms = stable_unique(term_pool, limit=10)
        rep_subset = (
            merged_document_topics.loc[merged_document_topics["micro_topic_id"] == int(group_row.merged_micro_topic_id)]
            .sort_values(["micro_topic_probability", "chunk_id"], ascending=[False, True])
            .drop_duplicates(subset=["chunk_id"], keep="first")
            .head(max_representative_chunks)
        )
        rep_texts = rep_subset["text"].fillna("").astype(str).tolist()
        count = int(sum(int(meta.get("Count", 0)) for meta in member_meta))
        if bool(group_row.is_singleton_group) and member_names:
            name = member_names[0]
        else:
            slug = slugify(top_terms[:4], fallback=normalize_text(group_row.final_merge_group_id))
            name = f"{int(group_row.merged_micro_topic_id)}_{slug}"
        rows.append(
            {
                "Topic": int(group_row.merged_micro_topic_id),
                "Count": count,
                "Name": name,
                "Representation": json_dumps(top_terms),
                "Representative_Docs": json_dumps(rep_texts),
                "final_merge_group_id": str(group_row.final_merge_group_id),
                "group_size": int(group_row.group_size),
                "is_singleton_group": bool(group_row.is_singleton_group),
                "member_micro_topic_ids_json": group_row.group_members_json,
                "member_topic_names_json": json_dumps(member_names),
            }
        )
    return pd.DataFrame(rows).sort_values(["Topic"]).reset_index(drop=True)


def build_topics_over_time(
    raw_topics_over_time: pd.DataFrame,
    raw_to_merged: dict[int, int],
    merged_id_to_group: dict[int, str],
) -> pd.DataFrame:
    frame = raw_topics_over_time.copy()
    frame["Topic"] = pd.to_numeric(frame["Topic"], errors="coerce")
    frame["Timestamp"] = pd.to_numeric(frame["Timestamp"], errors="coerce")
    frame = frame.dropna(subset=["Topic", "Timestamp"]).copy()
    frame["Topic"] = frame["Topic"].astype(int)
    frame["Timestamp"] = frame["Timestamp"].astype(int)
    frame = frame.loc[frame["Topic"].isin(raw_to_merged)].copy()
    frame["merged_micro_topic_id"] = frame["Topic"].map(raw_to_merged)

    rows: list[dict[str, Any]] = []
    for (merged_id, year), group in frame.groupby(["merged_micro_topic_id", "Timestamp"], dropna=False):
        words: list[str] = []
        for value in group["Words"].fillna("").astype(str).tolist():
            words.extend(parse_topic_terms(value))
        rows.append(
            {
                "Topic": int(merged_id),
                "Words": ", ".join(stable_unique(words, limit=5)),
                "Frequency": int(pd.to_numeric(group["Frequency"], errors="coerce").fillna(0).sum()),
                "Timestamp": int(year),
                "final_merge_group_id": merged_id_to_group[int(merged_id)],
            }
        )
    return pd.DataFrame(rows).sort_values(["Topic", "Timestamp"]).reset_index(drop=True)


def build_representative_docs(
    merged_document_topics: pd.DataFrame,
    group_lookup: dict[int, str],
    max_representative_chunks: int,
) -> pd.DataFrame:
    reps = (
        merged_document_topics.sort_values(
            ["micro_topic_id", "micro_topic_probability", "chunk_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["micro_topic_id", "chunk_id"], keep="first")
        .groupby("micro_topic_id", dropna=False)
        .head(max_representative_chunks)
        .copy()
    )
    reps["final_merge_group_id"] = reps["micro_topic_id"].map(group_lookup)
    keep_cols = [
        "micro_topic_id",
        "chunk_id",
        "source_doc_id",
        "year",
        "text",
        "final_merge_group_id",
        "raw_micro_topic_id",
    ]
    keep_cols = [col for col in keep_cols if col in reps.columns]
    return reps[keep_cols].reset_index(drop=True)


def subgroup_manifest(
    raw_manifest: dict[str, Any],
    merged_document_topics: pd.DataFrame,
    topic_info: pd.DataFrame,
    mapping_subgroup: pd.DataFrame,
) -> dict[str, Any]:
    manifest = dict(raw_manifest)
    manifest["row_count"] = int(merged_document_topics.shape[0])
    manifest["unique_docs"] = int(merged_document_topics["source_doc_id"].nunique()) if not merged_document_topics.empty else 0
    if not merged_document_topics.empty:
        years = pd.to_numeric(merged_document_topics["year"], errors="coerce").dropna().astype(int)
        manifest["year_min"] = int(years.min()) if not years.empty else None
        manifest["year_max"] = int(years.max()) if not years.empty else None
        manifest["year_count"] = int(years.nunique())
    manifest["n_topics_found"] = int(topic_info.shape[0])
    manifest["largest_topic_size"] = int(topic_info["Count"].max()) if not topic_info.empty else 0
    manifest["merged_from_review"] = True
    manifest["final_merge_group_count"] = int(mapping_subgroup["final_merge_group_id"].nunique())
    return manifest


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    ensure_directory(args.output_root)
    mapping = ensure_mapping_columns(pd.read_csv(args.mapping_csv))

    summary_rows: list[dict[str, Any]] = []
    keyword_rows: list[dict[str, Any]] = []
    representative_index_rows: list[dict[str, Any]] = []

    subgroup_dirs = sorted(path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary")
    for subgroup_dir in subgroup_dirs:
        subgroup = subgroup_dir.name
        mapping_subgroup = mapping.loc[mapping["subgroup"] == subgroup].copy()
        if mapping_subgroup.empty:
            continue

        raw_manifest_json = json.loads((subgroup_dir / "manifest.json").read_text(encoding="utf-8"))
        raw_topic_info = pd.read_csv(subgroup_dir / "topic_info.csv")
        raw_topics_over_time = pd.read_csv(subgroup_dir / "topics_over_time.csv")
        raw_document_topics = pd.read_parquet(subgroup_dir / "document_topics.parquet")

        raw_document_topics["micro_topic_id"] = pd.to_numeric(raw_document_topics["micro_topic_id"], errors="coerce")
        raw_document_topics = raw_document_topics.dropna(subset=["micro_topic_id"]).copy()
        raw_document_topics["micro_topic_id"] = raw_document_topics["micro_topic_id"].astype(int)

        raw_topic_info["Topic"] = pd.to_numeric(raw_topic_info["Topic"], errors="coerce")
        raw_topic_info = raw_topic_info.dropna(subset=["Topic"]).copy()
        raw_topic_info["Topic"] = raw_topic_info["Topic"].astype(int)
        raw_topic_info = raw_topic_info.loc[raw_topic_info["Topic"] != -1].copy()

        group_map = build_group_id_map(mapping_subgroup)
        mapping_subgroup = mapping_subgroup.merge(
            group_map[["final_merge_group_id", "merged_micro_topic_id"]],
            on="final_merge_group_id",
            how="left",
        )
        raw_to_merged = {
            int(row.micro_topic_id): int(row.merged_micro_topic_id)
            for row in mapping_subgroup.itertuples(index=False)
        }
        if not raw_to_merged:
            continue

        valid_document_topics = raw_document_topics.loc[raw_document_topics["micro_topic_id"].isin(raw_to_merged)].copy()
        if valid_document_topics.empty:
            continue
        valid_document_topics["raw_micro_topic_id"] = valid_document_topics["micro_topic_id"].astype(int)
        valid_document_topics["micro_topic_id"] = valid_document_topics["raw_micro_topic_id"].map(raw_to_merged).astype(int)
        valid_document_topics = valid_document_topics.merge(
            mapping_subgroup[
                ["micro_topic_id", "final_merge_group_id", "merged_micro_topic_id", "group_size", "group_members_json"]
            ].rename(columns={"micro_topic_id": "raw_micro_topic_id"}),
            on="raw_micro_topic_id",
            how="left",
        )
        valid_document_topics["micro_topic_id"] = valid_document_topics["merged_micro_topic_id"].astype(int)

        group_lookup = {
            int(row.merged_micro_topic_id): str(row.final_merge_group_id)
            for row in group_map.itertuples(index=False)
        }
        topic_info = build_topic_info(
            group_map=group_map,
            mapping_subgroup=mapping_subgroup,
            raw_topic_info=raw_topic_info,
            merged_document_topics=valid_document_topics,
            max_representative_chunks=args.max_representative_chunks,
        )
        topics_over_time = build_topics_over_time(
            raw_topics_over_time=raw_topics_over_time,
            raw_to_merged=raw_to_merged,
            merged_id_to_group=group_lookup,
        )
        representative_docs = build_representative_docs(
            merged_document_topics=valid_document_topics,
            group_lookup=group_lookup,
            max_representative_chunks=args.max_representative_chunks,
        )

        subgroup_output = args.output_root / subgroup
        ensure_directory(subgroup_output)
        valid_document_topics.to_parquet(subgroup_output / "document_topics.parquet", index=False)
        valid_document_topics.to_csv(subgroup_output / "document_topics.csv", index=False)
        topic_info.to_csv(subgroup_output / "topic_info.csv", index=False)
        topics_over_time.to_csv(subgroup_output / "topics_over_time.csv", index=False)
        representative_docs.to_csv(subgroup_output / "representative_docs.csv", index=False)
        pd.DataFrame(columns=HIERARCHY_COLUMNS).to_csv(subgroup_output / "hierarchical_topics.csv", index=False)

        manifest_payload = subgroup_manifest(
            raw_manifest=raw_manifest_json,
            merged_document_topics=valid_document_topics,
            topic_info=topic_info,
            mapping_subgroup=mapping_subgroup,
        )
        manifest_payload["source"] = str(mapping_subgroup["source"].iloc[0])
        manifest_payload["assigned_label"] = str(mapping_subgroup["assigned_label"].iloc[0])
        write_json(subgroup_output / "manifest.json", manifest_payload)

        summary_rows.append(manifest_payload)
        for row in topic_info.itertuples(index=False):
            keyword_rows.append(
                {
                    "subgroup": subgroup,
                    "source": manifest_payload["source"],
                    "assigned_label": manifest_payload["assigned_label"],
                    "micro_topic_id": int(row.Topic),
                    "final_merge_group_id": str(row.final_merge_group_id),
                    "topic_name_original": str(row.Name),
                    "group_size": int(row.group_size),
                    "member_micro_topic_ids_json": row.member_micro_topic_ids_json,
                    "top_terms_json": row.Representation,
                }
            )
        if not representative_docs.empty:
            representative_index_rows.extend(representative_docs.to_dict(orient="records"))

    summary_dir = args.output_root / "summary"
    ensure_directory(summary_dir)
    subgroup_summary = pd.DataFrame(summary_rows).sort_values(["source", "assigned_label"]).reset_index(drop=True) if summary_rows else pd.DataFrame()
    top_keywords = pd.DataFrame(keyword_rows)
    representative_index = pd.DataFrame(representative_index_rows)
    subgroup_summary.to_csv(summary_dir / "subgroup_manifest_summary.csv", index=False)
    top_keywords.to_csv(summary_dir / "top_keywords_summary.csv", index=False)
    representative_index.to_csv(summary_dir / "representative_docs_index.csv", index=False)
    write_json(
        summary_dir / "manifest.json",
        {
            "micro_root": str(args.output_root),
            "mapping_csv": str(args.mapping_csv),
            "subgroup_count": int(subgroup_summary.shape[0]),
            "outputs": {
                "subgroup_manifest_summary": str(summary_dir / "subgroup_manifest_summary.csv"),
                "top_keywords_summary": str(summary_dir / "top_keywords_summary.csv"),
                "representative_docs_index": str(summary_dir / "representative_docs_index.csv"),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
