#!/usr/bin/env python3
"""Build canonical microtopic profile tables for cross-source matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import (
    MACRO_TOPIC_ORDER,
    MICRO_ROOT,
    OUTPUT_ROOT,
    SOURCE_ORDER,
    build_profile_text,
    clean_microtopic_label,
    configure_logging,
    ensure_directory,
    json_dumps,
    list_subgroup_dirs,
    ordered_frame,
    parse_topic_terms,
    read_json,
    subgroup_macro_topic_name,
    truncate_text,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=MICRO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-representative-chunks", type=int, default=5)
    parser.add_argument("--max-representative-chars", type=int, default=480)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def representative_chunk_map(
    representative_docs: pd.DataFrame,
    document_topics: pd.DataFrame,
    max_chunks: int,
    max_chars: int,
) -> dict[int, list[str]]:
    records: dict[int, list[str]] = {}
    if not representative_docs.empty:
        reps = representative_docs.copy()
        reps["micro_topic_id"] = pd.to_numeric(reps["micro_topic_id"], errors="coerce")
        reps = reps.dropna(subset=["micro_topic_id"]).copy()
        reps["micro_topic_id"] = reps["micro_topic_id"].astype(int)
        reps = reps.loc[reps["micro_topic_id"] != -1].copy()
        reps["_order"] = range(len(reps))
        reps = reps.drop_duplicates(subset=["micro_topic_id", "chunk_id"], keep="first")
        for micro_topic_id, group in reps.groupby("micro_topic_id", dropna=False):
            chunks = [
                truncate_text(text, max_chars)
                for text in group.sort_values("_order")["text"].fillna("").astype(str).tolist()
                if str(text).strip()
            ][:max_chunks]
            if chunks:
                records[int(micro_topic_id)] = chunks

    if records:
        return records

    fallback = document_topics.loc[document_topics["micro_topic_id"] != -1].copy()
    fallback = fallback.sort_values(
        ["micro_topic_id", "micro_topic_probability", "chunk_id"],
        ascending=[True, False, True],
    )
    for micro_topic_id, group in fallback.groupby("micro_topic_id", dropna=False):
        chunks = [
            truncate_text(text, max_chars)
            for text in group["text"].fillna("").astype(str).tolist()
            if str(text).strip()
        ][:max_chunks]
        if chunks:
            records[int(micro_topic_id)] = chunks
    return records


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    ensure_directory(args.output_root)

    profile_rows: list[dict] = []
    annual_rows: list[dict] = []
    denominator_rows: list[dict] = []

    subgroup_dirs = list_subgroup_dirs(args.micro_root)
    for subgroup_dir in subgroup_dirs:
        manifest = read_json(subgroup_dir / "manifest.json")
        document_topics = pd.read_parquet(subgroup_dir / "document_topics.parquet")
        topic_info = pd.read_csv(subgroup_dir / "topic_info.csv")
        representative_docs = pd.read_csv(subgroup_dir / "representative_docs.csv")

        document_topics["micro_topic_id"] = pd.to_numeric(
            document_topics["micro_topic_id"], errors="coerce"
        )
        document_topics["year"] = pd.to_numeric(document_topics["year"], errors="coerce")
        document_topics = document_topics.dropna(subset=["micro_topic_id", "year"]).copy()
        document_topics["micro_topic_id"] = document_topics["micro_topic_id"].astype(int)
        document_topics["year"] = document_topics["year"].astype(int)

        valid_topics = document_topics.loc[document_topics["micro_topic_id"] != -1].copy()
        if valid_topics.empty:
            continue

        topic_info = topic_info.copy()
        topic_info["Topic"] = pd.to_numeric(topic_info["Topic"], errors="coerce")
        topic_info = topic_info.dropna(subset=["Topic"]).copy()
        topic_info["Topic"] = topic_info["Topic"].astype(int)
        topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()
        topic_meta = topic_info.set_index("Topic").to_dict(orient="index")

        macro_topic_name = subgroup_macro_topic_name(valid_topics, manifest)
        representative_map = representative_chunk_map(
            representative_docs=representative_docs,
            document_topics=valid_topics,
            max_chunks=args.max_representative_chunks,
            max_chars=args.max_representative_chars,
        )

        denominators = (
            valid_topics.groupby(["source", "assigned_label", "year"], dropna=False)
            .agg(
                yes_doc_n=("source_doc_id", "nunique"),
                yes_chunk_n=("chunk_id", "size"),
            )
            .reset_index()
        )
        denominator_rows.extend(denominators.to_dict(orient="records"))

        annual_counts = (
            valid_topics.groupby(["source", "assigned_label", "micro_topic_id", "year"], dropna=False)
            .agg(
                microtopic_doc_n=("source_doc_id", "nunique"),
                microtopic_chunk_n=("chunk_id", "size"),
            )
            .reset_index()
        )
        annual_counts["subgroup"] = subgroup_dir.name
        annual_counts["macro_topic_name"] = macro_topic_name
        annual_rows.extend(annual_counts.to_dict(orient="records"))

        for micro_topic_id, group in valid_topics.groupby("micro_topic_id", dropna=False):
            year_counts = (
                group.groupby("year", dropna=False)
                .agg(doc_n=("source_doc_id", "nunique"), chunk_n=("chunk_id", "size"))
                .reset_index()
                .sort_values("year")
            )
            years_present = year_counts["year"].astype(int).tolist()
            topic_record = topic_meta.get(int(micro_topic_id), {})
            microtopic_label = str(topic_record.get("Name", f"{micro_topic_id}")).strip()
            top_terms = parse_topic_terms(topic_record.get("Representation"))
            representative_chunks = representative_map.get(int(micro_topic_id), [])
            profile_rows.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": str(group["source"].iloc[0]),
                    "assigned_label": str(group["assigned_label"].iloc[0]),
                    "macro_topic_name": macro_topic_name,
                    "microtopic_id": int(micro_topic_id),
                    "microtopic_label": microtopic_label,
                    "microtopic_label_clean": clean_microtopic_label(microtopic_label),
                    "top_terms": json_dumps(top_terms),
                    "representative_chunks": json_dumps(representative_chunks),
                    "document_count": int(group["source_doc_id"].nunique()),
                    "chunk_count": int(group.shape[0]),
                    "years_present": json_dumps(years_present),
                    "active_year_count": int(len(years_present)),
                    "first_year": int(min(years_present)),
                    "last_year": int(max(years_present)),
                    "year_doc_counts": json_dumps(
                        {str(int(row.year)): int(row.doc_n) for row in year_counts.itertuples(index=False)}
                    ),
                    "year_chunk_counts": json_dumps(
                        {str(int(row.year)): int(row.chunk_n) for row in year_counts.itertuples(index=False)}
                    ),
                    "profile_text": build_profile_text(
                        microtopic_label=clean_microtopic_label(microtopic_label) or microtopic_label,
                        top_terms=top_terms,
                        representative_chunks=representative_chunks,
                    ),
                }
            )

    profiles = pd.DataFrame(profile_rows)
    annual = pd.DataFrame(annual_rows)
    denominators = pd.DataFrame(denominator_rows).drop_duplicates(
        subset=["source", "assigned_label", "year"]
    )

    if not profiles.empty:
        profiles["source"] = pd.Categorical(profiles["source"], categories=SOURCE_ORDER, ordered=True)
        profiles["assigned_label"] = pd.Categorical(
            profiles["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
        )
        profiles = profiles.sort_values(["source", "assigned_label", "microtopic_id"]).reset_index(drop=True)

    if not annual.empty:
        annual["source"] = pd.Categorical(annual["source"], categories=SOURCE_ORDER, ordered=True)
        annual["assigned_label"] = pd.Categorical(
            annual["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
        )
        annual = annual.sort_values(["source", "assigned_label", "micro_topic_id", "year"]).reset_index(drop=True)

    if not denominators.empty:
        denominators["source"] = pd.Categorical(
            denominators["source"], categories=SOURCE_ORDER, ordered=True
        )
        denominators["assigned_label"] = pd.Categorical(
            denominators["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
        )
        denominators = denominators.sort_values(["source", "assigned_label", "year"]).reset_index(drop=True)

    profiles_path = args.output_root / "microtopic_profiles.csv"
    profiles.to_csv(profiles_path, index=False)
    jsonl_path = args.output_root / "microtopic_profiles.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in profiles.to_dict(orient="records"):
            handle.write(json_dumps(row) + "\n")

    annual_path = args.output_root / "microtopic_annual_counts.csv"
    annual.to_csv(annual_path, index=False)
    denominator_path = args.output_root / "source_topic_year_denominators.csv"
    denominators.to_csv(denominator_path, index=False)

    write_json(
        args.output_root / "profile_manifest.json",
        {
            "micro_root": str(args.micro_root),
            "subgroup_count": int(len(subgroup_dirs)),
            "profile_row_count": int(profiles.shape[0]),
            "annual_count_row_count": int(annual.shape[0]),
            "denominator_row_count": int(denominators.shape[0]),
            "max_representative_chunks": args.max_representative_chunks,
            "max_representative_chars": args.max_representative_chars,
            "outputs": {
                "microtopic_profiles_csv": str(profiles_path),
                "microtopic_profiles_jsonl": str(jsonl_path),
                "microtopic_annual_counts_csv": str(annual_path),
                "source_topic_year_denominators_csv": str(denominator_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
