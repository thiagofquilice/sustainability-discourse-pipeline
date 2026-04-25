#!/usr/bin/env python3
"""Build public-safe data files for the Streamlit results explorer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
MACRO_TOPIC_NAMES = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution and environmental stewardship",
}
DENOMINATOR_COLUMNS = {
    "academic": "academic_document_count",
    "media": "media_document_count",
    "corporate": "corporate_document_count",
}
YEARS = list(range(2000, 2026))
DOCUMENT_ID_COLUMNS = ["source_doc_id", "document_id", "doc_id"]
TEXT_COLUMNS = [f"chunk_text_{idx}" for idx in range(1, 6)]
CHUNK_ID_COLUMNS = [f"chunk_id_{idx}" for idx in range(1, 6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snippet-chars", type=int, default=360)
    public_group = parser.add_mutually_exclusive_group()
    public_group.add_argument("--public-safe", dest="public_safe", action="store_true", default=True)
    public_group.add_argument("--local-full-text", dest="public_safe", action="store_false")
    return parser.parse_args()


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(frame: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    frame.to_csv(path, index=False)
    return path


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_json(value: object, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def make_topic_id(subgroup: object, group_id: object) -> str:
    return f"{subgroup}::{group_id}"


def detect_document_id_column(columns: pd.Index) -> str:
    for column in DOCUMENT_ID_COLUMNS:
        if column in columns:
            return column
    raise KeyError("Could not find source_doc_id, document_id, or doc_id in document_topics.csv")


def macro_topic_name(macro_topic: object, fallback: object = "") -> str:
    fallback_text = clean_text(fallback)
    return fallback_text or MACRO_TOPIC_NAMES.get(str(macro_topic), str(macro_topic))


def source_link(source: str, source_doc_id: str) -> str:
    if not source_doc_id:
        return ""
    if source == "media":
        return f"https://www.theguardian.com/{source_doc_id}"
    if source == "academic":
        return f"https://www.semanticscholar.org/paper/{source_doc_id}"
    return ""


def snippet_text(text: str, public_safe: bool, limit: int) -> tuple[str, bool]:
    text = clean_text(text)
    if not public_safe or len(text) <= limit:
        return text, False
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{clipped} ...", True


def load_topic_status(relative_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    topic_tables_root = relative_root / "topic_tables"
    for macro_topic in MACRO_TOPIC_ORDER:
        for status_name, file_suffix in [
            ("aligned", "aligned_aggregate_topics"),
            ("external_relevant_unpaired", "unpaired_external_topics"),
            ("excluded", "excluded_topics"),
        ]:
            path = topic_tables_root / f"{macro_topic}_{file_suffix}.csv"
            if not path.exists():
                continue
            frame = read_csv(path)
            if frame.empty:
                continue
            subset = frame[
                [
                    "macro_topic",
                    "source_noncorporate",
                    "subgroup_noncorporate",
                    "final_merge_group_id_noncorporate",
                ]
            ].copy()
            subset = subset.rename(
                columns={
                    "source_noncorporate": "source",
                    "subgroup_noncorporate": "subgroup",
                    "final_merge_group_id_noncorporate": "final_merge_group_id",
                }
            )
            subset["paper_status"] = status_name
            frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=["macro_topic", "source", "subgroup", "final_merge_group_id", "paper_status"])
    statuses = pd.concat(frames, ignore_index=True)
    statuses["final_merge_group_id"] = statuses["final_merge_group_id"].astype(str)
    return statuses.drop_duplicates(["subgroup", "final_merge_group_id"], keep="first")


def build_topics(pipeline_root: Path) -> pd.DataFrame:
    review_root = pipeline_root / "outputs" / "corporate_focus_review_with_overrides"
    relative_root = pipeline_root / "outputs" / "paper_tables" / "corporate_focus_relative_longitudinal_series"
    groups = read_csv(review_root / "corporate_focus_all_groups_optional.csv")
    statuses = load_topic_status(relative_root)

    groups["final_merge_group_id"] = groups["final_merge_group_id"].astype(str)
    topics = groups.merge(
        statuses[["subgroup", "final_merge_group_id", "paper_status"]],
        on=["subgroup", "final_merge_group_id"],
        how="left",
    )
    topics.loc[topics["source"] == "corporate", "paper_status"] = "corporate_anchor"
    topics["paper_status"] = topics["paper_status"].fillna("reviewed_external")
    topics["topic_id"] = [make_topic_id(row.subgroup, row.final_merge_group_id) for row in topics.itertuples(index=False)]
    topics["macro_topic_name"] = [
        macro_topic_name(topic, name)
        for topic, name in zip(topics["macro_topic"], topics.get("macro_topic_name", ""))
    ]
    topics["display_label"] = topics["topic_label_refined"].fillna("").map(clean_text)
    fallback = topics["topic_name_original"].fillna("").map(clean_text)
    topics.loc[topics["display_label"] == "", "display_label"] = fallback
    topics["display_label"] = topics["display_label"].fillna(topics["topic_id"])

    columns = [
        "topic_id",
        "macro_topic",
        "macro_topic_name",
        "source",
        "subgroup",
        "final_merge_group_id",
        "micro_topic_id",
        "display_label",
        "topic_name_original",
        "topic_label_refined",
        "overall_summary",
        "phase_1_years",
        "phase_2_years",
        "phase_3_years",
        "phase_4_years",
        "phase_1_summary",
        "phase_2_summary",
        "phase_3_summary",
        "phase_4_summary",
        "evolution_pattern",
        "evidence_note",
        "paper_status",
        "selection_bucket",
        "noncorporate_inclusion_reason",
        "best_matched_corporate_group_id",
        "best_dyad",
        "topic_size",
        "group_size",
        "active_year_count",
        "active_year_min",
        "active_year_max",
        "overall_keywords",
    ]
    for column in columns:
        if column not in topics.columns:
            topics[column] = ""
    return topics[columns].sort_values(["macro_topic", "source", "paper_status", "display_label"], kind="mergesort")


def load_selected_key(pipeline_root: Path) -> pd.DataFrame:
    selected_path = pipeline_root / "outputs" / "corporate_focus_stage12_colab_drive_with_overrides" / "data" / "selected_micro_topics.csv"
    selected = read_csv(
        selected_path,
        usecols=["subgroup", "source", "assigned_label", "micro_topic_id", "final_merge_group_id"],
    )
    selected["final_merge_group_id"] = selected["final_merge_group_id"].astype(str)
    return selected.drop_duplicates()


def build_year_series(pipeline_root: Path, topics: pd.DataFrame) -> pd.DataFrame:
    merged_root = pipeline_root / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
    topic_key = topics[
        ["topic_id", "macro_topic", "source", "subgroup", "final_merge_group_id"]
    ].drop_duplicates().copy()
    topic_key["final_merge_group_id"] = topic_key["final_merge_group_id"].astype(str)
    output_rows: list[dict[str, object]] = []

    for subgroup, requested in topic_key.groupby("subgroup", sort=True):
        path = merged_root / str(subgroup) / "document_topics.csv"
        header = pd.read_csv(path, nrows=0)
        doc_column = detect_document_id_column(header.columns)
        doc_topics = pd.read_csv(
            path,
            usecols=[doc_column, "assigned_label", "source", "year", "final_merge_group_id"],
        )
        doc_topics = doc_topics.rename(columns={doc_column: "stable_doc_id"})
        doc_topics = doc_topics.dropna(
            subset=["stable_doc_id", "assigned_label", "source", "year", "final_merge_group_id"]
        ).copy()
        doc_topics["stable_doc_id"] = doc_topics["stable_doc_id"].astype(str)
        doc_topics["final_merge_group_id"] = doc_topics["final_merge_group_id"].astype(str)
        doc_topics["year"] = pd.to_numeric(doc_topics["year"], errors="coerce")
        doc_topics = doc_topics.dropna(subset=["year"]).copy()
        doc_topics["year"] = doc_topics["year"].astype(int)

        annual_denominator = (
            doc_topics.groupby(["assigned_label", "source", "year"], sort=False)["stable_doc_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        period_denominator = (
            doc_topics.groupby(["assigned_label", "source"], sort=False)["stable_doc_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        topic_year_counts = (
            doc_topics.groupby(["final_merge_group_id", "assigned_label", "source", "year"], sort=False)[
                "stable_doc_id"
            ]
            .nunique()
            .astype(int)
            .to_dict()
        )

        for topic in requested.itertuples(index=False):
            macro_topic = str(topic.macro_topic)
            source = str(topic.source)
            group_id = str(topic.final_merge_group_id)
            period_docs = int(period_denominator.get((macro_topic, source), 0))
            for year in YEARS:
                docs = int(topic_year_counts.get((group_id, macro_topic, source, year), 0))
                annual_docs = int(annual_denominator.get((macro_topic, source, year), 0))
                output_rows.append(
                    {
                        "topic_id": topic.topic_id,
                        "macro_topic": macro_topic,
                        "source": source,
                        "year": int(year),
                        "year_frequency": docs,
                        "year_document_count": docs,
                        "denominator_source_docs_in_macro_topic": period_docs,
                        "relative_share_of_macro_topic_source_docs": float(docs) / float(period_docs)
                        if period_docs
                        else 0.0,
                        "annual_source_macro_document_count": annual_docs,
                        "annual_document_prevalence": float(docs) / float(annual_docs) if annual_docs else 0.0,
                    }
                )

    return pd.DataFrame(output_rows).sort_values(["topic_id", "year"], kind="mergesort")


def collect_words(rows: pd.DataFrame, limit: int = 14) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for value in rows["year_specific_words"].tolist():
        for word in parse_json(value, []):
            text = clean_text(word)
            key = text.lower()
            if text and key not in seen:
                words.append(text)
                seen.add(key)
            if len(words) >= limit:
                return words
    return words


def docs_from_chunk_records(row: pd.Series, public_safe: bool, snippet_chars: int) -> list[dict[str, object]]:
    records = parse_json(row.get("chunk_records_json", ""), [])
    docs: list[dict[str, object]] = []
    if records:
        for record in records:
            source = clean_text(row.get("source", ""))
            source_doc_id = clean_text(record.get("source_doc_id", ""))
            text, truncated = snippet_text(str(record.get("text", "")), public_safe, snippet_chars)
            if not text:
                continue
            docs.append(
                {
                    "year": int(record.get("year", row.get("year", 0))),
                    "source": source,
                    "chunk_id": clean_text(record.get("chunk_id", "")),
                    "source_doc_id": source_doc_id,
                    "document_id": source_doc_id,
                    "source_link": source_link(source, source_doc_id),
                    "micro_topic_probability": record.get("micro_topic_probability", ""),
                    "snippet": text,
                    "is_truncated": bool(truncated),
                }
            )
    if docs:
        return docs

    for idx, text_col in enumerate(TEXT_COLUMNS, start=1):
        raw_text = row.get(text_col, "")
        text, truncated = snippet_text(str(raw_text), public_safe, snippet_chars)
        if not text:
            continue
        chunk_id = clean_text(row.get(f"chunk_id_{idx}", ""))
        docs.append(
            {
                "year": int(row.get("year", 0)),
                "source": clean_text(row.get("source", "")),
                "chunk_id": chunk_id,
                "source_doc_id": "",
                "document_id": "",
                "source_link": "",
                "micro_topic_probability": "",
                "snippet": text,
                "is_truncated": bool(truncated),
            }
        )
    return docs


def collect_docs(rows: pd.DataFrame, public_safe: bool, snippet_chars: int, limit: int = 3) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        for doc in docs_from_chunk_records(row, public_safe, snippet_chars):
            key = (str(doc.get("chunk_id", "")), str(doc.get("snippet", ""))[:80])
            if key in seen:
                continue
            docs.append(doc)
            seen.add(key)
            if len(docs) >= limit:
                return docs
    return docs


def build_year_evidence(pipeline_root: Path, topics: pd.DataFrame, public_safe: bool, snippet_chars: int) -> pd.DataFrame:
    evidence_path = pipeline_root / "outputs" / "corporate_focus_stage12_colab_drive_with_overrides" / "data" / "micro_topic_year_evidence.csv"
    selected = load_selected_key(pipeline_root)
    topic_ids = set(topics["topic_id"])
    output_rows: list[dict[str, object]] = []
    usecols = [
        "subgroup",
        "source",
        "assigned_label",
        "macro_topic_name",
        "micro_topic_id",
        "topic_name_original",
        "year",
        "year_frequency",
        "year_specific_words",
        "overall_keywords",
        "chunk_records_json",
        *CHUNK_ID_COLUMNS,
        *TEXT_COLUMNS,
    ]

    for evidence in pd.read_csv(evidence_path, usecols=usecols, chunksize=50000):
        merged = evidence.merge(
            selected,
            on=["subgroup", "source", "assigned_label", "micro_topic_id"],
            how="inner",
        )
        merged["topic_id"] = [make_topic_id(row.subgroup, row.final_merge_group_id) for row in merged.itertuples(index=False)]
        merged = merged[merged["topic_id"].isin(topic_ids)]
        if merged.empty:
            continue
        for (topic_id, year), group in merged.groupby(["topic_id", "year"], sort=False):
            output_rows.append(
                {
                    "topic_id": topic_id,
                    "year": int(year),
                    "top_words_json": json.dumps(collect_words(group), ensure_ascii=False),
                    "representative_docs_json": json.dumps(
                        collect_docs(group, public_safe, snippet_chars),
                        ensure_ascii=False,
                    ),
                    "public_safe": bool(public_safe),
                }
            )
    return pd.DataFrame(output_rows).sort_values(["topic_id", "year"], kind="mergesort")


def build_relations(pipeline_root: Path) -> pd.DataFrame:
    relative_root = pipeline_root / "outputs" / "paper_tables" / "corporate_focus_relative_longitudinal_series"
    frames: list[pd.DataFrame] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        path = relative_root / "topic_tables" / f"{macro_topic}_aligned_aggregate_topics.csv"
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    relations = pd.concat(frames, ignore_index=True)
    relations["corporate_topic_id"] = [
        make_topic_id(row.subgroup_corporate, row.final_merge_group_id_corporate)
        for row in relations.itertuples(index=False)
    ]
    relations["external_topic_id"] = [
        make_topic_id(row.subgroup_noncorporate, row.final_merge_group_id_noncorporate)
        for row in relations.itertuples(index=False)
    ]
    columns = [
        "macro_topic",
        "macro_topic_name",
        "corporate_topic_id",
        "subgroup_corporate",
        "final_merge_group_id_corporate",
        "topic_label_refined_corporate",
        "corporate_unique_document_count",
        "external_topic_id",
        "source_noncorporate",
        "subgroup_noncorporate",
        "final_merge_group_id_noncorporate",
        "topic_label_refined_noncorporate",
        "overall_summary_noncorporate",
        "external_unique_document_count",
        "best_cosine_similarity",
        "best_pair_id",
        "direct_pair_count",
    ]
    for column in columns:
        if column not in relations.columns:
            relations[column] = ""
    return relations[columns].drop_duplicates()


def build_panel_series(pipeline_root: Path) -> pd.DataFrame:
    path = (
        pipeline_root
        / "outputs"
        / "paper_tables"
        / "corporate_focus_relative_longitudinal_series"
        / "corporate_topic_external_relative_longitudinal.csv"
    )
    series = read_csv(path)
    series["corporate_topic_id"] = [
        make_topic_id(row.corporate_subgroup, row.corporate_final_merge_group_id)
        for row in series.itertuples(index=False)
    ]
    return series


def build_unpaired_series(pipeline_root: Path) -> pd.DataFrame:
    path = (
        pipeline_root
        / "outputs"
        / "paper_tables"
        / "corporate_focus_relative_longitudinal_series"
        / "unpaired_external_relative_longitudinal.csv"
    )
    series = read_csv(path)
    series["topic_id"] = [make_topic_id(row.subgroup, row.final_merge_group_id) for row in series.itertuples(index=False)]
    return series


def build_source_relation_tables(pipeline_root: Path) -> dict[str, pd.DataFrame]:
    relation_root = pipeline_root / "outputs" / "paper_tables" / "corporate_focus_source_topic_relations"
    files = {
        "source_relation_aggregate_summary": "aggregate_source_relations_summary.csv",
        "source_relation_individual_summary": "individual_source_relations_summary.csv",
        "source_relation_aggregate_lag_details": "aggregate_source_relations_lag_details.csv",
        "source_relation_individual_lag_details": "individual_source_relations_lag_details.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = relation_root / filename
        tables[key] = read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def parse_interpretations(markdown_path: Path) -> dict[str, Any]:
    if not markdown_path.exists():
        return {"anchors": {}, "unpaired": {}}
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    current_macro = ""
    current_anchor = ""
    in_unpaired = False
    anchors: dict[str, str] = {}
    unpaired: dict[str, str] = {}
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_anchor, current_macro, in_unpaired
        text = "\n".join(line for line in buffer).strip()
        if not text:
            buffer = []
            return
        if in_unpaired and current_macro:
            unpaired[current_macro] = text
        elif current_anchor:
            anchors[current_anchor] = text
        buffer = []

    for line in lines:
        if line.startswith("## T"):
            flush()
            current_macro = line.split(" ", 2)[1].strip()
            current_anchor = ""
            in_unpaired = False
            continue
        if line.startswith("### Unpaired"):
            flush()
            current_anchor = ""
            in_unpaired = True
            continue
        if line.startswith("### "):
            flush()
            in_unpaired = False
            continue
        if line.startswith("#### "):
            flush()
            current_anchor = line.replace("#### ", "", 1).strip()
            in_unpaired = False
            continue
        if current_anchor or in_unpaired:
            buffer.append(line)
    flush()
    return {"anchors": anchors, "unpaired": unpaired}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topics = build_topics(args.pipeline_root)
    year_series = build_year_series(args.pipeline_root, topics)
    year_evidence = build_year_evidence(args.pipeline_root, topics, args.public_safe, args.snippet_chars)
    relations = build_relations(args.pipeline_root)
    panel_series = build_panel_series(args.pipeline_root)
    unpaired_series = build_unpaired_series(args.pipeline_root)
    source_relation_tables = build_source_relation_tables(args.pipeline_root)
    interpretations = parse_interpretations(
        args.pipeline_root
        / "outputs"
        / "paper_tables"
        / "corporate_focus_relative_longitudinal_series"
        / "longitudinal_panel_interpretations.md"
    )

    outputs = {
        "topics": str(write_csv(topics, args.output_dir, "topics.csv")),
        "year_series": str(write_csv(year_series, args.output_dir, "year_series.csv")),
        "year_evidence": str(write_csv(year_evidence, args.output_dir, "year_evidence.csv")),
        "relations": str(write_csv(relations, args.output_dir, "relations.csv")),
        "panel_series": str(write_csv(panel_series, args.output_dir, "panel_series.csv")),
        "unpaired_series": str(write_csv(unpaired_series, args.output_dir, "unpaired_series.csv")),
        "panel_interpretations": str(args.output_dir / "panel_interpretations.json"),
    }
    for key, frame in source_relation_tables.items():
        outputs[key] = str(write_csv(frame, args.output_dir, f"{key}.csv"))
    (args.output_dir / "panel_interpretations.json").write_text(
        json.dumps(interpretations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "public_safe": bool(args.public_safe),
        "snippet_chars": int(args.snippet_chars),
        "pipeline_root": str(args.pipeline_root),
        "outputs": outputs,
        "row_counts": {
            "topics": int(len(topics)),
            "year_series": int(len(year_series)),
            "year_evidence": int(len(year_evidence)),
            "relations": int(len(relations)),
            "panel_series": int(len(panel_series)),
            "unpaired_series": int(len(unpaired_series)),
            "source_relation_aggregate_summary": int(len(source_relation_tables["source_relation_aggregate_summary"])),
            "source_relation_individual_summary": int(len(source_relation_tables["source_relation_individual_summary"])),
            "source_relation_aggregate_lag_details": int(
                len(source_relation_tables["source_relation_aggregate_lag_details"])
            ),
            "source_relation_individual_lag_details": int(
                len(source_relation_tables["source_relation_individual_lag_details"])
            ),
            "anchor_interpretations": int(len(interpretations.get("anchors", {}))),
            "unpaired_interpretations": int(len(interpretations.get("unpaired", {}))),
        },
    }
    (args.output_dir / "app_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
