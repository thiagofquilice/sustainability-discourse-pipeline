#!/usr/bin/env python3
"""Build source-by-stage filter counts for the corporate-focus paper tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DATASET_ROOT = Path("data/external")
FILTERED_CORPUS_PATH = DATASET_ROOT / "filtered_corpus.parquet"
SUBSET_SUMMARY_PATH = DATASET_ROOT / "subset_summary.json"
MERGED_ROOT = PIPELINE_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
SELECTED_TOPICS_PATH = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_stage12_colab_drive_with_overrides"
    / "data"
    / "selected_micro_topics.csv"
)
REVIEW_DECISIONS_PATH = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_review_with_overrides"
    / "corporate_focus_commented_decision_rows.csv"
)
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_filter_stage_counts"
OUTPUT_CSV_PATH = OUTPUT_DIR / "filter_stage_source_counts.csv"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "filter_stage_source_counts_manifest.json"

SOURCE_ORDER = ["academic", "media", "corporate"]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_review_decision(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text == "delete":
        return "delete"
    if text == "not related":
        return "not related"
    return ""


def detect_document_id_column(frame: pd.DataFrame) -> str:
    for name in ["source_doc_id", "doc_id", "document_id"]:
        if name in frame.columns:
            return name
    raise KeyError("Could not find a stable document identifier column.")


def load_raw_counts() -> tuple[dict[str, int], dict[str, int], str, dict[str, object]]:
    subset_summary = json.loads(SUBSET_SUMMARY_PATH.read_text(encoding="utf-8"))
    raw_frame = pd.read_parquet(FILTERED_CORPUS_PATH, columns=["source", "source_doc_id"])
    unique_docs = (
        raw_frame.groupby("source")["source_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    chunk_counts = {
        source: int(subset_summary["subset_source_distribution"].get(source, 0))
        for source in SOURCE_ORDER
    }
    validation = {
        "subset_type": subset_summary.get("subset_type"),
        "summary_source_doc_count": int(subset_summary.get("source_doc_count", 0)),
        "parquet_source_doc_count": int(sum(unique_docs.values())),
        "summary_chunk_total": int(subset_summary.get("subset_rows", 0)),
        "summary_chunk_counts": chunk_counts,
    }
    return unique_docs, chunk_counts, "source_doc_id", validation


def load_reviewed_document_topics() -> tuple[dict[str, pd.DataFrame], str]:
    frames: dict[str, pd.DataFrame] = {}
    document_id_column_used = ""
    for subgroup_dir in sorted(MERGED_ROOT.iterdir()):
        if not subgroup_dir.is_dir():
            continue
        document_topics_path = subgroup_dir / "document_topics.csv"
        if not document_topics_path.exists():
            continue
        sample = pd.read_csv(document_topics_path, nrows=5)
        doc_col = detect_document_id_column(sample)
        document_id_column_used = doc_col
        columns = [col for col in [doc_col, "source", "micro_topic_id", "final_merge_group_id"] if col in sample.columns]
        frame = pd.read_csv(document_topics_path, usecols=columns)
        frame = frame.rename(columns={doc_col: "stable_doc_id"})
        frame["subgroup"] = subgroup_dir.name
        frames[subgroup_dir.name] = frame
    if not frames:
        raise FileNotFoundError(f"No reviewed document_topics.csv files found under {MERGED_ROOT}")
    return frames, document_id_column_used


def build_reviewed_counts(
    document_topics_by_subgroup: dict[str, pd.DataFrame],
) -> tuple[dict[str, int], dict[str, int]]:
    all_topics = pd.concat(document_topics_by_subgroup.values(), ignore_index=True)

    six_topic_counts = (
        all_topics.groupby("source")["stable_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )

    if "micro_topic_id" not in all_topics.columns:
        raise KeyError("Expected micro_topic_id in reviewed document_topics.csv files.")
    microtopic_topics = all_topics.loc[all_topics["micro_topic_id"] != -1].copy()
    microtopic_counts = (
        microtopic_topics.groupby("source")["stable_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    return six_topic_counts, microtopic_counts


def load_delete_group_ids() -> set[str]:
    review_frame = pd.read_csv(REVIEW_DECISIONS_PATH)
    decision_column = (
        "_normalized_review_decision"
        if "_normalized_review_decision" in review_frame.columns
        else "review_decision"
    )
    review_frame["_normalized_review_decision"] = review_frame[decision_column].map(normalize_review_decision)
    return set(
        review_frame.loc[
            review_frame["_normalized_review_decision"] == "delete",
            "final_merge_group_id_noncorporate",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )


def build_selected_subset_counts(
    document_topics_by_subgroup: dict[str, pd.DataFrame],
) -> tuple[dict[str, int], dict[str, int]]:
    selected_topics = pd.read_csv(SELECTED_TOPICS_PATH)
    selected_groups = selected_topics[["subgroup", "source", "final_merge_group_id"]].drop_duplicates().copy()
    selected_groups["final_merge_group_id"] = selected_groups["final_merge_group_id"].astype(str)
    delete_group_ids = load_delete_group_ids()

    pre_frames: list[pd.DataFrame] = []
    post_frames: list[pd.DataFrame] = []

    for subgroup, subgroup_groups in selected_groups.groupby("subgroup", sort=True):
        if subgroup not in document_topics_by_subgroup:
            raise FileNotFoundError(f"Missing reviewed document_topics for subgroup {subgroup}")
        subgroup_topics = document_topics_by_subgroup[subgroup].copy()
        subgroup_topics["final_merge_group_id"] = subgroup_topics["final_merge_group_id"].astype(str)

        allowed_groups = set(subgroup_groups["final_merge_group_id"].tolist())
        pre_frame = subgroup_topics.loc[subgroup_topics["final_merge_group_id"].isin(allowed_groups)].copy()
        pre_frames.append(pre_frame)

        if str(subgroup_groups["source"].iloc[0]) == "corporate":
            kept_groups = allowed_groups
        else:
            kept_groups = {group_id for group_id in allowed_groups if group_id not in delete_group_ids}
        post_frame = subgroup_topics.loc[subgroup_topics["final_merge_group_id"].isin(kept_groups)].copy()
        post_frames.append(post_frame)

    pre_all = pd.concat(pre_frames, ignore_index=True)
    post_all = pd.concat(post_frames, ignore_index=True)

    pre_counts = (
        pre_all.groupby("source")["stable_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    post_counts = (
        post_all.groupby("source")["stable_doc_id"]
        .nunique()
        .reindex(SOURCE_ORDER, fill_value=0)
        .astype(int)
        .to_dict()
    )
    return pre_counts, post_counts


def make_row(
    filter_stage: str,
    counts: dict[str, int],
    unit: str,
    identifier_used: str,
    note: str,
) -> dict[str, object]:
    academic = int(counts.get("academic", 0))
    media = int(counts.get("media", 0))
    corporate = int(counts.get("corporate", 0))
    return {
        "filter_stage": filter_stage,
        "academic": academic,
        "media": media,
        "corporate": corporate,
        "total": academic + media + corporate,
        "unit": unit,
        "identifier_used": identifier_used,
        "note": note,
    }


def main() -> None:
    ensure_directory(OUTPUT_DIR)

    raw_unique_counts, raw_chunk_counts, raw_identifier, raw_validation = load_raw_counts()
    document_topics_by_subgroup, reviewed_identifier = load_reviewed_document_topics()
    reviewed_six_topic_counts, reviewed_microtopic_counts = build_reviewed_counts(document_topics_by_subgroup)
    pre_review_counts, post_review_counts = build_selected_subset_counts(document_topics_by_subgroup)

    rows = [
        make_row(
            filter_stage="Raw collected corpus",
            counts=raw_unique_counts,
            unit="unique_documents",
            identifier_used=raw_identifier,
            note="Raw three-source subset after industry filtering; stable document identifier from filtered_corpus.parquet.",
        ),
        make_row(
            filter_stage="Raw collected corpus",
            counts=raw_chunk_counts,
            unit="chunks",
            identifier_used="chunk rows from subset_summary.json",
            note="Chunk row counts, not documents. Values must match subset_source_distribution in subset_summary.json.",
        ),
        make_row(
            filter_stage="Reviewed six-topic corpus",
            counts=reviewed_six_topic_counts,
            unit="unique_documents",
            identifier_used=reviewed_identifier,
            note="Reviewed merged corpus. Uses source_doc_id as the stable document key; media doc_id remains row-level here. Six-topic and non-outlier microtopic counts coincide, so the separate microtopic row is omitted.",
        ),
        make_row(
            filter_stage="Selected corporate-focus subset, pre-final-review",
            counts=pre_review_counts,
            unit="unique_documents",
            identifier_used="deduplicated source_doc_id across selected final_merge_group_id values",
            note="Corporate-focus selected subset before comment-based review decisions. Deduplicated globally across selected groups.",
        ),
        make_row(
            filter_stage="Selected corporate-focus subset, post-final-review",
            counts=post_review_counts,
            unit="unique_documents",
            identifier_used="deduplicated source_doc_id after removing delete groups",
            note="Final paper subset after removing non-corporate delete cases. not related topics are retained.",
        ),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV_PATH, index=False)

    validations = {
        "raw_chunk_counts_match_subset_summary": raw_chunk_counts == raw_validation["summary_chunk_counts"],
        "raw_unique_document_total_matches_subset_summary": int(sum(raw_unique_counts.values()))
        == int(raw_validation["summary_source_doc_count"]),
        "reviewed_six_topic_equals_microtopic_counts": reviewed_six_topic_counts == reviewed_microtopic_counts,
        "post_review_not_greater_than_pre_review": all(
            int(post_review_counts[source]) <= int(pre_review_counts[source]) for source in SOURCE_ORDER
        ),
        "corporate_count_unchanged_after_review": int(post_review_counts["corporate"]) == int(pre_review_counts["corporate"]),
    }

    manifest = {
        "output_csv": str(OUTPUT_CSV_PATH),
        "raw_corpus": {
            "filtered_corpus_path": str(FILTERED_CORPUS_PATH),
            "subset_summary_path": str(SUBSET_SUMMARY_PATH),
            "identifier_used": raw_identifier,
            "counts": raw_unique_counts,
            "chunk_counts": raw_chunk_counts,
            "summary_chunk_total": raw_validation["summary_chunk_total"],
            "summary_source_doc_count": raw_validation["summary_source_doc_count"],
        },
        "reviewed_six_topic_corpus": {
            "merged_root": str(MERGED_ROOT),
            "identifier_used": reviewed_identifier,
            "counts": reviewed_six_topic_counts,
            "microtopic_counts": reviewed_microtopic_counts,
        },
        "selected_corporate_focus_subset": {
            "selected_topics_path": str(SELECTED_TOPICS_PATH),
            "review_decisions_path": str(REVIEW_DECISIONS_PATH),
            "pre_final_review_counts": pre_review_counts,
            "post_final_review_counts": post_review_counts,
        },
        "validations": validations,
    }
    OUTPUT_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(result.to_string(index=False))
    print("\nVALIDATIONS")
    print(json.dumps(validations, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
