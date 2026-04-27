#!/usr/bin/env python3
"""Build macro-topic document counts for the corporate-focus selected set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_SELECTED_TOPICS = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_focus_stage12_colab_drive_with_overrides"
    / "data"
    / "selected_micro_topics.csv"
)
DEFAULT_MERGED_ROOT = PIPELINE_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
DEFAULT_REVIEW_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_review_with_overrides"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_document_counts"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
SOURCE_ORDER = ["academic", "media", "corporate"]
MACRO_TOPIC_NAME_MAP = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution and environmental stewardship",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topics", type=Path, default=DEFAULT_SELECTED_TOPICS)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_document_id_column(frame: pd.DataFrame) -> str:
    for name in ["doc_id", "document_id", "source_doc_id"]:
        if name in frame.columns:
            return name
    raise KeyError("Could not find a document identifier column in document_topics.csv")


def normalize_review_decision(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text == "delete":
        return "delete"
    if text == "not related":
        return "not related"
    return ""


def load_final_status_map(review_root: Path) -> dict[str, str]:
    decisions_path = review_root / "corporate_focus_commented_decision_rows.csv"
    if not decisions_path.exists():
        return {}
    decisions = pd.read_csv(decisions_path)
    decision_column = "_normalized_review_decision" if "_normalized_review_decision" in decisions.columns else "review_decision"
    decisions["_normalized_review_decision"] = decisions[decision_column].map(normalize_review_decision)
    decisions = decisions.dropna(subset=["final_merge_group_id_noncorporate"]).copy()
    decisions["final_merge_group_id_noncorporate"] = decisions["final_merge_group_id_noncorporate"].astype(str)
    return dict(
        decisions.loc[decisions["_normalized_review_decision"] != "", ["final_merge_group_id_noncorporate", "_normalized_review_decision"]]
        .drop_duplicates(subset=["final_merge_group_id_noncorporate"], keep="last")
        .itertuples(index=False, name=None)
    )


def assign_paper_status(selected_topics: pd.DataFrame, review_root: Path) -> pd.DataFrame:
    status_map = load_final_status_map(review_root)
    result = selected_topics.copy()

    def status_for_row(row: pd.Series) -> str:
        if str(row["source"]) == "corporate":
            return "corporate_kept"
        decision = status_map.get(str(row["final_merge_group_id"]), "")
        if decision == "delete":
            return "excluded"
        if decision == "not related":
            return "external_relevant_unpaired"
        return "aligned"

    result["paper_status"] = result.apply(status_for_row, axis=1)
    return result


def build_document_counts(selected_topics: pd.DataFrame, merged_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    macro_source_docs: dict[tuple[str, str], set[str]] = {}
    macro_source_status_docs: dict[tuple[str, str, str], set[str]] = {}
    document_id_column_used = ""

    for subgroup, subgroup_selected in selected_topics.groupby("subgroup", sort=True):
        document_topics_path = merged_root / subgroup / "document_topics.csv"
        if not document_topics_path.exists():
            raise FileNotFoundError(f"Missing document_topics.csv for subgroup {subgroup}: {document_topics_path}")

        document_topics = pd.read_csv(document_topics_path)
        doc_col = detect_document_id_column(document_topics)
        document_id_column_used = doc_col

        allowed_groups = set(subgroup_selected["final_merge_group_id"].astype(str))
        filtered = document_topics.loc[
            document_topics["final_merge_group_id"].astype(str).isin(allowed_groups)
        ].copy()

        filtered = filtered.merge(
            subgroup_selected[["final_merge_group_id", "assigned_label", "source", "paper_status"]],
            on="final_merge_group_id",
            how="inner",
            validate="many_to_one",
            suffixes=("", "_selected"),
        )

        if "assigned_label_selected" in filtered.columns:
            filtered["assigned_label"] = filtered["assigned_label_selected"]
        if "source_selected" in filtered.columns:
            filtered["source"] = filtered["source_selected"]

        for (macro_topic, source), group in filtered.groupby(["assigned_label", "source"], sort=False):
            macro_source_docs.setdefault((macro_topic, source), set()).update(
                group[doc_col].dropna().astype(str).tolist()
            )
        for (macro_topic, source, paper_status), group in filtered.groupby(
            ["assigned_label", "source", "paper_status"], sort=False
        ):
            macro_source_status_docs.setdefault((macro_topic, source, paper_status), set()).update(
                group[doc_col].dropna().astype(str).tolist()
            )

    rows_pre_review: list[dict[str, object]] = []
    rows_final_by_source: list[dict[str, object]] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        source_sets = {source: macro_source_docs.get((macro_topic, source), set()) for source in SOURCE_ORDER}
        total_docs = set().union(*source_sets.values()) if source_sets else set()
        rows_pre_review.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": MACRO_TOPIC_NAME_MAP[macro_topic],
                "academic_document_count": len(source_sets["academic"]),
                "media_document_count": len(source_sets["media"]),
                "corporate_document_count": len(source_sets["corporate"]),
                "total_unique_document_count": len(total_docs),
            }
        )

        academic_aligned = macro_source_status_docs.get((macro_topic, "academic", "aligned"), set())
        academic_not_related = macro_source_status_docs.get((macro_topic, "academic", "external_relevant_unpaired"), set())
        media_aligned = macro_source_status_docs.get((macro_topic, "media", "aligned"), set())
        media_not_related = macro_source_status_docs.get((macro_topic, "media", "external_relevant_unpaired"), set())
        corporate_kept = macro_source_status_docs.get((macro_topic, "corporate", "corporate_kept"), set())
        final_kept = set().union(academic_aligned, academic_not_related, media_aligned, media_not_related, corporate_kept)

        rows_final_by_source.append(
            {
                "macro_topic": macro_topic,
                "macro_topic_name": MACRO_TOPIC_NAME_MAP[macro_topic],
                "academic_document_count": len(academic_aligned.union(academic_not_related)),
                "media_document_count": len(media_aligned.union(media_not_related)),
                "corporate_document_count": len(corporate_kept),
                "total_unique_document_count": len(final_kept),
            }
        )

    return pd.DataFrame(rows_pre_review), pd.DataFrame(rows_final_by_source), document_id_column_used


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    selected_topics = pd.read_csv(args.selected_topics)
    selected_topics = assign_paper_status(selected_topics, args.review_root)
    counts, counts_final_by_source, doc_column = build_document_counts(selected_topics, args.merged_root)

    csv_path = args.output_dir / "macro_topic_document_counts.csv"
    counts.to_csv(csv_path, index=False)
    final_by_source_csv_path = args.output_dir / "macro_topic_document_counts_final_by_source.csv"
    counts_final_by_source.to_csv(final_by_source_csv_path, index=False)

    manifest = {
        "selected_topics": str(args.selected_topics),
        "merged_root": str(args.merged_root),
        "review_root": str(args.review_root),
        "output_dir": str(args.output_dir),
        "document_id_column_used": doc_column,
        "macro_topic_rows": int(len(counts)),
        "pre_review_csv": str(csv_path),
        "final_by_source_csv": str(final_by_source_csv_path),
    }
    (args.output_dir / "macro_topic_document_counts_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("PRE_REVIEW_COUNTS")
    print(counts.to_string(index=False))
    print("\nFINAL_BY_SOURCE_COUNTS")
    print(counts_final_by_source.to_string(index=False))


if __name__ == "__main__":
    main()
