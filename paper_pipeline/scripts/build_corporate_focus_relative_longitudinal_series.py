#!/usr/bin/env python3
"""Build relative longitudinal series and topic tables for corporate-focus results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
REVIEW_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_review_with_overrides"
STAGE12_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_stage12_colab_drive_with_overrides"
PAPER_TABLES_ROOT = PIPELINE_ROOT / "outputs" / "paper_tables"

SELECTED_TOPICS_PATH = STAGE12_ROOT / "data" / "selected_micro_topics.csv"
YEAR_EVIDENCE_PATH = STAGE12_ROOT / "data" / "micro_topic_year_evidence.csv"
INCLUDED_CORPORATE_PATH = REVIEW_ROOT / "included_corporate_groups.csv"
ALL_GROUPS_PATH = REVIEW_ROOT / "corporate_focus_all_groups_optional.csv"
MASTER_REVIEW_PATH = REVIEW_ROOT / "corporate_focus_master_review.csv"
COMMENTED_DECISIONS_PATH = REVIEW_ROOT / "corporate_focus_commented_decision_rows.csv"
GROUP_EXCLUSION_DECISIONS_PATH = (
    PIPELINE_ROOT
    / "outputs"
    / "corporate_external_group_exclusion_review"
    / "group_exclusion_decisions.csv"
)
DENOMINATORS_PATH = (
    PAPER_TABLES_ROOT
    / "corporate_focus_document_counts"
    / "macro_topic_document_counts_final_by_source.csv"
)
MERGED_ROOT = PIPELINE_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"

OUTPUT_DIR = PAPER_TABLES_ROOT / "corporate_focus_relative_longitudinal_series"
OUTPUT_CSV_PATH = OUTPUT_DIR / "corporate_topic_external_relative_longitudinal.csv"
UNPAIRED_OUTPUT_CSV_PATH = OUTPUT_DIR / "unpaired_external_relative_longitudinal.csv"
LEGACY_OUTPUT_CSV_PATH = OUTPUT_DIR / "corporate_topic_external_relative_longitudinal_period_total_legacy.csv"
LEGACY_UNPAIRED_OUTPUT_CSV_PATH = OUTPUT_DIR / "unpaired_external_relative_longitudinal_period_total_legacy.csv"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "corporate_topic_external_relative_longitudinal_manifest.json"
TOPIC_TABLES_DIR = OUTPUT_DIR / "topic_tables"
TOPIC_TABLES_XLSX_PATH = TOPIC_TABLES_DIR / "corporate_focus_longitudinal_topic_tables.xlsx"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
YEARS = list(range(2000, 2026))
SERIES_CONFIG = [
    ("corporate", "corporate", "corporate_document_count"),
    ("academic_aggregate", "academic", "academic_document_count"),
    ("media_aggregate", "media", "media_document_count"),
]
MACRO_TOPIC_NAME_MAP = {
    "T1": "Clean energy transition",
    "T2": "Operational sustainability and circular production",
    "T3": "Sustainable products, services and consumption",
    "T4": "Climate strategy, carbon governance and disclosure",
    "T5": "Climate risk, adaptation and resilience",
    "T6": "Ecosystems, pollution, and environmental stewardship",
}
TOPIC_KEY = ["macro_topic", "subgroup_noncorporate", "final_merge_group_id_noncorporate"]
TOPIC_TABLE_COLUMNS = [
    "macro_topic",
    "macro_topic_name",
    "paper_status",
    "inclusion_reason",
    "source_noncorporate",
    "subgroup_noncorporate",
    "final_merge_group_id_noncorporate",
    "micro_topic_id_noncorporate",
    "topic_label_refined_noncorporate",
    "topic_name_original_noncorporate",
    "overall_summary_noncorporate",
    "phase_1_years_noncorporate",
    "phase_2_years_noncorporate",
    "phase_3_years_noncorporate",
    "phase_4_years_noncorporate",
    "external_unique_document_count",
    "external_topic_size_count",
    "external_group_size_count",
    "subgroup_corporate",
    "final_merge_group_id_corporate",
    "micro_topic_id_corporate",
    "topic_label_refined_corporate",
    "topic_name_original_corporate",
    "overall_summary_corporate",
    "corporate_unique_document_count",
    "best_cosine_similarity",
    "direct_pair_count",
    "best_pair_id",
    "manual_review_notes",
    "review_notes",
]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def clean_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def normalize_review_decision(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if text == "delete":
        return "delete"
    if text == "not related":
        return "not related"
    return ""


def decision_to_status(value: str) -> str:
    if value == "delete":
        return "excluded"
    if value == "not related":
        return "external_relevant_unpaired"
    return "aligned"


def topic_key(subgroup: object, final_merge_group_id: object) -> tuple[str, str]:
    return (str(subgroup), str(final_merge_group_id))


def load_group_exclusions() -> tuple[set[tuple[str, str]], list[str]]:
    if not GROUP_EXCLUSION_DECISIONS_PATH.exists():
        return set(), []
    decisions = pd.read_csv(GROUP_EXCLUSION_DECISIONS_PATH)
    decision_column = (
        "_normalized_review_decision"
        if "_normalized_review_decision" in decisions.columns
        else "review_decision"
    )
    excluded = decisions.loc[
        decisions[decision_column].fillna("").astype(str).str.strip().str.lower().eq("exclude")
    ].copy()
    if excluded.empty:
        return set(), []
    keys = {
        topic_key(row.subgroup, row.final_merge_group_id)
        for row in excluded.itertuples(index=False)
    }
    ids = sorted(
        excluded["topic_id"].fillna("").astype(str).replace("", pd.NA).dropna().unique().tolist()
    )
    return keys, ids


def exclude_group_rows(frame: pd.DataFrame, excluded_groups: set[tuple[str, str]]) -> pd.DataFrame:
    if not excluded_groups or frame.empty or not {"subgroup", "final_merge_group_id"}.issubset(frame.columns):
        return frame
    keep_mask = [
        topic_key(subgroup, group_id) not in excluded_groups
        for subgroup, group_id in zip(frame["subgroup"], frame["final_merge_group_id"])
    ]
    return frame.loc[keep_mask].copy()


def filter_review_for_group_exclusions(
    review: pd.DataFrame,
    excluded_groups: set[tuple[str, str]],
) -> pd.DataFrame:
    if not excluded_groups or review.empty:
        return review
    corporate_key_columns = ["subgroup_corporate", "final_merge_group_id_corporate"]
    external_key_columns = ["subgroup_noncorporate", "final_merge_group_id_noncorporate"]
    keep_mask = []
    for row in review.itertuples(index=False):
        corporate_group = topic_key(
            getattr(row, corporate_key_columns[0]),
            getattr(row, corporate_key_columns[1]),
        )
        external_group = topic_key(
            getattr(row, external_key_columns[0]),
            getattr(row, external_key_columns[1]),
        )
        keep_mask.append(corporate_group not in excluded_groups and external_group not in excluded_groups)
    return review.loc[keep_mask].copy()


def detect_document_id_column(columns: pd.Index) -> str:
    for name in ["source_doc_id", "document_id", "doc_id"]:
        if name in columns:
            return name
    raise KeyError("Could not find a stable document identifier column in document_topics.csv")


def macro_topic_name(macro_topic: object, fallback: object = "") -> str:
    text = clean_text(fallback).strip()
    return text or MACRO_TOPIC_NAME_MAP.get(str(macro_topic), str(macro_topic))


def load_document_maps(
    groups: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], set[str]],
    dict[tuple[str, str, int], set[str]],
    dict[tuple[str, str, int], int],
    dict[str, str],
]:
    counts: dict[tuple[str, str], int] = {}
    group_doc_sets: dict[tuple[str, str], set[str]] = {}
    group_year_doc_sets: dict[tuple[str, str, int], set[str]] = {}
    annual_denominators: dict[tuple[str, str, int], int] = {}
    doc_columns_used: dict[str, str] = {}
    groups = groups[["subgroup", "final_merge_group_id"]].dropna().drop_duplicates().copy()
    groups["final_merge_group_id"] = groups["final_merge_group_id"].astype(str)

    for subgroup, subgroup_frame in groups.groupby("subgroup", sort=True):
        path = MERGED_ROOT / subgroup / "document_topics.csv"
        header = pd.read_csv(path, nrows=0)
        doc_col = detect_document_id_column(header.columns)
        doc_columns_used[subgroup] = doc_col
        doc_topics = pd.read_csv(
            path,
            usecols=[doc_col, "assigned_label", "source", "year", "final_merge_group_id"],
        )
        doc_topics = doc_topics.rename(columns={doc_col: "stable_doc_id"})
        doc_topics = doc_topics.dropna(subset=["stable_doc_id", "assigned_label", "source", "year"]).copy()
        doc_topics["final_merge_group_id"] = doc_topics["final_merge_group_id"].astype(str)
        doc_topics["stable_doc_id"] = doc_topics["stable_doc_id"].astype(str)
        doc_topics["year"] = pd.to_numeric(doc_topics["year"], errors="coerce")
        doc_topics = doc_topics.dropna(subset=["year"]).copy()
        doc_topics["year"] = doc_topics["year"].astype(int)

        annual_counts = (
            doc_topics.groupby(["assigned_label", "source", "year"], sort=False)["stable_doc_id"]
            .nunique()
            .astype(int)
        )
        for (macro_topic, source, year), value in annual_counts.items():
            annual_denominators[(str(macro_topic), str(source), int(year))] = int(value)

        allowed = set(subgroup_frame["final_merge_group_id"].astype(str))
        filtered = doc_topics.loc[doc_topics["final_merge_group_id"].isin(allowed)].copy()
        for group_id, group in filtered.groupby("final_merge_group_id", sort=False):
            docs = set(group["stable_doc_id"].dropna().astype(str))
            group_doc_sets[(subgroup, str(group_id))] = docs
            counts[(subgroup, str(group_id))] = len(docs)
        for (group_id, year), group in filtered.groupby(["final_merge_group_id", "year"], sort=False):
            group_year_doc_sets[(subgroup, str(group_id), int(year))] = set(
                group["stable_doc_id"].dropna().astype(str)
            )
    return counts, group_doc_sets, group_year_doc_sets, annual_denominators, doc_columns_used


def group_document_count(group_doc_counts: dict[tuple[str, str], int], subgroup: object, group_id: object) -> int:
    return int(group_doc_counts.get((str(subgroup), str(group_id)), 0))


def group_year_documents(
    group_year_doc_sets: dict[tuple[str, str, int], set[str]],
    subgroup: object,
    group_id: object,
    year: object,
) -> set[str]:
    return group_year_doc_sets.get((str(subgroup), str(group_id), int(year)), set())


def load_selected_year_evidence() -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_TOPICS_PATH)
    evidence = pd.read_csv(
        YEAR_EVIDENCE_PATH,
        usecols=["subgroup", "source", "assigned_label", "micro_topic_id", "year", "year_frequency"],
    )
    selected_key = selected[
        [
            "subgroup",
            "source",
            "assigned_label",
            "micro_topic_id",
            "final_merge_group_id",
            "topic_size",
        ]
    ].drop_duplicates()

    merged = evidence.merge(
        selected_key,
        on=["subgroup", "source", "assigned_label", "micro_topic_id"],
        how="inner",
        validate="many_to_one",
    )
    merged["final_merge_group_id"] = merged["final_merge_group_id"].astype(str)
    merged["macro_topic_name"] = merged["assigned_label"].map(MACRO_TOPIC_NAME_MAP)
    return merged


def load_group_metadata() -> pd.DataFrame:
    groups = pd.read_csv(ALL_GROUPS_PATH)
    groups["final_merge_group_id"] = groups["final_merge_group_id"].astype(str)
    groups["macro_topic_name"] = [
        macro_topic_name(topic, name)
        for topic, name in zip(groups["macro_topic"], groups.get("macro_topic_name", ""))
    ]
    return groups


def load_review_with_status() -> pd.DataFrame:
    master = pd.read_csv(MASTER_REVIEW_PATH)
    decisions = pd.read_csv(COMMENTED_DECISIONS_PATH)
    decision_column = "_normalized_review_decision" if "_normalized_review_decision" in decisions.columns else "review_decision"
    decisions = decisions[TOPIC_KEY + [decision_column]].copy()
    decisions["_normalized_review_decision"] = decisions[decision_column].map(normalize_review_decision)
    decisions = decisions.loc[decisions["_normalized_review_decision"] != ""]
    decisions = decisions[TOPIC_KEY + ["_normalized_review_decision"]].drop_duplicates(
        subset=TOPIC_KEY,
        keep="last",
    )

    review = master.merge(decisions, on=TOPIC_KEY, how="left")
    review["_normalized_review_decision"] = review["_normalized_review_decision"].fillna("")
    review["paper_status"] = review["_normalized_review_decision"].map(decision_to_status)
    review["macro_topic_name"] = [
        macro_topic_name(topic, name)
        for topic, name in zip(review["macro_topic"], review.get("macro_topic_name", ""))
    ]
    review["final_merge_group_id_noncorporate"] = review["final_merge_group_id_noncorporate"].astype(str)
    review["final_merge_group_id_corporate"] = review["final_merge_group_id_corporate"].astype(str)
    return review


def load_final_aligned_map(review: pd.DataFrame) -> pd.DataFrame:
    aligned = review.loc[review["paper_status"] == "aligned"].copy()
    return aligned[
        [
            "macro_topic",
            "subgroup_noncorporate",
            "source_noncorporate",
            "final_merge_group_id_noncorporate",
            "subgroup_corporate",
            "final_merge_group_id_corporate",
        ]
    ].drop_duplicates()


def build_corporate_metadata(
    group_doc_counts: dict[tuple[str, str], int],
    group_metadata: pd.DataFrame,
    included_corporate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if included_corporate is None:
        included_corporate = pd.read_csv(INCLUDED_CORPORATE_PATH)
    included_corporate["final_merge_group_id"] = included_corporate["final_merge_group_id"].astype(str)
    included_corporate["macro_topic"] = included_corporate["assigned_label"]

    metadata = group_metadata[
        [
            "subgroup",
            "final_merge_group_id",
            "topic_label_refined",
            "overall_summary",
            "topic_size",
            "group_size",
        ]
    ].rename(
        columns={
            "topic_size": "topic_size_metadata",
            "group_size": "group_size_metadata",
        }
    )

    corporate_meta = included_corporate.merge(
        metadata,
        on=["subgroup", "final_merge_group_id"],
        how="left",
        validate="one_to_one",
    )

    corporate_meta["macro_topic_name"] = [
        macro_topic_name(topic, name)
        for topic, name in zip(corporate_meta["macro_topic"], corporate_meta.get("macro_topic_name", ""))
    ]
    corporate_meta["corporate_topic_label"] = corporate_meta["topic_label_refined"].fillna(
        corporate_meta["topic_name_original"]
    )
    corporate_meta["corporate_topic_summary"] = corporate_meta["overall_summary"].fillna("")
    corporate_meta["corporate_topic_size_count"] = corporate_meta["topic_size_metadata"].fillna(
        corporate_meta["topic_size"]
    )
    corporate_meta["corporate_group_size_count"] = corporate_meta["group_size_metadata"].fillna(
        corporate_meta["group_size"]
    )
    corporate_meta["corporate_unique_document_count"] = corporate_meta.apply(
        lambda row: group_document_count(group_doc_counts, row["subgroup"], row["final_merge_group_id"]),
        axis=1,
    )
    corporate_meta = corporate_meta[
        [
            "macro_topic",
            "macro_topic_name",
            "subgroup",
            "final_merge_group_id",
            "corporate_topic_label",
            "corporate_topic_summary",
            "corporate_topic_size_count",
            "corporate_group_size_count",
            "corporate_unique_document_count",
        ]
    ].copy()
    corporate_meta = corporate_meta.rename(
        columns={
            "subgroup": "corporate_subgroup",
            "final_merge_group_id": "corporate_final_merge_group_id",
        }
    )
    corporate_meta["corporate_final_merge_group_id"] = corporate_meta["corporate_final_merge_group_id"].astype(str)
    return corporate_meta.sort_values(
        by=["macro_topic", "corporate_topic_size_count", "corporate_topic_label"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_series_lookup(year_evidence: pd.DataFrame) -> dict[tuple[str, str, int], int]:
    grouped = (
        year_evidence.groupby(["subgroup", "final_merge_group_id", "year"], sort=False)["year_frequency"]
        .sum()
        .astype(int)
    )
    return {
        (str(subgroup), str(group_id), int(year)): int(value)
        for (subgroup, group_id, year), value in grouped.items()
    }


def build_external_contributors(aligned_map: pd.DataFrame) -> dict[tuple[str, str, str], set[tuple[str, str]]]:
    contributors: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for row in aligned_map.itertuples(index=False):
        corp_key = (row.subgroup_corporate, row.final_merge_group_id_corporate, row.source_noncorporate)
        contributor = (row.subgroup_noncorporate, row.final_merge_group_id_noncorporate)
        contributors.setdefault((corp_key[0], corp_key[1], corp_key[2]), set()).add(contributor)
    return contributors


def load_denominator_map() -> dict[str, dict[str, int]]:
    denominators = pd.read_csv(DENOMINATORS_PATH).rename(columns={"macro_topic": "macro_topic_key"})
    return {
        row.macro_topic_key: {
            "academic": int(row.academic_document_count),
            "media": int(row.media_document_count),
            "corporate": int(row.corporate_document_count),
        }
        for row in denominators.itertuples(index=False)
    }


def group_meta_lookup(group_metadata: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.subgroup), str(row.final_merge_group_id)): row._asdict()
        for row in group_metadata.itertuples(index=False)
    }


def sum_group_documents(
    contributors: set[tuple[str, str]],
    group_doc_counts: dict[tuple[str, str], int],
) -> int:
    return int(sum(group_doc_counts.get((str(subgroup), str(group_id)), 0) for subgroup, group_id in contributors))


def union_group_documents(
    contributors: set[tuple[str, str]],
    group_doc_sets: dict[tuple[str, str], set[str]],
) -> set[str]:
    docs: set[str] = set()
    for subgroup, group_id in contributors:
        docs.update(group_doc_sets.get((str(subgroup), str(group_id)), set()))
    return docs


def union_group_year_documents(
    contributors: set[tuple[str, str]],
    group_year_doc_sets: dict[tuple[str, str, int], set[str]],
    year: int,
) -> set[str]:
    docs: set[str] = set()
    for subgroup, group_id in contributors:
        docs.update(group_year_doc_sets.get((str(subgroup), str(group_id), int(year)), set()))
    return docs


def sum_group_meta_value(
    contributors: set[tuple[str, str]],
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
    column: str,
) -> int:
    total = 0
    for subgroup, group_id in contributors:
        value = metadata_lookup.get((str(subgroup), str(group_id)), {}).get(column, 0)
        if not pd.isna(value):
            total += int(value)
    return total


def contributors_to_json(contributors: set[tuple[str, str]]) -> str:
    values = [
        {"subgroup": str(subgroup), "final_merge_group_id": str(group_id)}
        for subgroup, group_id in sorted(contributors)
    ]
    return json.dumps(values, ensure_ascii=False)


def annual_document_prevalence(year_doc_count: int, annual_denominator: int) -> float:
    if annual_denominator <= 0:
        return 0.0
    return float(year_doc_count) / float(annual_denominator)


def collect_needed_groups(review: pd.DataFrame, included_corporate: pd.DataFrame) -> pd.DataFrame:
    corporate_from_roster = included_corporate[["subgroup", "final_merge_group_id"]].copy()
    corporate_from_review = review[["subgroup_corporate", "final_merge_group_id_corporate"]].rename(
        columns={
            "subgroup_corporate": "subgroup",
            "final_merge_group_id_corporate": "final_merge_group_id",
        }
    )
    noncorporate = review[["subgroup_noncorporate", "final_merge_group_id_noncorporate"]].rename(
        columns={
            "subgroup_noncorporate": "subgroup",
            "final_merge_group_id_noncorporate": "final_merge_group_id",
        }
    )
    groups = pd.concat([corporate_from_roster, corporate_from_review, noncorporate], ignore_index=True)
    groups["final_merge_group_id"] = groups["final_merge_group_id"].astype(str)
    return groups.dropna().drop_duplicates()


def build_aligned_relative_frame(
    year_evidence: pd.DataFrame,
    review: pd.DataFrame,
    corporate_meta: pd.DataFrame,
    denominator_map: dict[str, dict[str, int]],
    group_doc_counts: dict[tuple[str, str], int],
    group_doc_sets: dict[tuple[str, str], set[str]],
    group_year_doc_sets: dict[tuple[str, str, int], set[str]],
    annual_denominator_map: dict[tuple[str, str, int], int],
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    aligned_map = load_final_aligned_map(review)
    series_lookup = build_series_lookup(year_evidence)
    external_contributors = build_external_contributors(aligned_map)

    rows: list[dict[str, object]] = []
    for corp in corporate_meta.itertuples(index=False):
        for year in YEARS:
            for series_role, source, _denominator_column in SERIES_CONFIG:
                contributors: set[tuple[str, str]] = set()
                if series_role == "corporate":
                    year_frequency = series_lookup.get((corp.corporate_subgroup, corp.corporate_final_merge_group_id, year), 0)
                    series_doc_count = int(corp.corporate_unique_document_count)
                    series_topic_size = int(corp.corporate_topic_size_count)
                    year_doc_count = len(
                        group_year_documents(
                            group_year_doc_sets,
                            corp.corporate_subgroup,
                            corp.corporate_final_merge_group_id,
                            year,
                        )
                    )
                    contributor_count = 0
                else:
                    contributors = external_contributors.get(
                        (corp.corporate_subgroup, corp.corporate_final_merge_group_id, source),
                        set(),
                    )
                    year_frequency = sum(
                        series_lookup.get((subgroup, group_id, year), 0)
                        for subgroup, group_id in contributors
                    )
                    contributor_count = len(contributors)
                    series_doc_count = len(union_group_documents(contributors, group_doc_sets))
                    series_topic_size = sum_group_meta_value(contributors, metadata_lookup, "topic_size")
                    year_doc_count = len(union_group_year_documents(contributors, group_year_doc_sets, year))

                denominator = denominator_map[corp.macro_topic][source]
                relative = float(year_frequency) / float(denominator) if denominator else 0.0
                annual_denominator = annual_denominator_map.get((str(corp.macro_topic), source, int(year)), 0)
                rows.append(
                    {
                        "macro_topic": corp.macro_topic,
                        "macro_topic_name": corp.macro_topic_name,
                        "corporate_subgroup": corp.corporate_subgroup,
                        "corporate_final_merge_group_id": corp.corporate_final_merge_group_id,
                        "corporate_topic_label": corp.corporate_topic_label,
                        "corporate_topic_summary": corp.corporate_topic_summary,
                        "year": year,
                        "series_role": series_role,
                        "source": source,
                        "year_frequency": int(year_frequency),
                        "denominator_source_docs_in_macro_topic": int(denominator),
                        "relative_share_of_macro_topic_source_docs": relative,
                        "year_document_count": int(year_doc_count),
                        "annual_source_macro_document_count": int(annual_denominator),
                        "annual_document_prevalence": annual_document_prevalence(
                            int(year_doc_count),
                            int(annual_denominator),
                        ),
                        "n_contributing_external_topics": int(contributor_count),
                        "contributing_external_group_count": int(contributor_count),
                        "contributing_external_groups_json": contributors_to_json(contributors),
                        "series_unique_document_count": int(series_doc_count),
                        "series_topic_size_count": int(series_topic_size),
                        "corporate_topic_size_count": int(corp.corporate_topic_size_count),
                        "corporate_group_size_count": int(corp.corporate_group_size_count),
                        "corporate_unique_document_count": int(corp.corporate_unique_document_count),
                    }
                )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        by=["macro_topic", "corporate_topic_label", "corporate_final_merge_group_id", "series_role", "year"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_unpaired_relative_frame(
    year_evidence: pd.DataFrame,
    review: pd.DataFrame,
    denominator_map: dict[str, dict[str, int]],
    group_doc_counts: dict[tuple[str, str], int],
    group_year_doc_sets: dict[tuple[str, str, int], set[str]],
    annual_denominator_map: dict[tuple[str, str, int], int],
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    series_lookup = build_series_lookup(year_evidence)
    unpaired = (
        review.loc[review["paper_status"] == "external_relevant_unpaired"]
        .drop_duplicates(subset=TOPIC_KEY, keep="last")
        .copy()
    )

    rows: list[dict[str, object]] = []
    for topic in unpaired.itertuples(index=False):
        subgroup = str(topic.subgroup_noncorporate)
        group_id = str(topic.final_merge_group_id_noncorporate)
        source = str(topic.source_noncorporate)
        macro_topic = str(topic.macro_topic)
        meta = metadata_lookup.get((subgroup, group_id), {})
        denominator = denominator_map[macro_topic][source]
        topic_doc_count = group_document_count(group_doc_counts, subgroup, group_id)
        topic_size = clean_int(meta.get("topic_size", 0))
        group_size = clean_int(meta.get("group_size", 0))

        for year in YEARS:
            year_frequency = series_lookup.get((subgroup, group_id, year), 0)
            relative = float(year_frequency) / float(denominator) if denominator else 0.0
            year_doc_count = len(group_year_documents(group_year_doc_sets, subgroup, group_id, year))
            annual_denominator = annual_denominator_map.get((macro_topic, source, int(year)), 0)
            rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": macro_topic_name(macro_topic, getattr(topic, "macro_topic_name", "")),
                    "source": source,
                    "subgroup": subgroup,
                    "final_merge_group_id": group_id,
                    "micro_topic_id": getattr(topic, "micro_topic_id_noncorporate"),
                    "topic_label": clean_text(getattr(topic, "topic_label_refined_noncorporate")),
                    "topic_name_original": clean_text(getattr(topic, "topic_name_original_noncorporate")),
                    "topic_summary": clean_text(getattr(topic, "overall_summary_noncorporate")),
                    "year": year,
                    "year_frequency": int(year_frequency),
                    "denominator_source_docs_in_macro_topic": int(denominator),
                    "relative_share_of_macro_topic_source_docs": relative,
                    "year_document_count": int(year_doc_count),
                    "annual_source_macro_document_count": int(annual_denominator),
                    "annual_document_prevalence": annual_document_prevalence(
                        int(year_doc_count),
                        int(annual_denominator),
                    ),
                    "external_unique_document_count": int(topic_doc_count),
                    "external_topic_size_count": int(topic_size),
                    "external_group_size_count": int(group_size),
                    "best_matched_corporate_group_id": clean_text(getattr(topic, "final_merge_group_id_corporate")),
                    "best_matched_corporate_label": clean_text(getattr(topic, "topic_label_refined_corporate")),
                    "best_cosine_similarity": getattr(topic, "best_cosine_similarity"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["macro_topic", "source", "external_unique_document_count", "topic_label", "year"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def add_topic_table_counts(
    frame: pd.DataFrame,
    group_doc_counts: dict[tuple[str, str], int],
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    result = frame.copy()
    result["external_unique_document_count"] = result.apply(
        lambda row: group_document_count(
            group_doc_counts,
            row["subgroup_noncorporate"],
            row["final_merge_group_id_noncorporate"],
        ),
        axis=1,
    )
    result["corporate_unique_document_count"] = result.apply(
        lambda row: group_document_count(
            group_doc_counts,
            row["subgroup_corporate"],
            row["final_merge_group_id_corporate"],
        ),
        axis=1,
    )
    result["external_topic_size_count"] = result.apply(
        lambda row: clean_int(
            metadata_lookup
            .get((str(row["subgroup_noncorporate"]), str(row["final_merge_group_id_noncorporate"])), {})
            .get("topic_size", 0)
        ),
        axis=1,
    )
    result["external_group_size_count"] = result.apply(
        lambda row: clean_int(
            metadata_lookup
            .get((str(row["subgroup_noncorporate"]), str(row["final_merge_group_id_noncorporate"])), {})
            .get("group_size", 0)
        ),
        axis=1,
    )
    return result


def select_topic_table_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in TOPIC_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[TOPIC_TABLE_COLUMNS].sort_values(
        by=[
            "macro_topic",
            "source_noncorporate",
            "external_unique_document_count",
            "topic_label_refined_noncorporate",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    )


def build_topic_tables(
    review: pd.DataFrame,
    group_doc_counts: dict[tuple[str, str], int],
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    aligned = review.loc[review["paper_status"] == "aligned"].drop_duplicates(
        subset=[
            "macro_topic",
            "subgroup_corporate",
            "final_merge_group_id_corporate",
            "subgroup_noncorporate",
            "final_merge_group_id_noncorporate",
        ],
        keep="last",
    )
    unpaired = review.loc[review["paper_status"] == "external_relevant_unpaired"].drop_duplicates(
        subset=TOPIC_KEY,
        keep="last",
    )
    excluded = review.loc[review["paper_status"] == "excluded"].drop_duplicates(
        subset=TOPIC_KEY,
        keep="last",
    )
    return {
        "aligned_aggregate_topics": select_topic_table_columns(
            add_topic_table_counts(aligned, group_doc_counts, metadata_lookup)
        ),
        "unpaired_external_topics": select_topic_table_columns(
            add_topic_table_counts(unpaired, group_doc_counts, metadata_lookup)
        ),
        "excluded_topics": select_topic_table_columns(
            add_topic_table_counts(excluded, group_doc_counts, metadata_lookup)
        ),
    }


def write_topic_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    ensure_directory(TOPIC_TABLES_DIR)
    written: dict[str, str] = {}

    for macro_topic in MACRO_TOPIC_ORDER:
        for table_name, table in tables.items():
            macro_table = table.loc[table["macro_topic"] == macro_topic].copy()
            path = TOPIC_TABLES_DIR / f"{macro_topic}_{table_name}.csv"
            macro_table.to_csv(path, index=False)
            written[f"{macro_topic}_{table_name}"] = str(path)

    readme = pd.DataFrame(
        [
            {
                "section": "aligned_aggregate_topics",
                "description": "Academic and media consolidated topics that contribute to corporate-anchor longitudinal panels.",
            },
            {
                "section": "unpaired_external_topics",
                "description": "Relevant external topics retained without a corporate counterpart.",
            },
            {
                "section": "excluded_topics",
                "description": "Topics marked delete and excluded from substantive interpretation.",
            },
        ]
    )

    with pd.ExcelWriter(TOPIC_TABLES_XLSX_PATH, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        for macro_topic in MACRO_TOPIC_ORDER:
            startrow = 0
            for title, table_name in [
                ("Aligned aggregate topics used in corporate-anchor panels", "aligned_aggregate_topics"),
                ("Relevant external topics without corporate counterpart", "unpaired_external_topics"),
                ("Excluded topics", "excluded_topics"),
            ]:
                macro_table = tables[table_name].loc[tables[table_name]["macro_topic"] == macro_topic].copy()
                pd.DataFrame([[title]]).to_excel(
                    writer,
                    sheet_name=macro_topic,
                    startrow=startrow,
                    index=False,
                    header=False,
                )
                macro_table.to_excel(
                    writer,
                    sheet_name=macro_topic,
                    startrow=startrow + 1,
                    index=False,
                )
                startrow += len(macro_table) + 4

    written["excel"] = str(TOPIC_TABLES_XLSX_PATH)
    return written


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, object]]:
    excluded_groups, excluded_topic_ids = load_group_exclusions()
    year_evidence = load_selected_year_evidence()
    review = filter_review_for_group_exclusions(load_review_with_status(), excluded_groups)
    group_metadata = exclude_group_rows(load_group_metadata(), excluded_groups)
    metadata_lookup = group_meta_lookup(group_metadata)
    included_corporate = pd.read_csv(INCLUDED_CORPORATE_PATH)
    included_corporate["final_merge_group_id"] = included_corporate["final_merge_group_id"].astype(str)
    included_corporate = exclude_group_rows(included_corporate, excluded_groups)

    needed_groups = collect_needed_groups(review, included_corporate)
    (
        group_doc_counts,
        group_doc_sets,
        group_year_doc_sets,
        annual_denominator_map,
        doc_columns_used,
    ) = load_document_maps(needed_groups)
    corporate_meta = build_corporate_metadata(group_doc_counts, group_metadata, included_corporate)
    denominator_map = load_denominator_map()

    aligned_frame = build_aligned_relative_frame(
        year_evidence,
        review,
        corporate_meta,
        denominator_map,
        group_doc_counts,
        group_doc_sets,
        group_year_doc_sets,
        annual_denominator_map,
        metadata_lookup,
    )
    unpaired_frame = build_unpaired_relative_frame(
        year_evidence,
        review,
        denominator_map,
        group_doc_counts,
        group_year_doc_sets,
        annual_denominator_map,
        metadata_lookup,
    )
    topic_tables = build_topic_tables(review, group_doc_counts, metadata_lookup)
    topic_table_paths = write_topic_tables(topic_tables)

    validations = {
        "document_id_columns_used": doc_columns_used,
        "group_exclusion_decisions": str(GROUP_EXCLUSION_DECISIONS_PATH),
        "excluded_group_count": int(len(excluded_groups)),
        "excluded_topic_ids": excluded_topic_ids,
        "annual_denominator_source": "unique source documents by source, macro topic, and year from reviewed merged document_topics.csv",
        "annual_metric_columns": [
            "year_document_count",
            "annual_source_macro_document_count",
            "annual_document_prevalence",
        ],
        "annual_zero_denominator_rows": int(
            (aligned_frame["annual_source_macro_document_count"].eq(0)).sum()
            + (
                unpaired_frame["annual_source_macro_document_count"].eq(0).sum()
                if not unpaired_frame.empty
                else 0
            )
        ),
        "annual_zero_denominator_nonzero_numerator_rows": int(
            (
                aligned_frame["annual_source_macro_document_count"].eq(0)
                & aligned_frame["year_document_count"].gt(0)
            ).sum()
            + (
                (
                    unpaired_frame["annual_source_macro_document_count"].eq(0)
                    & unpaired_frame["year_document_count"].gt(0)
                ).sum()
                if not unpaired_frame.empty
                else 0
            )
        ),
        "retained_corporate_topics": int(len(corporate_meta)),
        "aligned_expected_rows": int(len(corporate_meta) * len(YEARS) * len(SERIES_CONFIG)),
        "aligned_actual_rows": int(len(aligned_frame)),
        "aligned_row_count_matches_expectation": int(len(aligned_frame))
        == int(len(corporate_meta) * len(YEARS) * len(SERIES_CONFIG)),
        "unpaired_topic_count": int(
            review.loc[review["paper_status"] == "external_relevant_unpaired"]
            .drop_duplicates(subset=TOPIC_KEY)
            .shape[0]
        ),
        "unpaired_expected_rows": int(
            review.loc[review["paper_status"] == "external_relevant_unpaired"]
            .drop_duplicates(subset=TOPIC_KEY)
            .shape[0]
            * len(YEARS)
        ),
        "unpaired_actual_rows": int(len(unpaired_frame)),
        "excluded_topic_count": int(
            review.loc[review["paper_status"] == "excluded"]
            .drop_duplicates(subset=TOPIC_KEY)
            .shape[0]
        ),
        "only_allowed_series_roles": sorted(aligned_frame["series_role"].unique().tolist()),
        "no_not_related_or_delete_contributors": True,
    }
    return aligned_frame, unpaired_frame, topic_table_paths, validations


def preserve_period_total_legacy_csv(source_path: Path, legacy_path: Path) -> None:
    if not source_path.exists() or legacy_path.exists():
        return
    columns = pd.read_csv(source_path, nrows=0).columns
    if "annual_document_prevalence" in columns:
        return
    shutil.copy2(source_path, legacy_path)


def main() -> None:
    ensure_directory(OUTPUT_DIR)
    ensure_directory(TOPIC_TABLES_DIR)

    preserve_period_total_legacy_csv(OUTPUT_CSV_PATH, LEGACY_OUTPUT_CSV_PATH)
    preserve_period_total_legacy_csv(UNPAIRED_OUTPUT_CSV_PATH, LEGACY_UNPAIRED_OUTPUT_CSV_PATH)

    aligned_frame, unpaired_frame, topic_table_paths, validations = build_outputs()
    aligned_frame.to_csv(OUTPUT_CSV_PATH, index=False)
    unpaired_frame.to_csv(UNPAIRED_OUTPUT_CSV_PATH, index=False)

    manifest = {
        "output_csv": str(OUTPUT_CSV_PATH),
        "unpaired_output_csv": str(UNPAIRED_OUTPUT_CSV_PATH),
        "legacy_period_total_csv": str(LEGACY_OUTPUT_CSV_PATH),
        "legacy_unpaired_period_total_csv": str(LEGACY_UNPAIRED_OUTPUT_CSV_PATH),
        "topic_tables_dir": str(TOPIC_TABLES_DIR),
        "topic_tables_excel": str(TOPIC_TABLES_XLSX_PATH),
        "topic_table_paths": topic_table_paths,
        "source_files": {
            "selected_topics": str(SELECTED_TOPICS_PATH),
            "year_evidence": str(YEAR_EVIDENCE_PATH),
            "included_corporate_groups": str(INCLUDED_CORPORATE_PATH),
            "all_groups": str(ALL_GROUPS_PATH),
            "master_review": str(MASTER_REVIEW_PATH),
            "commented_decisions": str(COMMENTED_DECISIONS_PATH),
            "denominators": str(DENOMINATORS_PATH),
            "merged_root": str(MERGED_ROOT),
        },
        "validations": validations,
        "aligned_macro_topic_rows": (
            aligned_frame.groupby("macro_topic").size().reindex(MACRO_TOPIC_ORDER, fill_value=0).astype(int).to_dict()
        ),
        "unpaired_macro_topic_rows": (
            unpaired_frame.groupby("macro_topic").size().reindex(MACRO_TOPIC_ORDER, fill_value=0).astype(int).to_dict()
            if not unpaired_frame.empty
            else {topic: 0 for topic in MACRO_TOPIC_ORDER}
        ),
        "series_role_rows": aligned_frame["series_role"].value_counts().sort_index().astype(int).to_dict(),
    }
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ALIGNED_ROWS_BY_MACRO")
    print(aligned_frame.groupby("macro_topic").size().reindex(MACRO_TOPIC_ORDER, fill_value=0).to_string())
    print("\nUNPAIRED_ROWS_BY_MACRO")
    if unpaired_frame.empty:
        print(pd.Series(0, index=MACRO_TOPIC_ORDER).to_string())
    else:
        print(unpaired_frame.groupby("macro_topic").size().reindex(MACRO_TOPIC_ORDER, fill_value=0).to_string())
    print("\nVALIDATIONS")
    print(json.dumps(validations, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
