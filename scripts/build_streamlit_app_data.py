#!/usr/bin/env python3
"""Build data files for the Streamlit reader companion."""

from __future__ import annotations

import argparse
import ast
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
TOPIC_TEXT_OVERRIDES = {
    "corporate_T1::corporate_T1_PG03": {
        "group_id": "corporate_T1_PG03",
        "old_label": "HCF Capability and Renewable Profitability Trajectory",
        "label": "Renewable Project Viability and Market Constraints",
        "summary": (
            "The micro-topic tracks corporate framing of renewable and adjacent energy-transition activity "
            "through commercialization constraints: market adoption, financing, transmission access, "
            "regulatory and permitting exposure, competition, and supply-chain or customer-demand risks. "
            "HCF is a prominent early example in the evidence, but the consolidated topic is broader than "
            "HCF-specific technology."
        ),
        "interpretation": (
            "The renewable project viability panel is corporate-only. It contains 57 corporate documents, "
            "is active from 2009 to 2025, and peaks in 2011 at 3.73% of corporate T1 documents. The qualitative "
            "topic description moves from early market-entry challenges toward infrastructure bottlenecks, "
            "financing, regulatory compliance, permitting, supply-chain vulnerabilities, and customer-demand "
            "risks in renewable and adjacent transition technologies. HCF is a frequent early example in the "
            "evidence, but the consolidated topic should be read more broadly as a commercialization and "
            "viability framing of renewable activity. The absence of retained academic or media counterparts "
            "suggests that this corporate discourse is not mirroring a broad external issue trajectory; instead, "
            "it translates transition language into business constraints around project development, market "
            "entry, and profitability."
        ),
    },
    "corporate_T1::corporate_T1_S015": {
        "group_id": "corporate_T1_S015",
        "old_label": "Contractual Risk and Negotiation Dynamics in Financing",
        "label": "Energy Financing, Contract Risk, and Negotiation Dynamics",
        "summary": (
            "The micro-topic tracks financing and contract risks in clean-energy and energy-market "
            "business activity, moving from fixed-price service-contract exposure and clean-energy "
            "promotional spending to competition, grid or market access, energy purchase agreement "
            "negotiation, cooperative and utility structures, and structural liabilities."
        ),
        "interpretation": (
            "The energy financing and contract-risk panel is corporate-only. It captures how clean-energy "
            "and energy-market business activity is translated into disclosure about fixed-price contract "
            "exposure, promotional spending, competitive pressure, grid or market access, negotiation of "
            "energy purchase agreements, cooperative and utility structures, and structural liabilities."
        ),
    },
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
GUARDIAN_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sec-raw-dir", type=Path, default=None)
    parser.add_argument("--group-exclusion-decisions", type=Path, default=None)
    parser.add_argument("--snippet-chars", type=int, default=360)
    parser.add_argument("--guardian-word-limit", type=int, default=300)
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
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def default_group_exclusion_decisions_path(pipeline_root: Path) -> Path:
    return (
        pipeline_root
        / "outputs"
        / "corporate_external_group_exclusion_review"
        / "group_exclusion_decisions.csv"
    )


def load_excluded_topics(path: Path | None) -> tuple[set[str], Path | None, pd.DataFrame]:
    if path is None or not path.exists():
        return set(), None, pd.DataFrame()
    decisions = read_csv(path).fillna("")
    if "topic_id" not in decisions.columns:
        raise KeyError(f"Expected topic_id in group exclusion decisions: {path}")
    if "_normalized_review_decision" in decisions.columns:
        mask = decisions["_normalized_review_decision"].map(clean_text).str.lower().eq("exclude")
    elif "review_decision" in decisions.columns:
        mask = decisions["review_decision"].map(clean_text).str.lower().eq("exclude")
    else:
        mask = pd.Series([True] * len(decisions), index=decisions.index)
    topic_ids = set(decisions.loc[mask, "topic_id"].map(clean_text))
    return {topic_id for topic_id in topic_ids if topic_id}, path, decisions.loc[mask].copy()


def parse_json(value: object, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def parse_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def compact_representation(value: object, limit: int = 12) -> str:
    items = [clean_text(item) for item in parse_list(value)]
    items = [item for item in items if item]
    if not items:
        text = clean_text(value)
        return "" if text.startswith("[") and text.endswith("]") else text
    return " · ".join(items[:limit])


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


def apply_topic_overrides(topics: pd.DataFrame) -> pd.DataFrame:
    if topics.empty:
        return topics
    updated = topics.copy()
    for topic_id, override in TOPIC_TEXT_OVERRIDES.items():
        mask = updated["topic_id"].astype(str) == topic_id
        if not mask.any():
            continue
        updated.loc[mask, "display_label"] = override["label"]
        if "topic_label_refined" in updated.columns:
            updated.loc[mask, "topic_label_refined"] = override["label"]
        if "overall_summary" in updated.columns:
            updated.loc[mask, "overall_summary"] = override["summary"]
    return updated


def apply_corporate_topic_label_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    updated = frame.copy()
    for topic_id, override in TOPIC_TEXT_OVERRIDES.items():
        group_id = str(override["group_id"])
        label = str(override["label"])
        summary = str(override["summary"])
        old_label = str(override["old_label"])

        if "corporate_topic_id" in updated.columns:
            mask = updated["corporate_topic_id"].astype(str) == topic_id
            for column in ["corporate_topic_label", "topic_label_refined_corporate"]:
                if column in updated.columns:
                    updated.loc[mask, column] = label
            if "corporate_topic_summary" in updated.columns:
                updated.loc[mask, "corporate_topic_summary"] = summary

        for id_column, label_column in [
            ("corporate_final_merge_group_id", "corporate_topic_label"),
            ("final_merge_group_id_corporate", "topic_label_refined_corporate"),
            ("best_matched_corporate_group_id", "best_matched_corporate_label"),
        ]:
            if id_column in updated.columns and label_column in updated.columns:
                mask = updated[id_column].astype(str) == group_id
                updated.loc[mask, label_column] = label

        for column in updated.columns:
            if column.endswith("label") or "label" in column:
                updated[column] = updated[column].replace(old_label, label)
    return updated


def apply_interpretation_overrides(interpretations: dict[str, Any]) -> dict[str, Any]:
    anchors = dict(interpretations.get("anchors", {}))
    for override in TOPIC_TEXT_OVERRIDES.values():
        old_label = str(override["old_label"])
        label = str(override["label"])
        anchors.pop(old_label, None)
        anchors[label] = str(override["interpretation"])
    return {**interpretations, "anchors": anchors}


def sec_record_for_source_doc_id(
    source_doc_id: str,
    sec_lookup: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    if not sec_lookup:
        return {}
    return sec_lookup.get(source_doc_id, {})


def source_link(source: str, source_doc_id: str, sec_lookup: dict[str, dict[str, str]] | None = None) -> str:
    if not source_doc_id:
        return ""
    if source == "media":
        return f"https://www.theguardian.com/{source_doc_id}"
    if source == "academic":
        return f"https://www.semanticscholar.org/paper/{source_doc_id}"
    if source == "corporate":
        sec_record = sec_record_for_source_doc_id(source_doc_id, sec_lookup)
        filing_url = clean_text(sec_record.get("filing_url", ""))
        if filing_url:
            return filing_url
        identifier = corporate_identifier_from_source_doc_id(source_doc_id, sec_lookup)
        if identifier:
            return f"https://www.sec.gov/edgar/browse/?CIK={identifier}&owner=exclude&action=getcompany"
    return ""


def accession_from_source_doc_id(source_doc_id: str) -> str:
    match = re.search(r"\d{10}-\d{2}-\d{6}", source_doc_id)
    return match.group(0) if match else ""


def corporate_identifier_from_source_doc_id(
    source_doc_id: str,
    sec_lookup: dict[str, dict[str, str]] | None = None,
) -> str:
    sec_record = sec_record_for_source_doc_id(source_doc_id, sec_lookup)
    cik = clean_text(sec_record.get("cik", ""))
    if cik:
        return cik.zfill(10) if cik.isdigit() else cik
    accession = accession_from_source_doc_id(source_doc_id)
    if not accession:
        return ""
    prefix = source_doc_id.split(f"-{accession}", 1)[0].strip("-")
    return prefix


def item_from_source_doc_id(source_doc_id: str) -> str:
    match = re.search(r"-(1A|7)$", source_doc_id)
    return match.group(1) if match else ""


def guardian_date_from_source_doc_id(source_doc_id: str) -> str:
    match = re.search(r"/(\d{4})/([a-z]{3})/(\d{1,2})/", source_doc_id)
    if not match:
        return ""
    year, month, day = match.groups()
    month_number = GUARDIAN_MONTHS.get(month.lower())
    if not month_number:
        return ""
    return f"{year}-{month_number}-{int(day):02d}"


def source_date(source: str, source_doc_id: str, year: object) -> str:
    if source == "media":
        return guardian_date_from_source_doc_id(source_doc_id) or clean_text(year)
    return clean_text(year)


def truncate_words(text: str, limit: int) -> tuple[str, bool, int]:
    words = text.split()
    if limit <= 0 or len(words) <= limit:
        return text, False, len(words)
    return " ".join(words[:limit]).rstrip(" ,;:") + " ...", True, limit


def display_text(
    text: str,
    source: str,
    public_safe: bool,
    snippet_limit: int,
    guardian_word_limit: int,
) -> tuple[str, bool, int, str]:
    text = clean_text(text)
    if not public_safe:
        return text, False, len(text.split()), "local_full_text"
    if source == "media":
        limited, truncated, word_count = truncate_words(text, guardian_word_limit)
        return limited, truncated, word_count, "guardian_limited_excerpt"
    if source in {"academic", "corporate"}:
        return text, False, len(text.split()), "full_research_unit"
    if len(text) <= snippet_limit:
        return text, False, len(text.split()), "public_safe_snippet"
    clipped = text[:snippet_limit].rsplit(" ", 1)[0].strip()
    return f"{clipped} ...", True, len(clipped.split()), "public_safe_snippet"


def reference_note(
    source: str,
    source_doc_id: str,
    date_text: str,
    guardian_word_limit: int,
    sec_lookup: dict[str, dict[str, str]] | None = None,
) -> str:
    link = source_link(source, source_doc_id, sec_lookup)
    if source == "media":
        return (
            f"The Guardian, {date_text or 'date unavailable'}. "
            f"Original article: {link}. "
            f"Excerpt limited to {guardian_word_limit} words because Guardian terms govern reuse of Guardian content."
        )
    if source == "academic":
        return f"Academic abstract/chunk, {date_text or 'year unavailable'}. Original record: {link}."
    if source == "corporate":
        sec_record = sec_record_for_source_doc_id(source_doc_id, sec_lookup)
        accession = accession_from_source_doc_id(source_doc_id)
        item = item_from_source_doc_id(source_doc_id)
        item_text = f", Item {item}" if item else ""
        accession_text = f" accession {accession}" if accession else ""
        sic_text = f", source SIC {sec_record['sic_primary']}" if sec_record.get("sic_primary") else ""
        return f"SEC 10-K source unit{item_text}, {date_text or 'year unavailable'}{accession_text}{sic_text}. SEC filing: {link}."
    return f"Source unit, {date_text or 'date unavailable'}. Original: {link}."


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


def build_topics(pipeline_root: Path, excluded_topic_ids: set[str] | None = None) -> pd.DataFrame:
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
    if excluded_topic_ids:
        topics = topics.loc[~topics["topic_id"].isin(excluded_topic_ids)].copy()

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
    topics = apply_topic_overrides(topics[columns])
    return topics.sort_values(["macro_topic", "source", "paper_status", "display_label"], kind="mergesort")


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


def build_topic_merge_components(pipeline_root: Path, topics: pd.DataFrame) -> pd.DataFrame:
    merged_root = pipeline_root / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
    original_root = pipeline_root / "outputs" / "bertopic_micro_unsupervised_multiaspect"
    rows: list[dict[str, object]] = []
    topic_key = topics[
        ["topic_id", "macro_topic", "source", "subgroup", "final_merge_group_id"]
    ].drop_duplicates().copy()
    topic_key["final_merge_group_id"] = topic_key["final_merge_group_id"].astype(str)

    for subgroup, requested in topic_key.groupby("subgroup", sort=True):
        merged_path = merged_root / str(subgroup) / "topic_info.csv"
        if not merged_path.exists():
            continue
        merged_info = read_csv(merged_path).fillna("")
        merged_info["final_merge_group_id"] = merged_info["final_merge_group_id"].astype(str)
        merged_lookup = {
            str(row.final_merge_group_id): row
            for row in merged_info.itertuples(index=False)
            if clean_text(getattr(row, "final_merge_group_id", ""))
        }

        original_path = original_root / str(subgroup) / "topic_info.csv"
        if original_path.exists():
            original_info = read_csv(original_path).fillna("")
            original_lookup = {
                str(row.Topic): row
                for row in original_info.itertuples(index=False)
                if clean_text(getattr(row, "Topic", ""))
            }
        else:
            original_lookup = {}

        for topic in requested.itertuples(index=False):
            group_id = str(topic.final_merge_group_id)
            merged_row = merged_lookup.get(group_id)
            if merged_row is None:
                continue
            member_ids = parse_list(getattr(merged_row, "member_micro_topic_ids_json", ""))
            member_names = parse_list(getattr(merged_row, "member_topic_names_json", ""))
            if not member_ids:
                member_ids = [getattr(merged_row, "Topic", "")]
            for order, component_id in enumerate(member_ids, start=1):
                component_key = str(component_id)
                original = original_lookup.get(component_key)
                fallback_name = clean_text(member_names[order - 1]) if order <= len(member_names) else ""
                rows.append(
                    {
                        "topic_id": topic.topic_id,
                        "macro_topic": topic.macro_topic,
                        "source": topic.source,
                        "subgroup": topic.subgroup,
                        "final_merge_group_id": group_id,
                        "component_micro_topic_id": component_key,
                        "component_topic_name": clean_text(getattr(original, "Name", "")) if original is not None else fallback_name,
                        "component_count": getattr(original, "Count", "") if original is not None else "",
                        "component_representation": compact_representation(
                            getattr(original, "Representation", "") if original is not None else ""
                        ),
                        "component_order": int(order),
                    }
                )

    columns = [
        "topic_id",
        "macro_topic",
        "source",
        "subgroup",
        "final_merge_group_id",
        "component_micro_topic_id",
        "component_topic_name",
        "component_count",
        "component_representation",
        "component_order",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(
        ["macro_topic", "source", "topic_id", "component_order"],
        kind="mergesort",
    )


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


def docs_from_chunk_records(
    row: pd.Series,
    public_safe: bool,
    snippet_chars: int,
    guardian_word_limit: int,
    sec_lookup: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    records = parse_json(row.get("chunk_records_json", ""), [])
    docs: list[dict[str, object]] = []
    if records:
        for record in records:
            source = clean_text(row.get("source", ""))
            source_doc_id = clean_text(record.get("source_doc_id", ""))
            text, truncated, word_count, display_policy = display_text(
                str(record.get("text", "")),
                source,
                public_safe,
                snippet_chars,
                guardian_word_limit,
            )
            if not text:
                continue
            date_text = source_date(source, source_doc_id, record.get("year", row.get("year", "")))
            link = source_link(source, source_doc_id, sec_lookup)
            docs.append(
                {
                    "year": int(record.get("year", row.get("year", 0))),
                    "source": source,
                    "chunk_id": clean_text(record.get("chunk_id", "")),
                    "source_doc_id": source_doc_id,
                    "document_id": source_doc_id,
                    "source_link": link,
                    "source_date": date_text,
                    "micro_topic_probability": record.get("micro_topic_probability", ""),
                    "snippet": text,
                    "is_truncated": bool(truncated),
                    "display_policy": display_policy,
                    "display_word_count": int(word_count),
                    "reference_note": reference_note(source, source_doc_id, date_text, guardian_word_limit, sec_lookup),
                }
            )
    if docs:
        return docs

    for idx, text_col in enumerate(TEXT_COLUMNS, start=1):
        raw_text = row.get(text_col, "")
        source = clean_text(row.get("source", ""))
        text, truncated, word_count, display_policy = display_text(
            str(raw_text),
            source,
            public_safe,
            snippet_chars,
            guardian_word_limit,
        )
        if not text:
            continue
        chunk_id = clean_text(row.get(f"chunk_id_{idx}", ""))
        source_doc_id = ""
        if source == "media" and chunk_id.startswith("guardian::"):
            source_doc_id = chunk_id.replace("guardian::", "", 1).split("::", 1)[0]
        date_text = source_date(source, source_doc_id, row.get("year", ""))
        link = source_link(source, source_doc_id, sec_lookup)
        docs.append(
            {
                "year": int(row.get("year", 0)),
                "source": source,
                "chunk_id": chunk_id,
                "source_doc_id": source_doc_id,
                "document_id": source_doc_id,
                "source_link": link,
                "source_date": date_text,
                "micro_topic_probability": "",
                "snippet": text,
                "is_truncated": bool(truncated),
                "display_policy": display_policy,
                "display_word_count": int(word_count),
                "reference_note": reference_note(source, source_doc_id, date_text, guardian_word_limit, sec_lookup),
            }
        )
    return docs


def collect_docs(
    rows: pd.DataFrame,
    public_safe: bool,
    snippet_chars: int,
    guardian_word_limit: int,
    sec_lookup: dict[str, dict[str, str]] | None = None,
    limit: int = 3,
) -> list[dict[str, object]]:
    docs: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        for doc in docs_from_chunk_records(row, public_safe, snippet_chars, guardian_word_limit, sec_lookup):
            key = (str(doc.get("chunk_id", "")), str(doc.get("snippet", ""))[:80])
            if key in seen:
                continue
            docs.append(doc)
            seen.add(key)
            if len(docs) >= limit:
                return docs
    return docs


def sec_raw_files(sec_raw_dir: Path | None) -> list[Path]:
    if sec_raw_dir is None:
        return []
    return sorted(sec_raw_dir.glob("10k_items_SIC*_with_amends.jsonl"))


def build_sec_lookup(sec_raw_dir: Path | None) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in sec_raw_files(sec_raw_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_doc_id = clean_text(record.get("id", ""))
                if not source_doc_id:
                    continue
                cik = clean_text(record.get("cik", ""))
                lookup[source_doc_id] = {
                    "cik": cik.zfill(10) if cik.isdigit() else cik,
                    "filing_url": clean_text(record.get("filingUrl", "")),
                    "sic_primary": clean_text(record.get("sic_primary", "")),
                    "company": clean_text(record.get("company", "")),
                    "accession": clean_text(record.get("accession", "")),
                }
    return lookup


def build_year_evidence(
    pipeline_root: Path,
    topics: pd.DataFrame,
    public_safe: bool,
    snippet_chars: int,
    guardian_word_limit: int,
    sec_lookup: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
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
                        collect_docs(group, public_safe, snippet_chars, guardian_word_limit, sec_lookup),
                        ensure_ascii=False,
                    ),
                    "public_safe": bool(public_safe),
                }
            )
    return pd.DataFrame(output_rows).sort_values(["topic_id", "year"], kind="mergesort")


def build_relations(pipeline_root: Path, excluded_topic_ids: set[str] | None = None) -> pd.DataFrame:
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
    if excluded_topic_ids:
        relations = relations.loc[
            ~relations["corporate_topic_id"].isin(excluded_topic_ids)
            & ~relations["external_topic_id"].isin(excluded_topic_ids)
        ].copy()
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
    return apply_corporate_topic_label_overrides(relations[columns]).drop_duplicates()


def build_panel_series(pipeline_root: Path, excluded_topic_ids: set[str] | None = None) -> pd.DataFrame:
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
    if excluded_topic_ids:
        series = series.loc[~series["corporate_topic_id"].isin(excluded_topic_ids)].copy()
    return apply_corporate_topic_label_overrides(series)


def build_unpaired_series(pipeline_root: Path, excluded_topic_ids: set[str] | None = None) -> pd.DataFrame:
    path = (
        pipeline_root
        / "outputs"
        / "paper_tables"
        / "corporate_focus_relative_longitudinal_series"
        / "unpaired_external_relative_longitudinal.csv"
    )
    series = read_csv(path)
    series["topic_id"] = [make_topic_id(row.subgroup, row.final_merge_group_id) for row in series.itertuples(index=False)]
    if excluded_topic_ids:
        series = series.loc[~series["topic_id"].isin(excluded_topic_ids)].copy()
    return apply_corporate_topic_label_overrides(series)


def build_source_relation_tables(
    pipeline_root: Path,
    excluded_topic_ids: set[str] | None = None,
    excluded_topic_labels: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    relation_root = pipeline_root / "outputs" / "paper_tables" / "corporate_focus_source_topic_relations"
    files = {
        "source_relation_aggregate_same_year": "aggregate_spearman_peak_summary_table.csv",
        "source_relation_individual_same_year": "individual_spearman_peak_summary_table.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = relation_root / filename
        frame = read_csv(path) if path.exists() else pd.DataFrame()
        if not frame.empty and "relation_id" not in frame.columns:
            level = "aggregate" if "aggregate" in key else "individual"
            frame.insert(0, "relation_id", [f"{level}::{index:04d}" for index in range(1, len(frame) + 1)])
        if not frame.empty and excluded_topic_ids:
            excluded_group_ids = {topic_id.split("::", 1)[1] for topic_id in excluded_topic_ids if "::" in topic_id}
            if "external_final_merge_group_id" in frame.columns:
                frame = frame.loc[~frame["external_final_merge_group_id"].astype(str).isin(excluded_group_ids)].copy()
        if not frame.empty and excluded_topic_labels and "corporate_topic_label" in frame.columns:
            frame = frame.loc[~frame["corporate_topic_label"].map(clean_text).isin(excluded_topic_labels)].copy()
        tables[key] = apply_corporate_topic_label_overrides(frame)
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

    exclusion_decisions_path = args.group_exclusion_decisions or default_group_exclusion_decisions_path(args.pipeline_root)
    excluded_topic_ids, resolved_exclusion_decisions_path, excluded_topics = load_excluded_topics(exclusion_decisions_path)
    excluded_topic_labels = set(excluded_topics.get("topic_label", pd.Series(dtype=str)).map(clean_text)) if not excluded_topics.empty else set()

    topics = build_topics(args.pipeline_root, excluded_topic_ids)
    year_series = build_year_series(args.pipeline_root, topics)
    topic_merge_components = build_topic_merge_components(args.pipeline_root, topics)
    sec_lookup = build_sec_lookup(args.sec_raw_dir)
    year_evidence = build_year_evidence(
        args.pipeline_root,
        topics,
        args.public_safe,
        args.snippet_chars,
        args.guardian_word_limit,
        sec_lookup,
    )
    relations = build_relations(args.pipeline_root, excluded_topic_ids)
    panel_series = build_panel_series(args.pipeline_root, excluded_topic_ids)
    unpaired_series = build_unpaired_series(args.pipeline_root, excluded_topic_ids)
    source_relation_tables = build_source_relation_tables(
        args.pipeline_root,
        excluded_topic_ids,
        excluded_topic_labels,
    )
    interpretations = parse_interpretations(
        args.pipeline_root
        / "outputs"
        / "paper_tables"
        / "corporate_focus_relative_longitudinal_series"
        / "longitudinal_panel_interpretations.md"
    )
    interpretations = apply_interpretation_overrides(interpretations)

    outputs = {
        "topics": str(write_csv(topics, args.output_dir, "topics.csv")),
        "year_series": str(write_csv(year_series, args.output_dir, "year_series.csv")),
        "topic_merge_components": str(write_csv(topic_merge_components, args.output_dir, "topic_merge_components.csv")),
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
        "public_text_policy": "source_aware_research_units",
        "guardian_word_limit": int(args.guardian_word_limit),
        "source_text_policy": {
            "academic": "full abstract/chunk used in the research, with source record link when available",
            "corporate": "full SEC 10-K chunk used in the research, with exact SEC filing link, accession reference, and source SIC when available",
            "media": "Guardian excerpt limited by words, with date and original article link",
        },
        "source_workspace": "external_not_included",
        "sec_lookup_rows": int(len(sec_lookup)),
        "group_exclusion_decisions": str(resolved_exclusion_decisions_path) if resolved_exclusion_decisions_path else "",
        "excluded_topic_count": int(len(excluded_topic_ids)),
        "excluded_topic_ids": sorted(excluded_topic_ids),
        "outputs": outputs,
        "row_counts": {
            "topics": int(len(topics)),
            "year_series": int(len(year_series)),
            "topic_merge_components": int(len(topic_merge_components)),
            "year_evidence": int(len(year_evidence)),
            "relations": int(len(relations)),
            "panel_series": int(len(panel_series)),
            "unpaired_series": int(len(unpaired_series)),
            "source_relation_aggregate_same_year": int(
                len(source_relation_tables["source_relation_aggregate_same_year"])
            ),
            "source_relation_individual_same_year": int(
                len(source_relation_tables["source_relation_individual_same_year"])
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
