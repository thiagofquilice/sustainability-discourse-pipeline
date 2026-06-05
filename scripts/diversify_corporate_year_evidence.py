#!/usr/bin/env python3
"""Diversify Streamlit evidence snippets for selected sources.

This post-processing script keeps the original Streamlit data schema intact.
It reorders corporate representative_docs_json lists so that the first three
visible snippets prefer distinct company keys, and media lists so that the
first three visible snippets prefer distinct news articles.

Because the released app data stores only the visible snippets, the script
also reads the upstream micro-topic year evidence file, which contains
additional candidate chunks per component micro-topic/year. Current app
snippets are preserved first; upstream-only candidates are appended to the
candidate pool when needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from build_streamlit_app_data import build_sec_lookup, docs_from_chunk_records


DEFAULT_APP_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "app_data"
DEFAULT_LOOKUP = Path(
    "/home/thiago/data/results/corporate_report_duplicate_audit/07_paper_subset_document_to_report.csv"
)
DEFAULT_PIPELINE_EVIDENCE = Path(
    "/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/"
    "corporate_focus_stage12_colab_drive_with_overrides/data/micro_topic_year_evidence.csv"
)
DEFAULT_SELECTED_TOPICS = Path(
    "/home/thiago/1_Supervised_BERTopic/paper_6topic_discourse_pipeline/outputs/"
    "corporate_focus_stage12_colab_drive_with_overrides/data/selected_micro_topics.csv"
)
DEFAULT_SEC_RAW_DIR = Path("/home/thiago/Topic_modelling_dataset_unico")
DIVERSIFIED_SOURCES = {"corporate", "media"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-data-dir", type=Path, default=DEFAULT_APP_DATA_DIR)
    parser.add_argument("--lookup-csv", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--pipeline-evidence-csv", type=Path, default=DEFAULT_PIPELINE_EVIDENCE)
    parser.add_argument("--selected-topics-csv", type=Path, default=DEFAULT_SELECTED_TOPICS)
    parser.add_argument("--sec-raw-dir", type=Path, default=DEFAULT_SEC_RAW_DIR)
    parser.add_argument("--visible-limit", type=int, default=3)
    return parser.parse_args()


def parse_json_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def source_from_topic_id(topic_id: object) -> str:
    text = str(topic_id or "")
    if "_" in text:
        return text.split("_", 1)[0]
    if "::" in text:
        return text.split("::", 1)[0]
    return ""


def load_company_lookup(path: Path) -> dict[str, str]:
    columns = ["source_doc_id", "company_key"]
    lookup = pd.read_csv(path, usecols=columns).fillna("")
    lookup = lookup.drop_duplicates(subset=["source_doc_id"], keep="first")
    return dict(zip(lookup["source_doc_id"].astype(str), lookup["company_key"].astype(str), strict=False))


def load_app_manifest(app_data_dir: Path) -> dict[str, Any]:
    manifest_path = app_data_dir / "app_data_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def doc_company_key(doc: dict[str, Any], company_lookup: dict[str, str]) -> str:
    source_doc_id = str(doc.get("source_doc_id", "") or doc.get("document_id", "") or "")
    company_key = company_lookup.get(source_doc_id, "")
    return company_key or f"unknown_doc:{source_doc_id}"


def top_source_doc_ids(docs: list[dict[str, Any]], limit: int) -> str:
    values: list[str] = []
    for doc in docs[:limit]:
        values.append(str(doc.get("source_doc_id", "") or doc.get("document_id", "") or ""))
    return "|".join(values)


def diversity_unit_for_source(source: str) -> str:
    if source == "corporate":
        return "company"
    if source == "media":
        return "article"
    return "document"


def doc_diversity_key(source: str, doc: dict[str, Any], company_lookup: dict[str, str]) -> str:
    source_doc_id = str(doc.get("source_doc_id", "") or doc.get("document_id", "") or "")
    if source == "corporate":
        return doc_company_key(doc, company_lookup)
    if source == "media":
        return source_doc_id or f"unknown_chunk:{doc.get('chunk_id', '')}"
    return source_doc_id or f"unknown_chunk:{doc.get('chunk_id', '')}"


def unique_diversity_count(
    source: str,
    docs: list[dict[str, Any]],
    company_lookup: dict[str, str],
    limit: int,
) -> int:
    keys = [
        doc_diversity_key(source, doc, company_lookup)
        for doc in docs[:limit]
        if str(doc.get("source_doc_id", "") or doc.get("document_id", "") or doc.get("chunk_id", ""))
    ]
    return len(set(keys))


def diversity_keys_for_docs(
    source: str,
    docs: list[dict[str, Any]],
    company_lookup: dict[str, str],
    limit: int,
) -> str:
    return "|".join(doc_diversity_key(source, doc, company_lookup) for doc in docs[:limit])


def has_repeated_diversity_key(
    source: str,
    docs: list[dict[str, Any]],
    company_lookup: dict[str, str],
    limit: int,
) -> bool:
    keys = [doc_diversity_key(source, doc, company_lookup) for doc in docs[:limit]]
    return len(keys) != len(set(keys))


def merge_candidate_docs(
    app_docs: list[dict[str, Any]],
    upstream_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    docs = list(app_docs)
    seen_chunk_ids = {str(doc.get("chunk_id", "")) for doc in docs if str(doc.get("chunk_id", ""))}
    added = 0
    for doc in upstream_docs:
        chunk_id = str(doc.get("chunk_id", ""))
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        docs.append(doc)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        added += 1
    return docs, added


def build_upstream_candidate_docs(
    *,
    evidence_csv: Path,
    selected_topics_csv: Path,
    sec_lookup: dict[str, dict[str, str]] | None,
    public_safe: bool,
    snippet_chars: int,
    guardian_word_limit: int,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    selected = pd.read_csv(
        selected_topics_csv,
        usecols=["subgroup", "source", "assigned_label", "micro_topic_id", "final_merge_group_id"],
    ).fillna("")
    selected = selected.loc[selected["source"].astype(str).isin(DIVERSIFIED_SOURCES)].copy()
    selected["micro_topic_id"] = pd.to_numeric(selected["micro_topic_id"], errors="coerce").astype("Int64")

    usecols = [
        "subgroup",
        "source",
        "assigned_label",
        "micro_topic_id",
        "year",
        "chunk_records_json",
    ]
    evidence = pd.read_csv(evidence_csv, usecols=usecols).fillna("")
    evidence = evidence.loc[evidence["source"].astype(str).isin(DIVERSIFIED_SOURCES)].copy()
    evidence["micro_topic_id"] = pd.to_numeric(evidence["micro_topic_id"], errors="coerce").astype("Int64")
    merged = evidence.merge(
        selected,
        on=["subgroup", "source", "assigned_label", "micro_topic_id"],
        how="inner",
    )
    merged["topic_id"] = merged["subgroup"].astype(str) + "::" + merged["final_merge_group_id"].astype(str)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")

    candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (topic_id, year), group in merged.groupby(["topic_id", "year"], sort=False):
        docs: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            docs.extend(
                docs_from_chunk_records(
                    row,
                    public_safe=public_safe,
                    snippet_chars=snippet_chars,
                    guardian_word_limit=guardian_word_limit,
                    sec_lookup=sec_lookup,
                )
            )
        candidates[(str(topic_id), int(year))] = docs
    return candidates


def diversify_docs(
    docs: list[dict[str, Any]],
    source: str,
    company_lookup: dict[str, str],
    visible_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_indices: list[int] = []
    used_keys: set[str] = set()

    for index, doc in enumerate(docs):
        if len(selected_indices) >= min(visible_limit, len(docs)):
            break
        diversity_key = doc_diversity_key(source, doc, company_lookup)
        if diversity_key in used_keys:
            continue
        selected_indices.append(index)
        used_keys.add(diversity_key)

    first_pass_count = len(selected_indices)
    if source != "media":
        for index, _doc in enumerate(docs):
            if len(selected_indices) >= min(visible_limit, len(docs)):
                break
            if index in selected_indices:
                continue
            selected_indices.append(index)

    selected_set = set(selected_indices)
    if source == "media":
        reordered = [docs[index] for index in selected_indices]
        reordered_keys = {doc_diversity_key(source, doc, company_lookup) for doc in reordered}
        for index, doc in enumerate(docs):
            if index in selected_set:
                continue
            diversity_key = doc_diversity_key(source, doc, company_lookup)
            if diversity_key in reordered_keys:
                continue
            reordered.append(doc)
            reordered_keys.add(diversity_key)
    else:
        reordered = [docs[index] for index in selected_indices] + [
            doc for index, doc in enumerate(docs) if index not in selected_set
        ]

    original_top = docs[:visible_limit]
    diverse_top = reordered[:visible_limit]
    original_unique = unique_diversity_count(source, original_top, company_lookup, visible_limit)
    diverse_unique = unique_diversity_count(source, diverse_top, company_lookup, visible_limit)
    available_unique = unique_diversity_count(source, docs, company_lookup, len(docs))
    changed_order = top_source_doc_ids(original_top, visible_limit) != top_source_doc_ids(diverse_top, visible_limit)
    fallback_used = first_pass_count < min(visible_limit, len(docs))
    diversity_unit = diversity_unit_for_source(source)
    if diverse_unique > original_unique:
        outcome = "improved"
    elif len(docs) < 2:
        outcome = "too_few_snippets"
    elif available_unique <= original_unique:
        outcome = f"no_available_{diversity_unit}_gain"
    elif not changed_order:
        outcome = "already_diverse"
    else:
        outcome = "reordered_without_count_gain"

    company_top_original = diversity_keys_for_docs(source, original_top, company_lookup, visible_limit) if source == "corporate" else ""
    company_top_diverse = diversity_keys_for_docs(source, diverse_top, company_lookup, visible_limit) if source == "corporate" else ""
    article_top_original = diversity_keys_for_docs(source, original_top, company_lookup, visible_limit) if source == "media" else ""
    article_top_diverse = diversity_keys_for_docs(source, diverse_top, company_lookup, visible_limit) if source == "media" else ""

    metrics = {
        "n_snippets": len(docs),
        "diversity_unit": diversity_unit,
        "available_unique_units_all_snippets": available_unique,
        "original_unique_units_top3": original_unique,
        "diverse_unique_units_top3": diverse_unique,
        "changed_order": bool(changed_order),
        "fallback_used": bool(fallback_used),
        "original_has_repeated_unit_top3": has_repeated_diversity_key(
            source, original_top, company_lookup, visible_limit
        ),
        "diverse_has_repeated_unit_top3": has_repeated_diversity_key(
            source, diverse_top, company_lookup, visible_limit
        ),
        "improved_unique_units": bool(diverse_unique > original_unique),
        "diversity_outcome": outcome,
        "original_source_doc_ids_top3": top_source_doc_ids(original_top, visible_limit),
        "diverse_source_doc_ids_top3": top_source_doc_ids(diverse_top, visible_limit),
        "original_diversity_keys_top3": diversity_keys_for_docs(source, original_top, company_lookup, visible_limit),
        "diverse_diversity_keys_top3": diversity_keys_for_docs(source, diverse_top, company_lookup, visible_limit),
        "available_unique_companies_all_snippets": available_unique if source == "corporate" else "",
        "original_unique_companies_top3": original_unique if source == "corporate" else "",
        "diverse_unique_companies_top3": diverse_unique if source == "corporate" else "",
        "improved_unique_companies": bool(source == "corporate" and diverse_unique > original_unique),
        "original_company_keys_top3": company_top_original,
        "diverse_company_keys_top3": company_top_diverse,
        "available_unique_articles_all_snippets": available_unique if source == "media" else "",
        "original_unique_articles_top3": original_unique if source == "media" else "",
        "diverse_unique_articles_top3": diverse_unique if source == "media" else "",
        "improved_unique_articles": bool(source == "media" and diverse_unique > original_unique),
        "original_article_ids_top3": article_top_original,
        "diverse_article_ids_top3": article_top_diverse,
    }
    return reordered, metrics


def main() -> None:
    args = parse_args()
    app_data_dir = args.app_data_dir
    year_evidence_path = app_data_dir / "year_evidence.csv"
    original_path = app_data_dir / "year_evidence_original.csv"
    audit_path = app_data_dir / "year_evidence_company_diversity_audit.csv"
    manifest_path = app_data_dir / "company_diverse_evidence_manifest.json"

    if not year_evidence_path.exists():
        raise FileNotFoundError(f"Missing year_evidence.csv: {year_evidence_path}")
    if not args.lookup_csv.exists():
        raise FileNotFoundError(f"Missing company lookup CSV: {args.lookup_csv}")
    if not args.pipeline_evidence_csv.exists():
        raise FileNotFoundError(f"Missing pipeline evidence CSV: {args.pipeline_evidence_csv}")
    if not args.selected_topics_csv.exists():
        raise FileNotFoundError(f"Missing selected topics CSV: {args.selected_topics_csv}")

    if not original_path.exists():
        shutil.copy2(year_evidence_path, original_path)

    original = pd.read_csv(original_path).fillna("")
    updated = original.copy()
    company_lookup = load_company_lookup(args.lookup_csv)
    app_manifest = load_app_manifest(app_data_dir)
    public_safe = bool(app_manifest.get("public_safe", True))
    snippet_chars = int(app_manifest.get("snippet_chars", 360))
    guardian_word_limit = int(app_manifest.get("guardian_word_limit", 300))
    sec_lookup = build_sec_lookup(args.sec_raw_dir if args.sec_raw_dir.exists() else None)
    upstream_candidates = build_upstream_candidate_docs(
        evidence_csv=args.pipeline_evidence_csv,
        selected_topics_csv=args.selected_topics_csv,
        sec_lookup=sec_lookup,
        public_safe=public_safe,
        snippet_chars=snippet_chars,
        guardian_word_limit=guardian_word_limit,
    )

    audit_rows: list[dict[str, Any]] = []
    upstream_added_total = 0
    for index, row in original.iterrows():
        topic_id = row.get("topic_id", "")
        year = row.get("year", "")
        source = source_from_topic_id(topic_id)
        docs = parse_json_list(row.get("representative_docs_json", ""))
        if source not in DIVERSIFIED_SOURCES:
            audit_rows.append(
                {
                    "topic_id": topic_id,
                    "year": year,
                    "source": source,
                    "original_n_snippets": len(docs),
                    "upstream_candidate_snippets_added": 0,
                    "n_snippets": len(docs),
                    "candidate_pool_n_snippets": len(docs),
                    "retained_upstream_snippets_in_final_json": 0,
                    "diversity_unit": "",
                    "available_unique_units_all_snippets": "",
                    "original_unique_units_top3": "",
                    "diverse_unique_units_top3": "",
                    "available_unique_companies_all_snippets": "",
                    "original_unique_companies_top3": "",
                    "diverse_unique_companies_top3": "",
                    "available_unique_articles_all_snippets": "",
                    "original_unique_articles_top3": "",
                    "diverse_unique_articles_top3": "",
                    "changed_order": False,
                    "fallback_used": False,
                    "improved_unique_units": False,
                    "improved_unique_companies": False,
                    "improved_unique_articles": False,
                    "diversity_outcome": "not_diversified_source",
                    "original_source_doc_ids_top3": top_source_doc_ids(docs, args.visible_limit),
                    "diverse_source_doc_ids_top3": top_source_doc_ids(docs, args.visible_limit),
                    "original_diversity_keys_top3": "",
                    "diverse_diversity_keys_top3": "",
                    "original_company_keys_top3": "",
                    "diverse_company_keys_top3": "",
                    "original_article_ids_top3": "",
                    "diverse_article_ids_top3": "",
                }
            )
            continue

        upstream_key = (str(topic_id), int(year) if str(year).strip().isdigit() else -1)
        candidate_docs, upstream_added = merge_candidate_docs(docs, upstream_candidates.get(upstream_key, []))
        upstream_added_total += upstream_added
        reordered_docs, metrics = diversify_docs(candidate_docs, source, company_lookup, args.visible_limit)

        original_chunk_ids = {str(doc.get("chunk_id", "")) for doc in docs if str(doc.get("chunk_id", ""))}
        top_docs = reordered_docs[: args.visible_limit]
        top_chunk_ids = {str(doc.get("chunk_id", "")) for doc in top_docs if str(doc.get("chunk_id", ""))}
        if source == "media":
            final_docs = []
            seen_final_keys: set[str] = set()
            for doc in top_docs + docs:
                diversity_key = doc_diversity_key(source, doc, company_lookup)
                if diversity_key in seen_final_keys:
                    continue
                final_docs.append(doc)
                seen_final_keys.add(diversity_key)
        else:
            final_docs = top_docs + [
                doc for doc in docs if str(doc.get("chunk_id", "")) not in top_chunk_ids
            ]
        retained_upstream = sum(
            1 for doc in top_docs if str(doc.get("chunk_id", "")) not in original_chunk_ids
        )

        metrics["candidate_pool_n_snippets"] = metrics.pop("n_snippets")
        metrics["n_snippets"] = len(final_docs)
        metrics["retained_upstream_snippets_in_final_json"] = int(retained_upstream)
        updated.at[index, "representative_docs_json"] = json.dumps(final_docs, ensure_ascii=False)
        audit_rows.append(
            {
                "topic_id": topic_id,
                "year": int(year) if str(year).strip().isdigit() else year,
                "source": source,
                "original_n_snippets": len(docs),
                "upstream_candidate_snippets_added": upstream_added,
                **metrics,
            }
        )

    updated.to_csv(year_evidence_path, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False)

    original_academic = original.loc[original["topic_id"].map(source_from_topic_id).eq("academic")].reset_index(drop=True)
    updated_academic = updated.loc[updated["topic_id"].map(source_from_topic_id).eq("academic")].reset_index(drop=True)
    corporate_audit = audit.loc[audit["source"].astype(str).eq("corporate")].copy()
    corporate_three = corporate_audit.loc[corporate_audit["original_n_snippets"] >= args.visible_limit].copy()
    media_audit = audit.loc[audit["source"].astype(str).eq("media")].copy()
    media_three = media_audit.loc[media_audit["original_n_snippets"] >= args.visible_limit].copy()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_data_dir": str(app_data_dir),
        "year_evidence": str(year_evidence_path),
        "year_evidence_original": str(original_path),
        "company_lookup_csv": str(args.lookup_csv),
        "pipeline_evidence_csv": str(args.pipeline_evidence_csv),
        "selected_topics_csv": str(args.selected_topics_csv),
        "sec_raw_dir": str(args.sec_raw_dir),
        "sec_lookup_rows": int(len(sec_lookup)),
        "candidate_pool_note": "Current app snippets are preserved first; upstream corporate/media evidence candidates are appended when absent.",
        "visible_limit": int(args.visible_limit),
        "upstream_candidate_snippets_added_to_pool": int(upstream_added_total),
        "upstream_snippets_retained_in_final_json": int(audit["retained_upstream_snippets_in_final_json"].sum()),
        "corporate_upstream_snippets_retained_in_final_json": int(
            corporate_audit["retained_upstream_snippets_in_final_json"].sum()
        ),
        "media_upstream_snippets_retained_in_final_json": int(
            media_audit["retained_upstream_snippets_in_final_json"].sum()
        ),
        "columns_unchanged": list(original.columns) == list(updated.columns),
        "row_count_unchanged": int(len(original)) == int(len(updated)),
        "academic_rows_identical": bool(original_academic.equals(updated_academic)),
        "corporate_rows": int(len(corporate_audit)),
        "corporate_rows_with_3_or_more_snippets": int(len(corporate_three)),
        "corporate_rows_repeated_company_top3_before": int(
            corporate_three["original_has_repeated_unit_top3"].sum()
        ),
        "corporate_rows_repeated_company_top3_after": int(
            corporate_three["diverse_has_repeated_unit_top3"].sum()
        ),
        "corporate_rows_improved_unique_company_count": int(corporate_audit["improved_unique_companies"].sum()),
        "corporate_rows_changed_order": int(corporate_audit["changed_order"].sum()),
        "media_rows": int(len(media_audit)),
        "media_rows_with_3_or_more_snippets": int(len(media_three)),
        "media_rows_repeated_article_top3_before": int(
            media_three["original_has_repeated_unit_top3"].sum()
        ),
        "media_rows_repeated_article_top3_after": int(
            media_three["diverse_has_repeated_unit_top3"].sum()
        ),
        "media_rows_improved_unique_article_count": int(media_audit["improved_unique_articles"].sum()),
        "media_rows_changed_order": int(media_audit["changed_order"].sum()),
        "media_improved_from_1_to_2_or_3_articles": int(
            (
                (media_audit["original_unique_articles_top3"] == 1)
                & (media_audit["diverse_unique_articles_top3"] > 1)
            ).sum()
        ),
        "media_improved_from_2_to_3_articles": int(
            (
                (media_audit["original_unique_articles_top3"] == 2)
                & (media_audit["diverse_unique_articles_top3"] >= 3)
            ).sum()
        ),
        "improved_from_1_to_2_or_3": int(
            (
                (corporate_audit["original_unique_companies_top3"] == 1)
                & (corporate_audit["diverse_unique_companies_top3"] > 1)
            ).sum()
        ),
        "improved_from_2_to_3": int(
            (
                (corporate_audit["original_unique_companies_top3"] == 2)
                & (corporate_audit["diverse_unique_companies_top3"] >= 3)
            ).sum()
        ),
        "corporate_outcome_counts": {
            str(key): int(value)
            for key, value in corporate_audit["diversity_outcome"].value_counts().sort_index().to_dict().items()
        },
        "media_outcome_counts": {
            str(key): int(value)
            for key, value in media_audit["diversity_outcome"].value_counts().sort_index().to_dict().items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
