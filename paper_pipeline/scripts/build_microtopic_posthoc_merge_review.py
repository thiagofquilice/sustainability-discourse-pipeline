#!/usr/bin/env python3
"""Build post-hoc microtopic merge review candidates and workbook."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_source_microtopic_common import configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_OUTPUT_ROOT,
    PAIR_OUTPUT_ROOT,
    SELECTED_TOPICS_PATH,
    STAGE2_NARRATIVES_PATH,
    YEAR_EVIDENCE_PATH,
    YEAR_SUMMARIES_PATH,
    DISCOURSE_OUTPUT_ROOT,
    ensure_directory,
    json_dumps,
    load_stage2_rows,
    normalize_text,
    parse_json_dict,
    parse_json_list,
    safe_read_csv,
    sanitize_frame_for_excel,
    stringify_list,
    workbook_autofit,
    write_json,
)
from workflow_common import load_sentence_transformer


LOGGER = logging.getLogger("microtopic_posthoc_merge_review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=MERGE_OUTPUT_ROOT)
    parser.add_argument("--selected-topics", type=Path, default=SELECTED_TOPICS_PATH)
    parser.add_argument("--profiles-csv", type=Path, default=PAIR_OUTPUT_ROOT / "microtopic_profiles.csv")
    parser.add_argument(
        "--embedding-metadata-csv",
        type=Path,
        default=PAIR_OUTPUT_ROOT / "microtopic_profile_embedding_metadata.csv",
    )
    parser.add_argument(
        "--embedding-npy",
        type=Path,
        default=PAIR_OUTPUT_ROOT / "microtopic_profile_embeddings.npy",
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=PAIR_OUTPUT_ROOT / "embedding_manifest.json",
    )
    parser.add_argument("--annual-counts-csv", type=Path, default=PAIR_OUTPUT_ROOT / "microtopic_annual_counts.csv")
    parser.add_argument(
        "--denominators-csv",
        type=Path,
        default=PAIR_OUTPUT_ROOT / "source_topic_year_denominators.csv",
    )
    parser.add_argument("--year-summaries-csv", type=Path, default=YEAR_SUMMARIES_PATH)
    parser.add_argument("--year-evidence-csv", type=Path, default=YEAR_EVIDENCE_PATH)
    parser.add_argument("--stage2-csv", type=Path, default=STAGE2_NARRATIVES_PATH)
    parser.add_argument("--pair-join-ready-csv", type=Path, default=PAIR_OUTPUT_ROOT / "pair_join_ready_table.csv")
    parser.add_argument(
        "--same-issue-candidate-csv",
        type=Path,
        default=DISCOURSE_OUTPUT_ROOT / "candidate_pairs_topk.csv",
    )
    parser.add_argument(
        "--same-issue-compare-csv",
        type=Path,
        default=DISCOURSE_OUTPUT_ROOT / "same_issue_discourse_compare.csv",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--narrative-embed-batch-size", type=int, default=32)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def parse_int_list(value: Any) -> list[int]:
    parsed = parse_json_list(value)
    results: list[int] = []
    for item in parsed:
        try:
            results.append(int(item))
        except Exception:
            continue
    return results


def stable_unique_texts(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
        if limit is not None and len(ordered) >= limit:
            break
    return ordered


def build_stage1_digest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.sort_values(["subgroup", "micro_topic_id", "year"])
        .groupby(["subgroup", "micro_topic_id"], dropna=False)
        .agg(
            stage1_year_count=("year", "nunique"),
            stage1_year_min=("year", "min"),
            stage1_year_max=("year", "max"),
            stage1_focus_list=("year_focus_primary", lambda s: stable_unique_texts(s.astype(str).tolist(), limit=4)),
            stage1_frame_list=("year_frame_or_angle", lambda s: stable_unique_texts(s.astype(str).tolist(), limit=3)),
        )
        .reset_index()
    )
    grouped["stage1_focus_digest"] = grouped["stage1_focus_list"].map(lambda x: " || ".join(x))
    grouped["stage1_frame_digest"] = grouped["stage1_frame_list"].map(lambda x: " || ".join(x))
    grouped["stage1_focus_list_json"] = grouped["stage1_focus_list"].map(json_dumps)
    grouped["stage1_frame_list_json"] = grouped["stage1_frame_list"].map(json_dumps)
    return grouped


def build_stage2_text(row: pd.Series) -> str:
    parts = [
        f"Refined label: {normalize_text(row.get('topic_label_refined'))}",
        f"Overall summary: {normalize_text(row.get('overall_summary'))}",
    ]
    for idx in range(1, 5):
        years = normalize_text(row.get(f"phase_{idx}_years"))
        summary = normalize_text(row.get(f"phase_{idx}_summary"))
        if years or summary:
            parts.append(f"Phase {idx} ({years}): {summary}")
    evolution_pattern = normalize_text(row.get("evolution_pattern"))
    if evolution_pattern:
        parts.append(f"Evolution pattern: {evolution_pattern}")
    evidence_note = normalize_text(row.get("evidence_note"))
    if evidence_note:
        parts.append(f"Evidence note: {evidence_note}")
    return "\n".join(part for part in parts if normalize_text(part))


def compute_narrative_embeddings(
    stage2_rows: pd.DataFrame,
    embedding_manifest_path: Path,
    batch_size: int,
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], np.ndarray]]:
    if stage2_rows.empty or not embedding_manifest_path.exists():
        return {}, {}
    manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    model_name = normalize_text(manifest.get("embedding_model"))
    if not model_name:
        return {}, {}
    text_map: dict[tuple[str, int], str] = {}
    ready_text_rows: list[tuple[tuple[str, int], str]] = []
    for row in stage2_rows.itertuples(index=False):
        key = (str(row.subgroup), int(row.micro_topic_id))
        narrative_text = build_stage2_text(pd.Series(row._asdict()))
        text_map[key] = narrative_text
        if narrative_text:
            ready_text_rows.append((key, narrative_text))
    if not ready_text_rows:
        return text_map, {}
    model = load_sentence_transformer(model_name)
    embeddings = model.encode(
        [text for _, text in ready_text_rows],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vector_map = {
        key: np.asarray(vector, dtype="float32") for (key, _), vector in zip(ready_text_rows, embeddings, strict=True)
    }
    return text_map, vector_map


def build_temporal_share_maps(annual_counts: pd.DataFrame, denominators: pd.DataFrame) -> dict[tuple[str, int], dict[str, dict[int, float]]]:
    if annual_counts.empty or denominators.empty:
        return {}
    merged = annual_counts.merge(
        denominators,
        on=["source", "assigned_label", "year"],
        how="left",
    )
    merged["doc_share"] = merged["microtopic_doc_n"] / merged["yes_doc_n"].replace({0: np.nan})
    merged["chunk_share"] = merged["microtopic_chunk_n"] / merged["yes_chunk_n"].replace({0: np.nan})
    share_map: dict[tuple[str, int], dict[str, dict[int, float]]] = {}
    for row in merged.itertuples(index=False):
        key = (str(row.subgroup), int(row.micro_topic_id))
        share_map.setdefault(key, {"doc": {}, "chunk": {}})
        if pd.notna(row.doc_share):
            share_map[key]["doc"][int(row.year)] = float(row.doc_share)
        if pd.notna(row.chunk_share):
            share_map[key]["chunk"][int(row.year)] = float(row.chunk_share)
    return share_map


def series_correlation(map_a: dict[int, float], map_b: dict[int, float]) -> float | None:
    overlap_years = sorted(set(map_a).intersection(map_b))
    if len(overlap_years) < 2:
        return None
    series_a = pd.Series([map_a[year] for year in overlap_years], dtype=float)
    series_b = pd.Series([map_b[year] for year in overlap_years], dtype=float)
    if series_a.nunique(dropna=True) < 2 or series_b.nunique(dropna=True) < 2:
        return None
    corr = series_a.corr(series_b)
    if pd.isna(corr):
        return None
    return float(corr)


def parse_hierarchy_topics(value: Any) -> set[int]:
    values = parse_int_list(value)
    return set(values)


def build_hierarchy_maps(selected: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    subgroup_rows: dict[str, list[dict[str, Any]]] = {}
    for subgroup in sorted(selected["subgroup"].astype(str).unique()):
        path = (
            Path("paper_pipeline/outputs/bertopic_micro_unsupervised")
            / subgroup
            / "hierarchical_topics.csv"
        )
        frame = safe_read_csv(path)
        if frame.empty:
            subgroup_rows[subgroup] = []
            continue
        records: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            topics = parse_hierarchy_topics(getattr(row, "Topics", ""))
            if len(topics) < 2:
                continue
            records.append(
                {
                    "topics": topics,
                    "distance": float(getattr(row, "Distance")),
                    "parent_name": normalize_text(getattr(row, "Parent_Name", "")),
                    "parent_id": getattr(row, "Parent_ID", None),
                }
            )
        subgroup_rows[subgroup] = records
    return subgroup_rows


def hierarchy_pair_stats(topic_a: int, topic_b: int, hierarchy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {
        "hierarchical_min_distance": np.nan,
        "hierarchical_closest_parent_name": "",
        "hierarchical_closest_parent_size": np.nan,
        "hierarchical_found": False,
    }
    best_distance = None
    for record in hierarchy_rows:
        topics = record["topics"]
        if topic_a in topics and topic_b in topics:
            distance = float(record["distance"])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = {
                    "hierarchical_min_distance": distance,
                    "hierarchical_closest_parent_name": record["parent_name"],
                    "hierarchical_closest_parent_size": int(len(topics)),
                    "hierarchical_found": True,
                }
    return best


def build_cross_source_contexts(
    strict_pairs: pd.DataFrame,
    same_issue_candidates: pd.DataFrame,
    same_issue_compare: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, set[str]]]:
    contexts: dict[tuple[str, int], dict[str, set[str]]] = {}

    def ensure(key: tuple[str, int]) -> dict[str, set[str]]:
        if key not in contexts:
            contexts[key] = {
                "strict_partner_keys": set(),
                "strict_precedence_labels": set(),
                "strict_partner_sources": set(),
                "same_issue_partner_keys": set(),
                "same_issue_function_labels": set(),
                "same_issue_relation_labels": set(),
                "same_issue_assessment_labels": set(),
            }
        return contexts[key]

    for row in strict_pairs.itertuples(index=False):
        key_a = (str(row.subgroup_a), int(row.microtopic_id_a))
        key_b = (str(row.subgroup_b), int(row.microtopic_id_b))
        ctx_a = ensure(key_a)
        ctx_b = ensure(key_b)
        partner_a = f"{row.subgroup_b}:{int(row.microtopic_id_b)}"
        partner_b = f"{row.subgroup_a}:{int(row.microtopic_id_a)}"
        ctx_a["strict_partner_keys"].add(partner_a)
        ctx_b["strict_partner_keys"].add(partner_b)
        label = normalize_text(getattr(row, "precedence_type_label", ""))
        if label:
            ctx_a["strict_precedence_labels"].add(label)
            ctx_b["strict_precedence_labels"].add(label)
        ctx_a["strict_partner_sources"].add(str(row.source_b))
        ctx_b["strict_partner_sources"].add(str(row.source_a))

    compare_map: dict[str, dict[str, Any]] = {}
    if not same_issue_compare.empty and "pair_id" in same_issue_compare.columns:
        compare_map = {
            str(row["pair_id"]): row.to_dict()
            for _, row in same_issue_compare.iterrows()
        }

    for row in same_issue_candidates.itertuples(index=False):
        key_a = (str(row.subgroup_a), int(row.microtopic_id_a))
        key_b = (str(row.subgroup_b), int(row.microtopic_id_b))
        ctx_a = ensure(key_a)
        ctx_b = ensure(key_b)
        partner_a = f"{row.subgroup_b}:{int(row.microtopic_id_b)}"
        partner_b = f"{row.subgroup_a}:{int(row.microtopic_id_a)}"
        ctx_a["same_issue_partner_keys"].add(partner_a)
        ctx_b["same_issue_partner_keys"].add(partner_b)
        compare_row = compare_map.get(str(row.pair_id))
        if compare_row:
            label_a = normalize_text(compare_row.get("function_label_a"))
            label_b = normalize_text(compare_row.get("function_label_b"))
            relation = normalize_text(compare_row.get("discourse_function_relation"))
            assessment = normalize_text(compare_row.get("same_issue_assessment"))
            if label_a:
                ctx_a["same_issue_function_labels"].add(label_a)
            if label_b:
                ctx_b["same_issue_function_labels"].add(label_b)
            if relation:
                ctx_a["same_issue_relation_labels"].add(relation)
                ctx_b["same_issue_relation_labels"].add(relation)
            if assessment:
                ctx_a["same_issue_assessment_labels"].add(assessment)
                ctx_b["same_issue_assessment_labels"].add(assessment)
    return contexts


def overlap_stats(items_a: list[str], items_b: list[str]) -> tuple[int, float, list[str]]:
    set_a = {normalize_text(item).lower() for item in items_a if normalize_text(item)}
    set_b = {normalize_text(item).lower() for item in items_b if normalize_text(item)}
    shared = sorted(set_a.intersection(set_b))
    union = set_a.union(set_b)
    jaccard = float(len(shared) / len(union)) if union else np.nan
    return len(shared), jaccard, shared


def build_roster(
    selected: pd.DataFrame,
    profiles: pd.DataFrame,
    stage1_digest: pd.DataFrame,
    stage2_rows: pd.DataFrame,
    narrative_text_map: dict[tuple[str, int], str],
) -> pd.DataFrame:
    roster = selected.merge(
        profiles[
            [
                "subgroup",
                "microtopic_id",
                "microtopic_label_clean",
                "top_terms",
                "representative_chunks",
                "document_count",
                "chunk_count",
                "years_present",
                "profile_text",
            ]
        ].rename(columns={"microtopic_id": "micro_topic_id"}),
        on=["subgroup", "micro_topic_id"],
        how="left",
    )
    roster = roster.merge(stage1_digest, on=["subgroup", "micro_topic_id"], how="left")
    if not stage2_rows.empty:
        stage2_subset = stage2_rows[
            [
                "subgroup",
                "micro_topic_id",
                "topic_label_refined",
                "overall_summary",
                "phase_1_years",
                "phase_1_summary",
                "phase_2_years",
                "phase_2_summary",
                "phase_3_years",
                "phase_3_summary",
                "phase_4_years",
                "phase_4_summary",
                "evolution_pattern",
                "evidence_note",
                "is_narrative_row",
                "is_narrative_ready",
            ]
        ].copy()
        roster = roster.merge(stage2_subset, on=["subgroup", "micro_topic_id"], how="left")
    else:
        roster["is_narrative_row"] = False
        roster["is_narrative_ready"] = False
    roster["topic_label_refined"] = roster["topic_label_refined"].fillna("")
    roster["overall_summary"] = roster["overall_summary"].fillna("")
    roster["narrative_text"] = roster.apply(
        lambda row: narrative_text_map.get((str(row["subgroup"]), int(row["micro_topic_id"])), ""),
        axis=1,
    )
    roster["top_terms_list"] = roster["top_terms"].map(parse_json_list)
    roster["top_terms_display"] = roster["top_terms_list"].map(lambda items: stringify_list([str(x) for x in items]))
    roster["representative_chunk_preview"] = roster["representative_chunks"].map(
        lambda value: normalize_text((parse_json_list(value) or [""])[0])
    )
    roster["years_present_display"] = roster["years_present"].map(
        lambda value: stringify_list([str(item) for item in parse_json_list(value)], sep=", ")
    )
    roster["manual_merge_group_id"] = ""
    roster["manual_merge_confidence"] = ""
    roster["manual_merge_notes"] = ""
    roster = roster.sort_values(["source", "assigned_label", "subgroup", "micro_topic_id"]).reset_index(drop=True)
    return roster


def build_candidate_pairs(
    selected: pd.DataFrame,
    roster: pd.DataFrame,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    narrative_vectors: dict[tuple[str, int], np.ndarray],
    share_map: dict[tuple[str, int], dict[str, dict[int, float]]],
    hierarchy_map: dict[str, list[dict[str, Any]]],
    contexts: dict[tuple[str, int], dict[str, set[str]]],
    top_k: int,
    min_similarity: float,
) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["embedding_row_index"] = pd.to_numeric(metadata["embedding_row_index"], errors="coerce")
    metadata["microtopic_id"] = pd.to_numeric(metadata["microtopic_id"], errors="coerce")
    metadata = metadata.dropna(subset=["embedding_row_index", "microtopic_id"]).copy()
    metadata["embedding_row_index"] = metadata["embedding_row_index"].astype(int)
    metadata["microtopic_id"] = metadata["microtopic_id"].astype(int)

    roster_map = {
        (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
        for row in roster.itertuples(index=False)
    }
    meta_map = {
        (str(row.subgroup), int(row.microtopic_id)): int(row.embedding_row_index)
        for row in metadata.itertuples(index=False)
    }
    directed_rows: list[dict[str, Any]] = []

    for subgroup, subgroup_rows in selected.groupby("subgroup", dropna=False):
        topic_ids = [int(value) for value in subgroup_rows["micro_topic_id"].tolist()]
        keys = [(str(subgroup), topic_id) for topic_id in topic_ids if (str(subgroup), topic_id) in meta_map]
        if len(keys) < 2:
            continue
        index_rows = [meta_map[key] for key in keys]
        subgroup_embeddings = embeddings[index_rows]
        similarity = np.matmul(subgroup_embeddings, subgroup_embeddings.T)
        for i, key_a in enumerate(keys):
            scores = similarity[i]
            candidates: list[tuple[int, float]] = []
            for j, score in enumerate(scores):
                if i == j:
                    continue
                candidates.append((j, float(score)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            for rank, (j, score) in enumerate(candidates, start=1):
                if rank > top_k and score < min_similarity:
                    continue
                key_b = keys[j]
                directed_rows.append(
                    {
                        "subgroup": str(subgroup),
                        "key_a": key_a,
                        "key_b": key_b,
                        "profile_similarity": score,
                        "rank_from_a": rank,
                        "selected_by_rank": rank <= top_k,
                        "selected_by_threshold": score >= min_similarity,
                    }
                )

    if not directed_rows:
        return pd.DataFrame()

    pair_records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in directed_rows:
        subgroup = row["subgroup"]
        _, micro_a = row["key_a"]
        _, micro_b = row["key_b"]
        left_id, right_id = sorted([micro_a, micro_b])
        pair_key = (subgroup, left_id, right_id)
        record = pair_records.setdefault(
            pair_key,
            {
                "subgroup": subgroup,
                "micro_topic_id_a": left_id,
                "micro_topic_id_b": right_id,
                "profile_similarity": row["profile_similarity"],
                "rank_from_a": np.nan,
                "rank_from_b": np.nan,
                "selected_from_a": False,
                "selected_from_b": False,
            },
        )
        record["profile_similarity"] = max(record["profile_similarity"], row["profile_similarity"])
        if micro_a == left_id:
            record["rank_from_a"] = row["rank_from_a"]
            record["selected_from_a"] = True
        else:
            record["rank_from_b"] = row["rank_from_a"]
            record["selected_from_b"] = True

    candidate_rows: list[dict[str, Any]] = []
    for _, base in sorted(pair_records.items()):
        key_a = (base["subgroup"], int(base["micro_topic_id_a"]))
        key_b = (base["subgroup"], int(base["micro_topic_id_b"]))
        row_a = roster_map[key_a]
        row_b = roster_map[key_b]
        terms_a = [str(item) for item in parse_json_list(row_a.get("top_terms"))]
        terms_b = [str(item) for item in parse_json_list(row_b.get("top_terms"))]
        shared_term_count, term_jaccard, shared_terms = overlap_stats(terms_a, terms_b)
        years_a = parse_int_list(row_a.get("years_present"))
        years_b = parse_int_list(row_b.get("years_present"))
        shared_years = sorted(set(years_a).intersection(years_b))
        union_years = sorted(set(years_a).union(years_b))
        year_jaccard = float(len(shared_years) / len(union_years)) if union_years else np.nan
        shares_a = share_map.get(key_a, {"doc": {}, "chunk": {}})
        shares_b = share_map.get(key_b, {"doc": {}, "chunk": {}})
        doc_corr = series_correlation(shares_a.get("doc", {}), shares_b.get("doc", {}))
        chunk_corr = series_correlation(shares_a.get("chunk", {}), shares_b.get("chunk", {}))
        hierarchy_stats = hierarchy_pair_stats(
            int(base["micro_topic_id_a"]),
            int(base["micro_topic_id_b"]),
            hierarchy_map.get(str(base["subgroup"]), []),
        )

        narrative_similarity = np.nan
        vector_a = narrative_vectors.get(key_a)
        vector_b = narrative_vectors.get(key_b)
        if vector_a is not None and vector_b is not None:
            narrative_similarity = float(np.dot(vector_a, vector_b))

        ctx_a = contexts.get(key_a, {})
        ctx_b = contexts.get(key_b, {})
        shared_strict_partner_keys = sorted(
            ctx_a.get("strict_partner_keys", set()).intersection(ctx_b.get("strict_partner_keys", set()))
        )
        shared_same_issue_partner_keys = sorted(
            ctx_a.get("same_issue_partner_keys", set()).intersection(ctx_b.get("same_issue_partner_keys", set()))
        )
        shared_strict_labels = sorted(
            ctx_a.get("strict_precedence_labels", set()).intersection(ctx_b.get("strict_precedence_labels", set()))
        )
        shared_function_labels = sorted(
            ctx_a.get("same_issue_function_labels", set()).intersection(ctx_b.get("same_issue_function_labels", set()))
        )
        shared_relation_labels = sorted(
            ctx_a.get("same_issue_relation_labels", set()).intersection(ctx_b.get("same_issue_relation_labels", set()))
        )
        shared_issue_assessment = sorted(
            ctx_a.get("same_issue_assessment_labels", set()).intersection(ctx_b.get("same_issue_assessment_labels", set()))
        )
        support_signal_count = int(
            sum(
                [
                    base["profile_similarity"] >= 0.65,
                    pd.notna(narrative_similarity) and float(narrative_similarity) >= 0.60,
                    shared_term_count >= 2,
                    len(shared_years) >= 4,
                    doc_corr is not None and doc_corr >= 0.30,
                    chunk_corr is not None and chunk_corr >= 0.30,
                    bool(shared_strict_partner_keys),
                    bool(shared_same_issue_partner_keys),
                    hierarchy_stats["hierarchical_found"],
                ]
            )
        )
        candidate_rows.append(
            {
                "candidate_pair_id": f"{base['subgroup']}__{int(base['micro_topic_id_a'])}__{int(base['micro_topic_id_b'])}",
                "subgroup": base["subgroup"],
                "source": row_a["source"],
                "assigned_label": row_a["assigned_label"],
                "macro_topic_name": row_a.get("macro_topic_name", ""),
                "micro_topic_id_a": int(base["micro_topic_id_a"]),
                "topic_name_original_a": row_a["topic_name_original"],
                "topic_label_refined_a": row_a.get("topic_label_refined", ""),
                "microtopic_label_clean_a": row_a.get("microtopic_label_clean", ""),
                "document_count_a": row_a.get("document_count", np.nan),
                "chunk_count_a": row_a.get("chunk_count", np.nan),
                "active_year_count_a": row_a.get("active_year_count", np.nan),
                "micro_topic_id_b": int(base["micro_topic_id_b"]),
                "topic_name_original_b": row_b["topic_name_original"],
                "topic_label_refined_b": row_b.get("topic_label_refined", ""),
                "microtopic_label_clean_b": row_b.get("microtopic_label_clean", ""),
                "document_count_b": row_b.get("document_count", np.nan),
                "chunk_count_b": row_b.get("chunk_count", np.nan),
                "active_year_count_b": row_b.get("active_year_count", np.nan),
                "profile_similarity": float(base["profile_similarity"]),
                "rank_from_a": base["rank_from_a"],
                "rank_from_b": base["rank_from_b"],
                "selected_from_a": bool(base["selected_from_a"]),
                "selected_from_b": bool(base["selected_from_b"]),
                "selected_from_both_directions": bool(base["selected_from_a"] and base["selected_from_b"]),
                "narrative_similarity": narrative_similarity,
                "narrative_row_a": bool(row_a.get("is_narrative_row", False)),
                "narrative_row_b": bool(row_b.get("is_narrative_row", False)),
                "narrative_ready_a": bool(row_a.get("is_narrative_ready", False)),
                "narrative_ready_b": bool(row_b.get("is_narrative_ready", False)),
                "shared_top_term_count": shared_term_count,
                "top_term_jaccard": term_jaccard,
                "shared_top_terms_json": json_dumps(shared_terms),
                "shared_year_count": int(len(shared_years)),
                "year_jaccard": year_jaccard,
                "shared_years_json": json_dumps(shared_years),
                "doc_share_correlation": doc_corr,
                "chunk_share_correlation": chunk_corr,
                **hierarchy_stats,
                "shared_strict_cross_source_partner_count": int(len(shared_strict_partner_keys)),
                "shared_strict_cross_source_partner_keys_json": json_dumps(shared_strict_partner_keys),
                "shared_strict_precedence_labels_json": json_dumps(shared_strict_labels),
                "shared_same_issue_partner_count": int(len(shared_same_issue_partner_keys)),
                "shared_same_issue_partner_keys_json": json_dumps(shared_same_issue_partner_keys),
                "shared_discourse_function_labels_json": json_dumps(shared_function_labels),
                "shared_discourse_relation_labels_json": json_dumps(shared_relation_labels),
                "shared_issue_assessment_labels_json": json_dumps(shared_issue_assessment),
                "stage1_focus_digest_a": row_a.get("stage1_focus_digest", ""),
                "stage1_focus_digest_b": row_b.get("stage1_focus_digest", ""),
                "overall_summary_a": row_a.get("overall_summary", ""),
                "overall_summary_b": row_b.get("overall_summary", ""),
                "support_signal_count": support_signal_count,
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame = candidate_frame.sort_values(
        ["subgroup", "support_signal_count", "profile_similarity", "narrative_similarity"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return candidate_frame


def instructions_frame() -> pd.DataFrame:
    rows = [
        {
            "section": "Goal",
            "instruction": "Review candidate pairs only within the same subgroup and decide whether multiple microtopics should be merged analytically after BERTopic.",
        },
        {
            "section": "Decision rule",
            "instruction": "Use manual_merge_group_id in the microtopic_roster sheet as the source of truth. Leave it blank to keep a microtopic as a singleton.",
        },
        {
            "section": "ID format",
            "instruction": "Recommended pattern: academic_T1_G01, academic_T1_G02, media_T4_G01, etc. IDs must not mix subgroups.",
        },
        {
            "section": "When to merge",
            "instruction": "Merge when semantic profile similarity, Stage 1 yearly focus, Stage 2 narratives, temporal overlap, and hierarchical proximity all suggest the same underlying microtopic family.",
        },
        {
            "section": "When to keep separate",
            "instruction": "Keep separate when one microtopic has a clearly distinct trajectory, function, evidence base, or a different substantive focus even if some keywords overlap.",
        },
        {
            "section": "Cross-source context",
            "instruction": "Cross-source signals are supporting evidence only. They can show that two internal microtopics connect to the same broader issue family, but they should not override subgroup coherence.",
        },
        {
            "section": "Workbook use",
            "instruction": "Inspect candidate_pairs for detailed evidence, then record the final merge decision only in microtopic_roster. The pipeline will materialize final_merge_group_id from those roster entries later.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    selected = pd.read_csv(args.selected_topics)
    profiles = pd.read_csv(args.profiles_csv)
    metadata = pd.read_csv(args.embedding_metadata_csv)
    embeddings = np.load(args.embedding_npy)
    annual_counts = pd.read_csv(args.annual_counts_csv)
    denominators = pd.read_csv(args.denominators_csv)
    year_summaries = pd.read_csv(args.year_summaries_csv)
    year_evidence = pd.read_csv(args.year_evidence_csv)
    stage2_rows = load_stage2_rows(include_pending=True)
    strict_pairs = safe_read_csv(args.pair_join_ready_csv)
    same_issue_candidates = safe_read_csv(args.same_issue_candidate_csv)
    same_issue_compare = safe_read_csv(args.same_issue_compare_csv)

    selected["micro_topic_id"] = pd.to_numeric(selected["micro_topic_id"], errors="coerce")
    selected = selected.dropna(subset=["micro_topic_id"]).copy()
    selected["micro_topic_id"] = selected["micro_topic_id"].astype(int)

    stage1_digest = build_stage1_digest(year_summaries)
    LOGGER.info("Loaded %s selected microtopics for merge review.", selected.shape[0])

    narrative_text_map, narrative_vectors = compute_narrative_embeddings(
        stage2_rows=stage2_rows,
        embedding_manifest_path=args.embedding_manifest,
        batch_size=args.narrative_embed_batch_size,
    )
    roster = build_roster(
        selected=selected,
        profiles=profiles,
        stage1_digest=stage1_digest,
        stage2_rows=stage2_rows,
        narrative_text_map=narrative_text_map,
    )
    share_map = build_temporal_share_maps(annual_counts=annual_counts, denominators=denominators)
    hierarchy_map = build_hierarchy_maps(selected)
    contexts = build_cross_source_contexts(
        strict_pairs=strict_pairs,
        same_issue_candidates=same_issue_candidates,
        same_issue_compare=same_issue_compare,
    )
    candidate_pairs = build_candidate_pairs(
        selected=selected,
        roster=roster,
        metadata=metadata,
        embeddings=embeddings,
        narrative_vectors=narrative_vectors,
        share_map=share_map,
        hierarchy_map=hierarchy_map,
        contexts=contexts,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    candidate_csv = args.output_root / "merge_candidate_pairs.csv"
    roster_csv = args.output_root / "microtopic_roster.csv"
    workbook_path = args.output_root / "microtopic_posthoc_merge_review.xlsx"
    manifest_path = args.output_root / "merge_review_manifest.json"
    readme_path = args.output_root / "README_microtopic_posthoc_merge_review.md"
    candidate_summary_csv = args.output_root / "merge_candidate_summary.csv"

    candidate_pairs.to_csv(candidate_csv, index=False)
    roster.to_csv(roster_csv, index=False)
    summary = (
        candidate_pairs.groupby(["subgroup", "source", "assigned_label"], dropna=False)
        .agg(
            candidate_pair_count=("candidate_pair_id", "nunique"),
            max_profile_similarity=("profile_similarity", "max"),
            mean_profile_similarity=("profile_similarity", "mean"),
            mean_support_signal_count=("support_signal_count", "mean"),
        )
        .reset_index()
        .sort_values(["candidate_pair_count", "mean_profile_similarity"], ascending=[False, False])
        if not candidate_pairs.empty
        else pd.DataFrame()
    )
    summary.to_csv(candidate_summary_csv, index=False)

    roster_sheet = roster[
        [
            "subgroup",
            "source",
            "assigned_label",
            "micro_topic_id",
            "topic_name_original",
            "topic_label_refined",
            "overall_summary",
            "years_present_display",
            "document_count",
            "chunk_count",
            "top_terms_display",
            "representative_chunk_preview",
            "stage1_focus_digest",
            "is_narrative_row",
            "is_narrative_ready",
            "manual_merge_group_id",
            "manual_merge_confidence",
            "manual_merge_notes",
        ]
    ].rename(columns={"years_present_display": "years_present"})
    instructions = instructions_frame()

    candidate_pairs_excel = sanitize_frame_for_excel(candidate_pairs)
    roster_sheet_excel = sanitize_frame_for_excel(roster_sheet)
    instructions_excel = sanitize_frame_for_excel(instructions)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        candidate_pairs_excel.to_excel(writer, sheet_name="candidate_pairs", index=False)
        roster_sheet_excel.to_excel(writer, sheet_name="microtopic_roster", index=False)
        instructions_excel.to_excel(writer, sheet_name="instructions", index=False)
        if not summary.empty:
            sanitize_frame_for_excel(summary).to_excel(writer, sheet_name="candidate_summary", index=False)
        workbook_autofit(writer, "candidate_pairs", candidate_pairs_excel)
        workbook_autofit(writer, "microtopic_roster", roster_sheet_excel)
        workbook_autofit(writer, "instructions", instructions_excel, freeze_cell="A2")
        if not summary.empty:
            workbook_autofit(writer, "candidate_summary", sanitize_frame_for_excel(summary))

    readme_lines = [
        "# Post-hoc Microtopic Merge Review",
        "",
        f"- selected microtopics in scope: `{int(selected.shape[0])}`",
        f"- candidate pairs generated: `{int(candidate_pairs.shape[0])}`",
        f"- subgroup count covered: `{int(selected['subgroup'].nunique())}`",
        f"- narrative-ready microtopics available now: `{int(roster['is_narrative_ready'].fillna(False).sum())}`",
        "",
        "## Files",
        f"- workbook: `{workbook_path}`",
        f"- candidate pairs CSV: `{candidate_csv}`",
        f"- roster CSV: `{roster_csv}`",
        "",
        "## Review flow",
        "1. Inspect `candidate_pairs` for evidence-rich pair suggestions within each subgroup.",
        "2. Record the final decision only in `microtopic_roster.manual_merge_group_id`.",
        "3. Leave `manual_merge_group_id` blank for singleton microtopics.",
        "4. After manual review, run the materialization script to build `microtopic_to_merged_group.csv` and the derived merged outputs.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    write_json(
        manifest_path,
        {
            "selected_topics_csv": str(args.selected_topics),
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "row_counts": {
                "selected_microtopics": int(selected.shape[0]),
                "candidate_pairs": int(candidate_pairs.shape[0]),
                "roster_rows": int(roster.shape[0]),
                "narrative_ready_microtopics": int(roster["is_narrative_ready"].fillna(False).sum()),
            },
            "outputs": {
                "candidate_pairs_csv": str(candidate_csv),
                "candidate_summary_csv": str(candidate_summary_csv),
                "microtopic_roster_csv": str(roster_csv),
                "review_workbook": str(workbook_path),
                "readme": str(readme_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
