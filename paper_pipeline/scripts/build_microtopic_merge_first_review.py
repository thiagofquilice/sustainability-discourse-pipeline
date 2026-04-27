#!/usr/bin/env python3
"""Build an early merge review workbook directly from unsupervised BERTopic subgroup outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_cross_source_microtopic_profiles import representative_chunk_map
from cross_source_microtopic_common import (
    MACRO_TOPIC_ORDER,
    SOURCE_ORDER,
    build_profile_text,
    clean_microtopic_label,
    configure_logging,
    parse_topic_terms,
    read_json,
    subgroup_macro_topic_name,
    truncate_text,
)
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    RAW_MICRO_ROOT_MULTIASPECT,
    ensure_directory,
    json_dumps,
    normalize_text,
    parse_json_list,
    sanitize_frame_for_excel,
    safe_read_csv,
    stringify_list,
    workbook_autofit,
    write_json,
)
from workflow_common import detect_embedding_model_name, load_config, load_sentence_transformer


CONFIG_PATH = Path(
    "paper_pipeline/config/paper_6topic_pipeline_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=RAW_MICRO_ROOT_MULTIASPECT)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_REVIEW_MULTIASPECT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-representative-chunks", type=int, default=5)
    parser.add_argument("--max-representative-chars", type=int, default=480)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def parse_int_list(value: Any) -> list[int]:
    results: list[int] = []
    for item in parse_json_list(value):
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


def parse_aspect_terms(value: Any) -> list[str]:
    parsed = parse_json_list(value)
    extracted: list[str] = []
    if parsed:
        for item in parsed:
            term = ""
            if isinstance(item, str):
                term = item
            elif isinstance(item, (list, tuple)) and item:
                term = item[0]
            elif isinstance(item, dict):
                term = item.get("term") or item.get("text") or item.get("label") or ""
            cleaned = normalize_text(term)
            if cleaned:
                extracted.append(cleaned)
        if extracted:
            return stable_unique_texts(extracted)
    return parse_topic_terms(value)


def build_semantic_profile_text(
    microtopic_label: str,
    main_terms: list[str],
    pos_terms: list[str],
    aspect2_terms: list[str],
    representative_chunks: list[str],
) -> str:
    lines = [
        f"Microtopic label: {microtopic_label or 'Unknown'}",
        "Main terms: " + ("; ".join(main_terms) if main_terms else "None available"),
        "POS terms: " + ("; ".join(pos_terms) if pos_terms else "None available"),
        "Aspect2 terms: " + ("; ".join(aspect2_terms) if aspect2_terms else "None available"),
        "Representative chunks:",
    ]
    if representative_chunks:
        for idx, chunk in enumerate(representative_chunks, start=1):
            lines.append(f"[{idx}] {truncate_text(chunk, 480)}")
    else:
        lines.append("[1] No representative chunk available.")
    return "\n".join(lines)


def output_path(output_root: Path, stem: str, suffix: str) -> Path:
    if "multiaspect" in output_root.name.lower():
        return output_root / f"{stem}_multiaspect{suffix}"
    return output_root / f"{stem}{suffix}"


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
    return set(parse_int_list(value))


def build_hierarchy_rows(subgroup_dir: Path) -> list[dict[str, Any]]:
    frame = safe_read_csv(subgroup_dir / "hierarchical_topics.csv")
    if frame.empty:
        return []
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
    return records


def hierarchy_pair_stats(topic_a: int, topic_b: int, hierarchy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = {
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


def overlap_stats(items_a: list[str], items_b: list[str]) -> tuple[int, float, list[str]]:
    set_a = {normalize_text(item).lower() for item in items_a if normalize_text(item)}
    set_b = {normalize_text(item).lower() for item in items_b if normalize_text(item)}
    shared = sorted(set_a.intersection(set_b))
    union = set_a.union(set_b)
    jaccard = float(len(shared) / len(union)) if union else np.nan
    return len(shared), jaccard, shared


def build_subgroup_inputs(
    subgroup_dir: Path,
    max_representative_chunks: int,
    max_representative_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(subgroup_dir / "manifest.json")
    document_topics = pd.read_parquet(subgroup_dir / "document_topics.parquet")
    topic_info = pd.read_csv(subgroup_dir / "topic_info.csv")
    representative_docs = pd.read_csv(subgroup_dir / "representative_docs.csv")
    topic_aspects = safe_read_csv(subgroup_dir / "topic_aspects.csv")

    document_topics["micro_topic_id"] = pd.to_numeric(document_topics["micro_topic_id"], errors="coerce")
    document_topics["year"] = pd.to_numeric(document_topics["year"], errors="coerce")
    document_topics = document_topics.dropna(subset=["micro_topic_id", "year"]).copy()
    document_topics["micro_topic_id"] = document_topics["micro_topic_id"].astype(int)
    document_topics["year"] = document_topics["year"].astype(int)
    document_topics = document_topics.loc[document_topics["micro_topic_id"] != -1].copy()

    topic_info["Topic"] = pd.to_numeric(topic_info["Topic"], errors="coerce")
    topic_info = topic_info.dropna(subset=["Topic"]).copy()
    topic_info["Topic"] = topic_info["Topic"].astype(int)
    topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()

    if not topic_aspects.empty:
        topic_aspects["micro_topic_id"] = pd.to_numeric(topic_aspects["micro_topic_id"], errors="coerce")
        topic_aspects = topic_aspects.dropna(subset=["micro_topic_id"]).copy()
        topic_aspects["micro_topic_id"] = topic_aspects["micro_topic_id"].astype(int)
        topic_aspects = topic_aspects.loc[topic_aspects["micro_topic_id"] != -1].copy()

    hierarchy_rows = build_hierarchy_rows(subgroup_dir)
    return document_topics, topic_info, topic_aspects, representative_docs, manifest, hierarchy_rows


def build_roster_and_support(
    micro_root: Path,
    max_representative_chunks: int,
    max_representative_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    roster_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    hierarchy_map: dict[str, list[dict[str, Any]]] = {}

    subgroup_dirs = sorted(path for path in micro_root.iterdir() if path.is_dir() and path.name != "summary")
    for subgroup_dir in subgroup_dirs:
        document_topics, topic_info, topic_aspects, representative_docs, manifest, hierarchy_rows = build_subgroup_inputs(
            subgroup_dir=subgroup_dir,
            max_representative_chunks=max_representative_chunks,
            max_representative_chars=max_representative_chars,
        )
        if document_topics.empty or topic_info.empty:
            hierarchy_map[subgroup_dir.name] = hierarchy_rows
            continue

        macro_topic_name = subgroup_macro_topic_name(document_topics, manifest)
        rep_map = representative_chunk_map(
            representative_docs=representative_docs,
            document_topics=document_topics,
            max_chunks=max_representative_chunks,
            max_chars=max_representative_chars,
        )
        topic_meta = topic_info.set_index("Topic").to_dict(orient="index")
        topic_aspect_meta = (
            topic_aspects.set_index("micro_topic_id").to_dict(orient="index")
            if not topic_aspects.empty
            else {}
        )

        for micro_topic_id, group in document_topics.groupby("micro_topic_id", dropna=False):
            topic_record = topic_meta.get(int(micro_topic_id), {})
            if not topic_record:
                continue
            year_counts = (
                group.groupby("year", dropna=False)
                .agg(doc_n=("source_doc_id", "nunique"), chunk_n=("chunk_id", "size"))
                .reset_index()
                .sort_values("year")
            )
            years_present = year_counts["year"].astype(int).tolist()
            topic_aspect_record = topic_aspect_meta.get(int(micro_topic_id), {})
            main_terms = parse_aspect_terms(topic_aspect_record.get("main_terms_json")) or parse_topic_terms(
                topic_record.get("Representation")
            )
            pos_terms = parse_aspect_terms(topic_aspect_record.get("pos_terms_json")) or parse_topic_terms(
                topic_record.get("POS")
            )
            aspect2_terms = parse_aspect_terms(topic_aspect_record.get("aspect2_terms_json")) or parse_topic_terms(
                topic_record.get("Aspect2")
            )
            microtopic_label = str(topic_record.get("Name", f"{micro_topic_id}")).strip()
            representative_chunks = rep_map.get(int(micro_topic_id), [])

            clean_label = clean_microtopic_label(microtopic_label) or microtopic_label
            main_terms_display = normalize_text(topic_aspect_record.get("main_terms_display")) or stringify_list(
                [str(item) for item in main_terms]
            )
            pos_terms_display = normalize_text(topic_aspect_record.get("pos_terms_display")) or stringify_list(
                [str(item) for item in pos_terms]
            )
            aspect2_terms_display = normalize_text(topic_aspect_record.get("aspect2_terms_display")) or stringify_list(
                [str(item) for item in aspect2_terms]
            )
            representative_chunk_preview = normalize_text(representative_chunks[0]) if representative_chunks else ""
            row = {
                "subgroup": subgroup_dir.name,
                "source": str(group["source"].iloc[0]),
                "assigned_label": str(group["assigned_label"].iloc[0]),
                "macro_topic_name": macro_topic_name,
                "micro_topic_id": int(micro_topic_id),
                "topic_name_original": microtopic_label,
                "topic_label_refined": "",
                "overall_summary": "",
                "microtopic_label_clean": clean_label,
                "topic_size": int(topic_record.get("Count", group.shape[0])),
                "document_count": int(group["source_doc_id"].nunique()),
                "chunk_count": int(group.shape[0]),
                "active_year_count": int(len(years_present)),
                "first_year": int(min(years_present)) if years_present else None,
                "last_year": int(max(years_present)) if years_present else None,
                "years_present": json_dumps(years_present),
                "main_terms_json": json_dumps(main_terms),
                "main_terms_display": main_terms_display,
                "pos_terms_json": json_dumps(pos_terms),
                "pos_terms_display": pos_terms_display,
                "aspect2_terms_json": json_dumps(aspect2_terms),
                "aspect2_terms_display": aspect2_terms_display,
                "top_terms": json_dumps(main_terms),
                "top_terms_display": main_terms_display,
                "representative_chunks": json_dumps(representative_chunks),
                "representative_chunk_preview": representative_chunk_preview,
                "profile_text": build_semantic_profile_text(
                    microtopic_label=clean_label,
                    main_terms=main_terms,
                    pos_terms=pos_terms,
                    aspect2_terms=aspect2_terms,
                    representative_chunks=representative_chunks,
                ),
                "manual_merge_group_id": "",
                "manual_merge_confidence": "",
                "manual_merge_notes": "",
            }
            roster_rows.append(row)
            profile_rows.append(
                {
                    "subgroup": row["subgroup"],
                    "source": row["source"],
                    "assigned_label": row["assigned_label"],
                    "macro_topic_name": row["macro_topic_name"],
                    "microtopic_id": row["micro_topic_id"],
                    "microtopic_label": row["topic_name_original"],
                    "microtopic_label_clean": row["microtopic_label_clean"],
                    "main_terms_json": row["main_terms_json"],
                    "pos_terms_json": row["pos_terms_json"],
                    "aspect2_terms_json": row["aspect2_terms_json"],
                    "representative_chunks": row["representative_chunks"],
                    "document_count": row["document_count"],
                    "chunk_count": row["chunk_count"],
                    "years_present": row["years_present"],
                    "active_year_count": row["active_year_count"],
                    "first_year": row["first_year"],
                    "last_year": row["last_year"],
                    "profile_text": row["profile_text"],
                }
            )
        hierarchy_map[subgroup_dir.name] = hierarchy_rows

    roster = pd.DataFrame(roster_rows)
    profiles = pd.DataFrame(profile_rows)
    if not roster.empty:
        roster["source"] = pd.Categorical(roster["source"], categories=SOURCE_ORDER, ordered=True)
        roster["assigned_label"] = pd.Categorical(roster["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        roster = roster.sort_values(["source", "assigned_label", "subgroup", "micro_topic_id"]).reset_index(drop=True)
    if not profiles.empty:
        profiles["source"] = pd.Categorical(profiles["source"], categories=SOURCE_ORDER, ordered=True)
        profiles["assigned_label"] = pd.Categorical(profiles["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        profiles = profiles.sort_values(["source", "assigned_label", "microtopic_id"]).reset_index(drop=True)
    return roster, profiles, hierarchy_map


def compute_profile_embeddings(
    profiles: pd.DataFrame,
    batch_size: int,
    model_name: str | None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if profiles.empty:
        return pd.DataFrame(), np.empty((0, 0), dtype="float32"), model_name or ""
    config = load_config(CONFIG_PATH)
    selected_model = model_name or detect_embedding_model_name(config)
    model = load_sentence_transformer(selected_model)
    texts = profiles["profile_text"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    metadata = profiles.copy()
    metadata.insert(0, "embedding_row_index", np.arange(len(metadata), dtype=int))
    return metadata, embeddings, selected_model


def build_candidate_pairs(
    roster: pd.DataFrame,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    hierarchy_map: dict[str, list[dict[str, Any]]],
    top_k: int,
    min_similarity: float,
) -> pd.DataFrame:
    if roster.empty or metadata.empty or embeddings.size == 0:
        return pd.DataFrame()

    roster_map = {
        (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
        for row in roster.itertuples(index=False)
    }
    meta_map = {
        (str(row.subgroup), int(row.microtopic_id)): int(row.embedding_row_index)
        for row in metadata.itertuples(index=False)
    }

    directed_rows: list[dict[str, Any]] = []
    for subgroup, subgroup_rows in roster.groupby("subgroup", dropna=False):
        keys = [
            (str(subgroup), int(topic_id))
            for topic_id in subgroup_rows["micro_topic_id"].astype(int).tolist()
            if (str(subgroup), int(topic_id)) in meta_map
        ]
        if len(keys) < 2:
            continue
        subgroup_embeddings = embeddings[[meta_map[key] for key in keys]]
        similarity = subgroup_embeddings @ subgroup_embeddings.T
        for idx_a, key_a in enumerate(keys):
            candidates: list[tuple[int, float]] = []
            for idx_b, score in enumerate(similarity[idx_a]):
                if idx_a == idx_b:
                    continue
                candidates.append((idx_b, float(score)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            kept = 0
            for idx_b, score in candidates:
                if score < min_similarity:
                    continue
                kept += 1
                if kept > top_k:
                    break
                key_b = keys[idx_b]
                directed_rows.append(
                    {
                        "subgroup": str(subgroup),
                        "key_a": key_a,
                        "key_b": key_b,
                        "profile_similarity": score,
                        "rank_from_a": kept,
                    }
                )

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
        main_terms_a = [str(item) for item in parse_aspect_terms(row_a.get("main_terms_json"))]
        main_terms_b = [str(item) for item in parse_aspect_terms(row_b.get("main_terms_json"))]
        pos_terms_a = [str(item) for item in parse_aspect_terms(row_a.get("pos_terms_json"))]
        pos_terms_b = [str(item) for item in parse_aspect_terms(row_b.get("pos_terms_json"))]
        aspect2_terms_a = [str(item) for item in parse_aspect_terms(row_a.get("aspect2_terms_json"))]
        aspect2_terms_b = [str(item) for item in parse_aspect_terms(row_b.get("aspect2_terms_json"))]
        shared_main_term_count, main_term_jaccard, shared_main_terms = overlap_stats(main_terms_a, main_terms_b)
        shared_pos_term_count, pos_term_jaccard, shared_pos_terms = overlap_stats(pos_terms_a, pos_terms_b)
        shared_aspect2_term_count, aspect2_term_jaccard, shared_aspect2_terms = overlap_stats(
            aspect2_terms_a, aspect2_terms_b
        )
        hierarchy_stats = hierarchy_pair_stats(
            int(base["micro_topic_id_a"]),
            int(base["micro_topic_id_b"]),
            hierarchy_map.get(str(base["subgroup"]), []),
        )
        semantic_support_count = int(
            sum(
                [
                    base["profile_similarity"] >= 0.65,
                    shared_main_term_count >= 2,
                    shared_pos_term_count >= 2,
                    shared_aspect2_term_count >= 2,
                    bool(hierarchy_stats["hierarchical_found"]),
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
                "microtopic_label_clean_a": row_a.get("microtopic_label_clean", ""),
                "document_count_a": row_a.get("document_count", np.nan),
                "chunk_count_a": row_a.get("chunk_count", np.nan),
                "active_year_count_a": row_a.get("active_year_count", np.nan),
                "main_terms_display_a": row_a.get("main_terms_display", ""),
                "pos_terms_display_a": row_a.get("pos_terms_display", ""),
                "aspect2_terms_display_a": row_a.get("aspect2_terms_display", ""),
                "representative_chunk_preview_a": row_a.get("representative_chunk_preview", ""),
                "micro_topic_id_b": int(base["micro_topic_id_b"]),
                "topic_name_original_b": row_b["topic_name_original"],
                "microtopic_label_clean_b": row_b.get("microtopic_label_clean", ""),
                "document_count_b": row_b.get("document_count", np.nan),
                "chunk_count_b": row_b.get("chunk_count", np.nan),
                "active_year_count_b": row_b.get("active_year_count", np.nan),
                "main_terms_display_b": row_b.get("main_terms_display", ""),
                "pos_terms_display_b": row_b.get("pos_terms_display", ""),
                "aspect2_terms_display_b": row_b.get("aspect2_terms_display", ""),
                "representative_chunk_preview_b": row_b.get("representative_chunk_preview", ""),
                "profile_similarity": float(base["profile_similarity"]),
                "rank_from_a": base["rank_from_a"],
                "rank_from_b": base["rank_from_b"],
                "selected_from_a": bool(base["selected_from_a"]),
                "selected_from_b": bool(base["selected_from_b"]),
                "selected_from_both_directions": bool(base["selected_from_a"] and base["selected_from_b"]),
                "shared_main_term_count": shared_main_term_count,
                "main_term_jaccard": main_term_jaccard,
                "shared_main_terms_json": json_dumps(shared_main_terms),
                "shared_pos_term_count": shared_pos_term_count,
                "pos_term_jaccard": pos_term_jaccard,
                "shared_pos_terms_json": json_dumps(shared_pos_terms),
                "shared_aspect2_term_count": shared_aspect2_term_count,
                "aspect2_term_jaccard": aspect2_term_jaccard,
                "shared_aspect2_terms_json": json_dumps(shared_aspect2_terms),
                **hierarchy_stats,
                "semantic_support_count": semantic_support_count,
                "manual_pair_review": "",
                "manual_pair_notes": "",
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    if candidate_frame.empty:
        return candidate_frame
    candidate_frame = candidate_frame.sort_values(
        ["subgroup", "semantic_support_count", "profile_similarity"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return candidate_frame


def instructions_frame() -> pd.DataFrame:
    rows = [
        {
            "section": "Goal",
            "instruction": "Review candidate pairs within each subgroup immediately after unsupervised BERTopic and decide whether multiple raw microtopics should be merged analytically.",
        },
        {
            "section": "Decision rule",
            "instruction": "Use manual_merge_group_id in the microtopic_roster sheet as the source of truth. Leave it blank to keep the topic as a singleton.",
        },
        {
            "section": "ID format",
            "instruction": "Recommended pattern: academic_T1_G01, academic_T1_G02, media_T4_G01. IDs must stay unique within subgroup and must not mix subgroups.",
        },
        {
            "section": "Evidence",
            "instruction": "Base the decision on semantic evidence: profile similarity, Main/POS/Aspect2 overlaps, hierarchical proximity, topic size, and representative text.",
        },
        {
            "section": "When to merge",
            "instruction": "Merge when the two topics appear to be the same substantive cluster expressed with slightly different wording or fragmentation inside the same subgroup.",
        },
        {
            "section": "When to keep separate",
            "instruction": "Keep separate when the topics differ in substantive family, representative evidence, or semantic surface even if one aspect partially overlaps.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    roster, profiles, hierarchy_map = build_roster_and_support(
        micro_root=args.micro_root,
        max_representative_chunks=args.max_representative_chunks,
        max_representative_chars=args.max_representative_chars,
    )
    metadata, embeddings, embedding_model = compute_profile_embeddings(
        profiles=profiles,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    candidate_pairs = build_candidate_pairs(
        roster=roster,
        metadata=metadata,
        embeddings=embeddings,
        hierarchy_map=hierarchy_map,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    candidate_csv = args.output_root / "merge_candidate_pairs.csv"
    roster_csv = args.output_root / "microtopic_roster.csv"
    workbook_path = output_path(args.output_root, "microtopic_merge_first_review", ".xlsx")
    manifest_path = args.output_root / "merge_review_manifest.json"
    readme_path = output_path(args.output_root, "README_microtopic_merge_first_review", ".md")
    candidate_summary_csv = args.output_root / "merge_candidate_summary.csv"
    profile_csv = args.output_root / "microtopic_profiles.csv"
    metadata_csv = args.output_root / "microtopic_profile_embedding_metadata.csv"
    embeddings_npy = args.output_root / "microtopic_profile_embeddings.npy"
    embedding_manifest_path = args.output_root / "embedding_manifest.json"

    profiles.to_csv(profile_csv, index=False)
    metadata.to_csv(metadata_csv, index=False)
    np.save(embeddings_npy, embeddings)
    candidate_pairs.to_csv(candidate_csv, index=False)
    roster.to_csv(roster_csv, index=False)
    write_json(
        embedding_manifest_path,
        {
            "embedding_model": embedding_model,
            "row_count": int(embeddings.shape[0]) if embeddings.size else 0,
            "dims": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
            "dtype": str(embeddings.dtype) if embeddings.size else "float32",
            "batch_size": args.batch_size,
            "outputs": {
                "profiles_csv": str(profile_csv),
                "embedding_metadata_csv": str(metadata_csv),
                "embeddings_npy": str(embeddings_npy),
            },
        },
    )

    summary = (
        candidate_pairs.groupby(["subgroup", "source", "assigned_label"], dropna=False)
        .agg(
            candidate_pair_count=("candidate_pair_id", "nunique"),
            max_profile_similarity=("profile_similarity", "max"),
            mean_profile_similarity=("profile_similarity", "mean"),
            mean_semantic_support_count=("semantic_support_count", "mean"),
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
            "main_terms_display",
            "pos_terms_display",
            "aspect2_terms_display",
            "document_count",
            "chunk_count",
            "representative_chunk_preview",
            "manual_merge_group_id",
            "manual_merge_confidence",
            "manual_merge_notes",
        ]
    ]
    instructions = instructions_frame()

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sanitize_frame_for_excel(candidate_pairs).to_excel(writer, sheet_name="candidate_pairs", index=False)
        sanitize_frame_for_excel(roster_sheet).to_excel(writer, sheet_name="microtopic_roster", index=False)
        sanitize_frame_for_excel(instructions).to_excel(writer, sheet_name="instructions", index=False)
        if not summary.empty:
            sanitize_frame_for_excel(summary).to_excel(writer, sheet_name="candidate_summary", index=False)
        workbook_autofit(writer, "candidate_pairs", sanitize_frame_for_excel(candidate_pairs))
        workbook_autofit(writer, "microtopic_roster", sanitize_frame_for_excel(roster_sheet))
        workbook_autofit(writer, "instructions", sanitize_frame_for_excel(instructions))
        if not summary.empty:
            workbook_autofit(writer, "candidate_summary", sanitize_frame_for_excel(summary))

    readme_lines = [
        "# Merge-First Microtopic Review",
        "",
        f"- raw non-outlier microtopics in scope: `{int(roster.shape[0])}`",
        f"- candidate pairs generated: `{int(candidate_pairs.shape[0])}`",
        f"- subgroup count covered: `{int(roster['subgroup'].nunique()) if not roster.empty else 0}`",
        f"- semantic evidence basis: `Main + POS + Aspect2 + hierarchy`",
        "",
        "## Files",
        f"- workbook: `{workbook_path}`",
        f"- candidate pairs CSV: `{candidate_csv}`",
        f"- roster CSV: `{roster_csv}`",
        "",
        "## Review flow",
        "1. Inspect `candidate_pairs` for within-subgroup merge suggestions.",
        "2. Prioritize semantic similarity and Main/POS/Aspect2 agreement; ignore temporal resemblance as merge evidence.",
        "3. Record the final decision only in `microtopic_roster.manual_merge_group_id`.",
        "4. Leave `manual_merge_group_id` blank for singleton topics.",
        "5. Materialize the approved mapping and build the merged micro-root for downstream analyses.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    write_json(
        manifest_path,
        {
            "micro_root": str(args.micro_root),
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "embedding_model": embedding_model,
            "review_mode": "semantic_multiaspect",
            "row_counts": {
                "raw_non_outlier_microtopics": int(roster.shape[0]),
                "candidate_pairs": int(candidate_pairs.shape[0]),
                "roster_rows": int(roster.shape[0]),
            },
            "outputs": {
                "candidate_pairs_csv": str(candidate_csv),
                "candidate_summary_csv": str(candidate_summary_csv),
                "microtopic_roster_csv": str(roster_csv),
                "review_workbook": str(workbook_path),
                "microtopic_profiles_csv": str(profile_csv),
                "embedding_metadata_csv": str(metadata_csv),
                "embedding_npy": str(embeddings_npy),
                "embedding_manifest": str(embedding_manifest_path),
                "readme": str(readme_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
