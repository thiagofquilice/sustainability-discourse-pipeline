#!/usr/bin/env python3
"""Build a hierarchical BERTopic merge-review workbook from local subgroup hierarchies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_cross_source_microtopic_profiles import representative_chunk_map
from cross_source_microtopic_common import (
    MACRO_TOPIC_ORDER,
    SOURCE_ORDER,
    clean_microtopic_label,
    configure_logging,
    parse_topic_terms,
    read_json,
    subgroup_macro_topic_name,
    truncate_text,
)
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT,
    RAW_MICRO_ROOT,
    ensure_directory,
    json_dumps,
    normalize_text,
    parse_json_list,
    sanitize_frame_for_excel,
    stringify_list,
    workbook_autofit,
    write_json,
)


@dataclass
class NodeSummary:
    topic_ids: list[int]
    name: str
    terms: list[str]
    preview: str


class DisjointSet:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union_many(self, items: list[int]) -> None:
        cleaned = [int(item) for item in items]
        if len(cleaned) < 2:
            return
        root = self.find(cleaned[0])
        for item in cleaned[1:]:
            other = self.find(item)
            if other != root:
                self.parent[other] = root

    def components(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for item in sorted(self.parent):
            buckets[self.find(item)].append(int(item))
        return [sorted(values) for values in buckets.values()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=RAW_MICRO_ROOT)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_HIERARCHICAL_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--distance-quantile", type=float, default=0.25)
    parser.add_argument("--max-representative-chunks", type=int, default=5)
    parser.add_argument("--max-representative-chars", type=int, default=480)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def parse_int_list(value: Any) -> list[int]:
    values: list[int] = []
    for item in parse_json_list(value):
        try:
            values.append(int(item))
        except Exception:
            continue
    return values


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


def digest_terms(topic_ids: list[int], topic_records: dict[int, dict[str, Any]], limit: int = 12) -> list[str]:
    pool: list[str] = []
    for topic_id in topic_ids:
        record = topic_records.get(int(topic_id), {})
        pool.extend(parse_topic_terms(record.get("Representation")))
    return stable_unique_texts(pool, limit=limit)


def digest_names(topic_ids: list[int], topic_records: dict[int, dict[str, Any]], limit: int = 6) -> list[str]:
    pool = [
        normalize_text(topic_records.get(int(topic_id), {}).get("Name", ""))
        for topic_id in topic_ids
    ]
    return stable_unique_texts(pool, limit=limit)


def digest_preview(topic_ids: list[int], representative_map: dict[int, list[str]]) -> str:
    previews: list[str] = []
    for topic_id in topic_ids:
        previews.extend(representative_map.get(int(topic_id), []))
    previews = stable_unique_texts(previews, limit=2)
    if not previews:
        return ""
    return " || ".join(truncate_text(text, 220) for text in previews)


def resolve_node_summary(
    node_id: int,
    hierarchy_by_parent: dict[int, dict[str, Any]],
    topic_records: dict[int, dict[str, Any]],
    representative_map: dict[int, list[str]],
    cache: dict[int, NodeSummary],
) -> NodeSummary:
    node_id = int(node_id)
    cached = cache.get(node_id)
    if cached is not None:
        return cached

    if node_id in topic_records:
        record = topic_records[node_id]
        topic_ids = [node_id]
        summary = NodeSummary(
            topic_ids=topic_ids,
            name=normalize_text(record.get("Name", f"{node_id}")),
            terms=stable_unique_texts(parse_topic_terms(record.get("Representation")), limit=8),
            preview=digest_preview(topic_ids, representative_map),
        )
        cache[node_id] = summary
        return summary

    row = hierarchy_by_parent.get(node_id)
    if row is None:
        summary = NodeSummary(topic_ids=[], name="", terms=[], preview="")
        cache[node_id] = summary
        return summary

    left_summary = resolve_node_summary(row["child_left_id"], hierarchy_by_parent, topic_records, representative_map, cache)
    right_summary = resolve_node_summary(row["child_right_id"], hierarchy_by_parent, topic_records, representative_map, cache)
    topic_ids = sorted({*left_summary.topic_ids, *right_summary.topic_ids})
    if not topic_ids:
        topic_ids = sorted({int(item) for item in row["topics"]})
    summary = NodeSummary(
        topic_ids=topic_ids,
        name=normalize_text(row.get("parent_name", "")),
        terms=digest_terms(topic_ids, topic_records, limit=10),
        preview=digest_preview(topic_ids, representative_map),
    )
    cache[node_id] = summary
    return summary


def build_topic_roster(
    subgroup_dir: Path,
    max_representative_chunks: int,
    max_representative_chars: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = read_json(subgroup_dir / "manifest.json")
    document_topics = pd.read_parquet(subgroup_dir / "document_topics.parquet")
    topic_info = pd.read_csv(subgroup_dir / "topic_info.csv")
    representative_docs = pd.read_csv(subgroup_dir / "representative_docs.csv")

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

    macro_topic_name = subgroup_macro_topic_name(document_topics, manifest)
    representative_map = representative_chunk_map(
        representative_docs=representative_docs,
        document_topics=document_topics,
        max_chunks=max_representative_chunks,
        max_chars=max_representative_chars,
    )
    topic_meta = topic_info.set_index("Topic").to_dict(orient="index")

    roster_rows: list[dict[str, Any]] = []
    for micro_topic_id, group in document_topics.groupby("micro_topic_id", dropna=False):
        topic_id = int(micro_topic_id)
        topic_record = topic_meta.get(topic_id)
        if not topic_record:
            continue
        years_present = sorted(group["year"].astype(int).unique().tolist())
        top_terms = parse_topic_terms(topic_record.get("Representation"))
        representative_chunks = representative_map.get(topic_id, [])
        microtopic_label = normalize_text(topic_record.get("Name", f"{topic_id}"))
        roster_rows.append(
            {
                "subgroup": subgroup_dir.name,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "macro_topic_name": macro_topic_name,
                "micro_topic_id": topic_id,
                "topic_name_original": microtopic_label,
                "microtopic_label_clean": clean_microtopic_label(microtopic_label) or microtopic_label,
                "document_count": int(group["source_doc_id"].nunique()),
                "chunk_count": int(group.shape[0]),
                "active_year_count": int(len(years_present)),
                "first_year": int(min(years_present)) if years_present else None,
                "last_year": int(max(years_present)) if years_present else None,
                "top_terms": json_dumps(top_terms),
                "top_terms_display": stringify_list(top_terms),
                "representative_chunks": json_dumps(representative_chunks),
                "representative_chunk_preview": normalize_text(representative_chunks[0]) if representative_chunks else "",
            }
        )

    roster = pd.DataFrame(roster_rows)
    return roster, manifest, topic_info, document_topics


def build_hierarchy_frame(subgroup_dir: Path) -> pd.DataFrame:
    hierarchy_path = subgroup_dir / "hierarchical_topics.csv"
    if not hierarchy_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(hierarchy_path)
    if frame.empty:
        return frame
    frame["Parent_ID"] = pd.to_numeric(frame["Parent_ID"], errors="coerce")
    frame["Child_Left_ID"] = pd.to_numeric(frame["Child_Left_ID"], errors="coerce")
    frame["Child_Right_ID"] = pd.to_numeric(frame["Child_Right_ID"], errors="coerce")
    frame["Distance"] = pd.to_numeric(frame["Distance"], errors="coerce")
    frame = frame.dropna(subset=["Parent_ID", "Child_Left_ID", "Child_Right_ID", "Distance"]).copy()
    frame["Parent_ID"] = frame["Parent_ID"].astype(int)
    frame["Child_Left_ID"] = frame["Child_Left_ID"].astype(int)
    frame["Child_Right_ID"] = frame["Child_Right_ID"].astype(int)
    frame["Topics_List"] = frame["Topics"].map(parse_int_list)
    frame = frame.loc[frame["Topics_List"].map(len) >= 2].copy()
    return frame


def sort_outputs(frame: pd.DataFrame, extra_sort: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    sorted_frame = frame.copy()
    if "source" in sorted_frame.columns:
        sorted_frame["source"] = pd.Categorical(sorted_frame["source"], categories=SOURCE_ORDER, ordered=True)
    if "assigned_label" in sorted_frame.columns:
        sorted_frame["assigned_label"] = pd.Categorical(
            sorted_frame["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
        )
    return sorted_frame.sort_values(extra_sort).reset_index(drop=True)


def build_review_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    singleton_rows: list[dict[str, Any]] = []

    subgroups_missing_hierarchy: list[str] = []
    subgroup_thresholds: list[dict[str, Any]] = []

    subgroup_dirs = sorted(path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary")
    total_topics = 0

    for subgroup_dir in subgroup_dirs:
        roster, manifest, topic_info, document_topics = build_topic_roster(
            subgroup_dir=subgroup_dir,
            max_representative_chunks=args.max_representative_chunks,
            max_representative_chars=args.max_representative_chars,
        )
        if roster.empty:
            continue

        total_topics += int(roster.shape[0])
        macro_topic_name = str(roster["macro_topic_name"].iloc[0]) if "macro_topic_name" in roster.columns else ""
        hierarchy = build_hierarchy_frame(subgroup_dir)
        topic_records = topic_info.set_index("Topic").to_dict(orient="index")
        representative_map = {
            int(row.micro_topic_id): parse_json_list(row.representative_chunks)
            for row in roster.itertuples(index=False)
        }

        if hierarchy.empty:
            subgroups_missing_hierarchy.append(subgroup_dir.name)
            for row in roster.itertuples(index=False):
                singleton_row = row._asdict()
                singleton_row["singleton_reason"] = "missing_hierarchical_topics"
                singleton_rows.append(singleton_row)
            subgroup_thresholds.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": manifest["source"],
                    "assigned_label": manifest["assigned_label"],
                    "distance_quantile": args.distance_quantile,
                    "distance_cutoff": np.nan,
                    "hierarchy_row_count": 0,
                    "filtered_row_count": 0,
                    "non_singleton_branchlet_count": 0,
                }
            )
            continue

        hierarchy_by_parent: dict[int, dict[str, Any]] = {}
        for row in hierarchy.itertuples(index=False):
            hierarchy_by_parent[int(row.Parent_ID)] = {
                "parent_id": int(row.Parent_ID),
                "parent_name": normalize_text(getattr(row, "Parent_Name", "")),
                "child_left_id": int(row.Child_Left_ID),
                "child_right_id": int(row.Child_Right_ID),
                "topics": [int(value) for value in getattr(row, "Topics_List", [])],
                "distance": float(row.Distance),
            }
        node_cache: dict[int, NodeSummary] = {}
        for topic_id in roster["micro_topic_id"].astype(int).tolist():
            resolve_node_summary(topic_id, hierarchy_by_parent, topic_records, representative_map, node_cache)

        distance_cutoff = float(hierarchy["Distance"].quantile(args.distance_quantile))
        filtered = hierarchy.loc[hierarchy["Distance"] <= distance_cutoff].copy()
        filtered = filtered.sort_values(["Distance", "Parent_ID"]).reset_index(drop=True)

        dsu = DisjointSet(roster["micro_topic_id"].astype(int).tolist())
        filtered_steps: list[dict[str, Any]] = []
        for idx, row in enumerate(filtered.itertuples(index=False), start=1):
            parent_summary = resolve_node_summary(int(row.Parent_ID), hierarchy_by_parent, topic_records, representative_map, node_cache)
            left_summary = resolve_node_summary(int(row.Child_Left_ID), hierarchy_by_parent, topic_records, representative_map, node_cache)
            right_summary = resolve_node_summary(int(row.Child_Right_ID), hierarchy_by_parent, topic_records, representative_map, node_cache)
            if len(parent_summary.topic_ids) < 2:
                continue
            dsu.union_many(parent_summary.topic_ids)
            filtered_steps.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": manifest["source"],
                    "assigned_label": manifest["assigned_label"],
                    "macro_topic_name": macro_topic_name,
                    "hier_step_id": f"{subgroup_dir.name}_HS{idx:03d}",
                    "distance": float(row.Distance),
                    "subgroup_distance_p25": distance_cutoff,
                    "parent_id": int(row.Parent_ID),
                    "parent_name": normalize_text(getattr(row, "Parent_Name", "")),
                    "parent_topic_ids_json": json_dumps(parent_summary.topic_ids),
                    "parent_topic_ids_display": stringify_list([str(item) for item in parent_summary.topic_ids], sep=", "),
                    "parent_topic_count": int(len(parent_summary.topic_ids)),
                    "child_left_id": int(row.Child_Left_ID),
                    "child_left_name": normalize_text(getattr(row, "Child_Left_Name", "")),
                    "child_left_topic_ids_json": json_dumps(left_summary.topic_ids),
                    "child_left_topic_ids_display": stringify_list([str(item) for item in left_summary.topic_ids], sep=", "),
                    "child_left_topic_count": int(len(left_summary.topic_ids)),
                    "child_left_terms_display": stringify_list(left_summary.terms),
                    "child_left_representative_preview": left_summary.preview,
                    "child_right_id": int(row.Child_Right_ID),
                    "child_right_name": normalize_text(getattr(row, "Child_Right_Name", "")),
                    "child_right_topic_ids_json": json_dumps(right_summary.topic_ids),
                    "child_right_topic_ids_display": stringify_list([str(item) for item in right_summary.topic_ids], sep=", "),
                    "child_right_topic_count": int(len(right_summary.topic_ids)),
                    "child_right_terms_display": stringify_list(right_summary.terms),
                    "child_right_representative_preview": right_summary.preview,
                    "step_review_notes": "",
                }
            )

        component_lists = dsu.components()
        component_lists.sort(key=lambda members: (len(members) == 1, members[0]))
        branchlet_map: dict[int, str] = {}
        branchlet_index = 0
        for members in component_lists:
            if len(members) < 2:
                continue
            branchlet_index += 1
            branchlet_id = f"{subgroup_dir.name}_HB{branchlet_index:02d}"
            for topic_id in members:
                branchlet_map[int(topic_id)] = branchlet_id

        subgroup_step_rows = []
        for row in filtered_steps:
            parent_ids = parse_int_list(row["parent_topic_ids_json"])
            branchlet_id = branchlet_map.get(parent_ids[0]) if parent_ids else ""
            row["proposed_branchlet_id"] = branchlet_id
            subgroup_step_rows.append(row)

        roster_lookup = {
            int(row.micro_topic_id): row._asdict()
            for row in roster.itertuples(index=False)
        }
        grouped_steps = defaultdict(list)
        for row in subgroup_step_rows:
            branchlet_id = normalize_text(row.get("proposed_branchlet_id", ""))
            if branchlet_id:
                grouped_steps[branchlet_id].append(row)

        for branchlet_id, rows in grouped_steps.items():
            member_ids = sorted(
                {
                    topic_id
                    for row in rows
                    for topic_id in parse_int_list(row["parent_topic_ids_json"])
                }
            )
            member_records = [roster_lookup[int(topic_id)] for topic_id in member_ids]
            branchlet_terms = digest_terms(member_ids, topic_records, limit=12)
            branchlet_names = digest_names(member_ids, topic_records, limit=8)
            distances = [float(row["distance"]) for row in rows]
            summary_rows.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": manifest["source"],
                    "assigned_label": manifest["assigned_label"],
                    "macro_topic_name": macro_topic_name,
                    "proposed_branchlet_id": branchlet_id,
                    "member_topic_ids": stringify_list([str(item) for item in member_ids], sep=", "),
                    "member_topic_ids_json": json_dumps(member_ids),
                    "member_topic_names": stringify_list(branchlet_names, sep=" || "),
                    "member_topic_count": int(len(member_ids)),
                    "internal_step_count": int(len(rows)),
                    "min_distance": float(min(distances)),
                    "max_distance": float(max(distances)),
                    "mean_distance": float(np.mean(distances)),
                    "representative_terms_digest": stringify_list(branchlet_terms),
                    "representative_topic_name_digest": stringify_list(branchlet_names, sep=" || "),
                    "group_decision": "",
                    "group_review_notes": "",
                }
            )
            for member in member_records:
                member_rows.append(
                    {
                        "subgroup": subgroup_dir.name,
                        "source": manifest["source"],
                        "assigned_label": manifest["assigned_label"],
                        "macro_topic_name": macro_topic_name,
                        "proposed_branchlet_id": branchlet_id,
                        "micro_topic_id": int(member["micro_topic_id"]),
                        "topic_name_original": str(member.get("topic_name_original", "")),
                        "microtopic_label_clean": str(member.get("microtopic_label_clean", "")),
                        "document_count": int(member.get("document_count", 0)),
                        "chunk_count": int(member.get("chunk_count", 0)),
                        "active_year_count": int(member.get("active_year_count", 0)),
                        "top_terms_display": str(member.get("top_terms_display", "")),
                        "representative_chunk_preview": str(member.get("representative_chunk_preview", "")),
                        "manual_review_group_id": "",
                        "manual_member_notes": "",
                    }
                )

        non_singleton_topic_ids = set(branchlet_map)
        for row in roster.itertuples(index=False):
            if int(row.micro_topic_id) in non_singleton_topic_ids:
                continue
            singleton_row = row._asdict()
            singleton_row["singleton_reason"] = "not_in_filtered_hierarchy_branchlet"
            singleton_rows.append(singleton_row)

        step_rows.extend(subgroup_step_rows)
        subgroup_thresholds.append(
            {
                "subgroup": subgroup_dir.name,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "distance_quantile": args.distance_quantile,
                "distance_cutoff": distance_cutoff,
                "hierarchy_row_count": int(hierarchy.shape[0]),
                "filtered_row_count": int(len(filtered_steps)),
                "non_singleton_branchlet_count": int(len(grouped_steps)),
            }
        )

    summary_groups = pd.DataFrame(summary_rows)
    merge_steps = pd.DataFrame(step_rows)
    group_members = pd.DataFrame(member_rows)
    singletons = pd.DataFrame(singleton_rows)

    summary_groups = sort_outputs(
        summary_groups,
        ["source", "assigned_label", "subgroup", "proposed_branchlet_id"],
    )
    merge_steps = sort_outputs(
        merge_steps,
        ["source", "assigned_label", "subgroup", "distance", "parent_id"],
    )
    group_members = sort_outputs(
        group_members,
        ["source", "assigned_label", "subgroup", "proposed_branchlet_id", "micro_topic_id"],
    )
    singletons = sort_outputs(
        singletons,
        ["source", "assigned_label", "subgroup", "micro_topic_id"],
    )

    manifest = {
        "micro_root": str(args.micro_root),
        "distance_quantile": args.distance_quantile,
        "subgroup_count": len(subgroup_dirs),
        "raw_topic_count": int(total_topics),
        "candidate_step_count": int(merge_steps.shape[0]),
        "proposed_branchlet_count": int(summary_groups.shape[0]),
        "topics_in_branchlets": int(group_members.shape[0]),
        "singleton_topic_count": int(singletons.shape[0]),
        "subgroups_missing_hierarchy": subgroups_missing_hierarchy,
        "subgroup_thresholds": subgroup_thresholds,
    }
    return summary_groups, merge_steps, group_members, singletons, manifest


def write_workbook(
    output_root: Path,
    summary_groups: pd.DataFrame,
    merge_steps: pd.DataFrame,
    group_members: pd.DataFrame,
    singletons: pd.DataFrame,
) -> None:
    instructions = pd.DataFrame(
        [
            {
                "rule": "accept",
                "description": "Keep the proposed branchlet as one final merged group.",
            },
            {
                "rule": "reject",
                "description": "Reject the branchlet and keep every member topic as a singleton.",
            },
            {
                "rule": "review",
                "description": "Use group_members.manual_review_group_id for every member in the branchlet.",
            },
            {
                "rule": "blank decision",
                "description": "Blank group_decision is treated as reject during materialization.",
            },
        ]
    )

    workbook_path = output_root / "microtopic_merge_first_hierarchical_review.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        workbook_frames = [
            ("summary_groups", summary_groups, "A2"),
            ("merge_steps", merge_steps, "A2"),
            ("group_members", group_members, "A2"),
            ("singletons", singletons, "A2"),
            ("instructions", instructions, "A2"),
        ]
        for sheet_name, frame, freeze_cell in workbook_frames:
            clean_frame = sanitize_frame_for_excel(frame)
            clean_frame.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook_autofit(writer, sheet_name, clean_frame, freeze_cell=freeze_cell)


def write_readme(output_root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Merge-First Hierarchical Review",
        "",
        "This review workbook is built from each subgroup's `hierarchical_topics.csv`.",
        "",
        "Rules used:",
        f"- candidate source: local hierarchical BERTopic merges within each `source x macro-topic` subgroup",
        f"- distance cutoff: subgroup-specific p{int(manifest['distance_quantile'] * 100):02d}",
        "- no temporal similarity, year-overlap, or lead-lag criteria",
        "- no cross-source or cross-macro-topic suggestions",
        "",
        "Workbook sheets:",
        "- `summary_groups`: main decision sheet (`accept`, `reject`, `review`)",
        "- `merge_steps`: supporting local merge-step detail",
        "- `group_members`: member-level manual override sheet for `review` groups",
        "- `singletons`: topics not captured by any proposed branchlet",
    ]
    (output_root / "README_microtopic_merge_first_hierarchical_review.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    summary_groups, merge_steps, group_members, singletons, manifest = build_review_frames(args)

    summary_csv = args.output_root / "proposed_branchlet_summary.csv"
    steps_csv = args.output_root / "proposed_branchlet_merge_steps.csv"
    members_csv = args.output_root / "proposed_branchlet_members.csv"
    singletons_csv = args.output_root / "singleton_topics.csv"

    summary_groups.to_csv(summary_csv, index=False)
    merge_steps.to_csv(steps_csv, index=False)
    group_members.to_csv(members_csv, index=False)
    singletons.to_csv(singletons_csv, index=False)

    write_workbook(args.output_root, summary_groups, merge_steps, group_members, singletons)
    write_json(args.output_root / "hierarchical_review_component_manifest.json", manifest)
    write_readme(args.output_root, manifest)

    print(args.output_root)


if __name__ == "__main__":
    main()
