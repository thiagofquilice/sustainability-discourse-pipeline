#!/usr/bin/env python3
"""Build a group-based merge review workbook from merge-first candidate pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_source_microtopic_common import MACRO_TOPIC_ORDER, SOURCE_ORDER, configure_logging
from microtopic_posthoc_merge_common import (
    MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    MERGE_FIRST_REVIEW_MULTIASPECT_OUTPUT_ROOT,
    ensure_directory,
    normalize_text,
    sanitize_frame_for_excel,
    safe_read_csv,
    stringify_list,
    workbook_autofit,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=MERGE_FIRST_REVIEW_MULTIASPECT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=MERGE_FIRST_GROUP_REVIEW_MULTIASPECT_OUTPUT_ROOT)
    parser.add_argument("--min-similarity", type=float, default=0.70)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def output_path(output_root: Path, stem: str, suffix: str) -> Path:
    if "multiaspect" in output_root.name.lower():
        return output_root / f"{stem}_multiaspect{suffix}"
    return output_root / f"{stem}{suffix}"


def cast_inputs(roster: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if roster.empty:
        raise SystemExit("No microtopic roster found in merge-first review root.")
    if pairs.empty:
        raise SystemExit("No merge candidate pairs found in merge-first review root.")

    roster = roster.copy()
    pairs = pairs.copy()
    roster["subgroup"] = roster["subgroup"].astype(str)
    roster["source"] = roster["source"].astype(str)
    roster["assigned_label"] = roster["assigned_label"].astype(str)
    roster["micro_topic_id"] = pd.to_numeric(roster["micro_topic_id"], errors="coerce")
    roster = roster.dropna(subset=["micro_topic_id"]).copy()
    roster["micro_topic_id"] = roster["micro_topic_id"].astype(int)
    for column in [
        "main_terms_display",
        "pos_terms_display",
        "aspect2_terms_display",
        "representative_chunk_preview",
        "topic_name_original",
        "microtopic_label_clean",
    ]:
        if column not in roster.columns:
            roster[column] = ""

    pairs["subgroup"] = pairs["subgroup"].astype(str)
    pairs["source"] = pairs["source"].astype(str)
    pairs["assigned_label"] = pairs["assigned_label"].astype(str)
    if "semantic_support_count" not in pairs.columns:
        pairs["semantic_support_count"] = pairs.get("support_signal_count", np.nan)
    if "main_term_jaccard" not in pairs.columns:
        pairs["main_term_jaccard"] = pairs.get("top_term_jaccard", np.nan)
    if "shared_main_term_count" not in pairs.columns:
        pairs["shared_main_term_count"] = pairs.get("shared_top_term_count", np.nan)
    for column in [
        "shared_pos_term_count",
        "pos_term_jaccard",
        "shared_aspect2_term_count",
        "aspect2_term_jaccard",
        "main_terms_display_a",
        "main_terms_display_b",
        "pos_terms_display_a",
        "pos_terms_display_b",
        "aspect2_terms_display_a",
        "aspect2_terms_display_b",
        "hierarchical_closest_parent_name",
        "representative_chunk_preview_a",
        "representative_chunk_preview_b",
    ]:
        if column not in pairs.columns:
            pairs[column] = ""
    pairs["micro_topic_id_a"] = pd.to_numeric(pairs["micro_topic_id_a"], errors="coerce")
    pairs["micro_topic_id_b"] = pd.to_numeric(pairs["micro_topic_id_b"], errors="coerce")
    pairs["profile_similarity"] = pd.to_numeric(pairs["profile_similarity"], errors="coerce")
    pairs["semantic_support_count"] = pd.to_numeric(pairs["semantic_support_count"], errors="coerce")
    pairs["shared_main_term_count"] = pd.to_numeric(pairs["shared_main_term_count"], errors="coerce")
    pairs["main_term_jaccard"] = pd.to_numeric(pairs["main_term_jaccard"], errors="coerce")
    pairs["shared_pos_term_count"] = pd.to_numeric(pairs["shared_pos_term_count"], errors="coerce")
    pairs["pos_term_jaccard"] = pd.to_numeric(pairs["pos_term_jaccard"], errors="coerce")
    pairs["shared_aspect2_term_count"] = pd.to_numeric(pairs["shared_aspect2_term_count"], errors="coerce")
    pairs["aspect2_term_jaccard"] = pd.to_numeric(pairs["aspect2_term_jaccard"], errors="coerce")
    pairs["hierarchical_min_distance"] = pd.to_numeric(pairs["hierarchical_min_distance"], errors="coerce")
    pairs = pairs.dropna(subset=["micro_topic_id_a", "micro_topic_id_b", "profile_similarity"]).copy()
    pairs["micro_topic_id_a"] = pairs["micro_topic_id_a"].astype(int)
    pairs["micro_topic_id_b"] = pairs["micro_topic_id_b"].astype(int)
    pairs["selected_from_both_directions"] = pairs["selected_from_both_directions"].map(coerce_bool)
    return roster, pairs


def filter_edges(pairs: pd.DataFrame, min_similarity: float) -> pd.DataFrame:
    filtered = pairs.loc[
        pairs["selected_from_both_directions"] & (pairs["profile_similarity"] >= float(min_similarity))
    ].copy()
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values(
        ["subgroup", "profile_similarity", "semantic_support_count", "micro_topic_id_a", "micro_topic_id_b"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    return filtered


def build_components(
    roster: pd.DataFrame,
    filtered_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    roster_map = {
        (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
        for row in roster.itertuples(index=False)
    }

    group_summary_rows: list[dict[str, Any]] = []
    group_member_rows: list[dict[str, Any]] = []
    singleton_rows: list[dict[str, Any]] = []
    group_edge_rows: list[dict[str, Any]] = []

    total_components = 0
    non_singleton_groups = 0
    topics_in_non_singletons = 0

    for subgroup, subgroup_roster in roster.groupby("subgroup", dropna=False, sort=False):
        topics = sorted(subgroup_roster["micro_topic_id"].astype(int).tolist())
        parent = {topic_id: topic_id for topic_id in topics}

        def find(topic_id: int) -> int:
            while parent[topic_id] != topic_id:
                parent[topic_id] = parent[parent[topic_id]]
                topic_id = parent[topic_id]
            return topic_id

        def union(topic_a: int, topic_b: int) -> None:
            root_a = find(topic_a)
            root_b = find(topic_b)
            if root_a != root_b:
                parent[root_b] = root_a

        subgroup_edges = filtered_edges.loc[filtered_edges["subgroup"] == str(subgroup)].copy()
        for row in subgroup_edges.itertuples(index=False):
            if int(row.micro_topic_id_a) in parent and int(row.micro_topic_id_b) in parent:
                union(int(row.micro_topic_id_a), int(row.micro_topic_id_b))

        components: dict[int, list[int]] = defaultdict(list)
        for topic_id in topics:
            components[find(topic_id)].append(int(topic_id))

        component_lists = [sorted(members) for members in components.values()]
        component_lists.sort(key=lambda members: (len(members) == 1, members[0]))
        total_components += len(component_lists)

        proposed_group_index = 0
        topic_to_group_id: dict[int, str] = {}
        for members in component_lists:
            if len(members) == 1:
                singleton_topic_id = int(members[0])
                row = roster_map[(str(subgroup), singleton_topic_id)].copy()
                row["singleton_reason"] = "no_non_singleton_component"
                singleton_rows.append(row)
                continue

            proposed_group_index += 1
            proposed_group_id = f"{subgroup}_PG{proposed_group_index:02d}"
            non_singleton_groups += 1
            topics_in_non_singletons += len(members)
            for topic_id in members:
                topic_to_group_id[int(topic_id)] = proposed_group_id

        subgroup_edge_rows = subgroup_edges.loc[
            subgroup_edges["micro_topic_id_a"].map(topic_to_group_id).notna()
            & (subgroup_edges["micro_topic_id_a"].map(topic_to_group_id) == subgroup_edges["micro_topic_id_b"].map(topic_to_group_id))
        ].copy()
        if not subgroup_edge_rows.empty:
            subgroup_edge_rows["proposed_group_id"] = subgroup_edge_rows["micro_topic_id_a"].map(topic_to_group_id)
        else:
            subgroup_edge_rows = subgroup_edge_rows.assign(proposed_group_id=pd.Series(dtype=str))

        for proposed_group_id, edge_group in subgroup_edge_rows.groupby("proposed_group_id", dropna=False, sort=False):
            member_ids = sorted(
                {
                    int(value)
                    for value in edge_group["micro_topic_id_a"].tolist() + edge_group["micro_topic_id_b"].tolist()
                }
            )
            member_rows = [roster_map[(str(subgroup), int(topic_id))] for topic_id in member_ids]
            for member_row in member_rows:
                group_member_rows.append(
                    {
                        "subgroup": str(member_row["subgroup"]),
                        "source": str(member_row["source"]),
                        "assigned_label": str(member_row["assigned_label"]),
                        "macro_topic_name": str(member_row.get("macro_topic_name", "")),
                        "proposed_group_id": str(proposed_group_id),
                        "proposed_group_size": int(len(member_ids)),
                        "micro_topic_id": int(member_row["micro_topic_id"]),
                        "topic_name_original": str(member_row.get("topic_name_original", "")),
                        "microtopic_label_clean": str(member_row.get("microtopic_label_clean", "")),
                        "document_count": int(member_row.get("document_count", 0)),
                        "chunk_count": int(member_row.get("chunk_count", 0)),
                        "active_year_count": int(member_row.get("active_year_count", 0)),
                        "main_terms_display": str(member_row.get("main_terms_display", "")),
                        "pos_terms_display": str(member_row.get("pos_terms_display", "")),
                        "aspect2_terms_display": str(member_row.get("aspect2_terms_display", "")),
                        "representative_chunk_preview": str(member_row.get("representative_chunk_preview", "")),
                        "manual_split_group_id": "",
                        "manual_member_notes": "",
                    }
                )

            group_edges = edge_group.copy()
            group_edges["proposed_group_size"] = int(len(member_ids))
            group_edges["edge_rank_within_group"] = range(1, len(group_edges) + 1)
            group_edge_rows.extend(
                group_edges[
                    [
                        "subgroup",
                        "source",
                        "assigned_label",
                        "macro_topic_name",
                        "proposed_group_id",
                        "proposed_group_size",
                        "micro_topic_id_a",
                        "micro_topic_id_b",
                        "profile_similarity",
                        "semantic_support_count",
                        "shared_main_term_count",
                        "main_term_jaccard",
                        "shared_pos_term_count",
                        "pos_term_jaccard",
                        "shared_aspect2_term_count",
                        "aspect2_term_jaccard",
                        "hierarchical_min_distance",
                        "hierarchical_closest_parent_name",
                        "main_terms_display_a",
                        "pos_terms_display_a",
                        "aspect2_terms_display_a",
                        "representative_chunk_preview_a",
                        "main_terms_display_b",
                        "pos_terms_display_b",
                        "aspect2_terms_display_b",
                        "representative_chunk_preview_b",
                        "edge_rank_within_group",
                    ]
                ].to_dict(orient="records")
            )

            hierarchy_dists = group_edges.loc[group_edges["hierarchical_min_distance"].notna(), "hierarchical_min_distance"]
            member_main_digest = stringify_list(
                [str(row.get("main_terms_display", "")) for row in member_rows if normalize_text(row.get("main_terms_display", ""))],
                sep=" || ",
            )
            member_pos_digest = stringify_list(
                [str(row.get("pos_terms_display", "")) for row in member_rows if normalize_text(row.get("pos_terms_display", ""))],
                sep=" || ",
            )
            member_aspect2_digest = stringify_list(
                [str(row.get("aspect2_terms_display", "")) for row in member_rows if normalize_text(row.get("aspect2_terms_display", ""))],
                sep=" || ",
            )
            group_summary_rows.append(
                {
                    "subgroup": str(member_rows[0]["subgroup"]),
                    "source": str(member_rows[0]["source"]),
                    "assigned_label": str(member_rows[0]["assigned_label"]),
                    "macro_topic_name": str(member_rows[0].get("macro_topic_name", "")),
                    "proposed_group_id": str(proposed_group_id),
                    "proposed_group_size": int(len(member_ids)),
                    "member_topic_ids": stringify_list([str(topic_id) for topic_id in member_ids], sep=", "),
                    "member_topic_names": stringify_list([str(row.get("topic_name_original", "")) for row in member_rows], sep=" || "),
                    "member_main_terms_digest": member_main_digest,
                    "member_pos_terms_digest": member_pos_digest,
                    "member_aspect2_terms_digest": member_aspect2_digest,
                    "mean_profile_similarity": float(group_edges["profile_similarity"].mean()),
                    "min_profile_similarity": float(group_edges["profile_similarity"].min()),
                    "max_profile_similarity": float(group_edges["profile_similarity"].max()),
                    "mean_main_term_jaccard": float(group_edges["main_term_jaccard"].mean()) if group_edges["main_term_jaccard"].notna().any() else np.nan,
                    "mean_pos_term_jaccard": float(group_edges["pos_term_jaccard"].mean()) if group_edges["pos_term_jaccard"].notna().any() else np.nan,
                    "mean_aspect2_term_jaccard": float(group_edges["aspect2_term_jaccard"].mean()) if group_edges["aspect2_term_jaccard"].notna().any() else np.nan,
                    "mean_semantic_support_count": float(group_edges["semantic_support_count"].mean()) if group_edges["semantic_support_count"].notna().any() else np.nan,
                    "largest_internal_distance": float(hierarchy_dists.max()) if not hierarchy_dists.empty else np.nan,
                    "internal_edge_count": int(group_edges.shape[0]),
                    "group_decision": "",
                    "group_review_notes": "",
                }
            )

    group_summary = pd.DataFrame(group_summary_rows)
    group_members = pd.DataFrame(group_member_rows)
    group_edges = pd.DataFrame(group_edge_rows)
    singletons = pd.DataFrame(singleton_rows)

    if not group_summary.empty:
        group_summary["source"] = pd.Categorical(group_summary["source"], categories=SOURCE_ORDER, ordered=True)
        group_summary["assigned_label"] = pd.Categorical(group_summary["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        group_summary = group_summary.sort_values(["source", "assigned_label", "subgroup", "proposed_group_id"]).reset_index(drop=True)
    if not group_members.empty:
        group_members["source"] = pd.Categorical(group_members["source"], categories=SOURCE_ORDER, ordered=True)
        group_members["assigned_label"] = pd.Categorical(group_members["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        group_members = group_members.sort_values(["source", "assigned_label", "subgroup", "proposed_group_id", "micro_topic_id"]).reset_index(drop=True)
    if not group_edges.empty:
        group_edges["source"] = pd.Categorical(group_edges["source"], categories=SOURCE_ORDER, ordered=True)
        group_edges["assigned_label"] = pd.Categorical(group_edges["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        group_edges = group_edges.sort_values(["source", "assigned_label", "subgroup", "proposed_group_id", "edge_rank_within_group"]).reset_index(drop=True)
    if not singletons.empty:
        singletons["source"] = pd.Categorical(singletons["source"], categories=SOURCE_ORDER, ordered=True)
        singletons["assigned_label"] = pd.Categorical(singletons["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
        singletons = singletons.sort_values(["source", "assigned_label", "subgroup", "micro_topic_id"]).reset_index(drop=True)

    manifest = {
        "row_counts": {
            "raw_topics": int(roster.shape[0]),
            "filtered_candidate_edges": int(filtered_edges.shape[0]),
            "total_components_all_topics": int(total_components),
            "non_singleton_proposed_groups": int(non_singleton_groups),
            "topics_in_non_singletons": int(topics_in_non_singletons),
            "singleton_topic_count": int(singletons.shape[0]),
        }
    }
    return group_summary, group_members, group_edges, singletons, manifest


def instructions_frame() -> pd.DataFrame:
    rows = [
        {
            "section": "Goal",
            "instruction": "Review merge candidates by proposed group rather than by pair. Each proposed group is a connected component built from mutual candidate edges with profile_similarity >= 0.70 inside one subgroup.",
        },
        {
            "section": "Decision",
            "instruction": "Set group_decision in group_summary to accept, reject, or split. Leave singleton rows alone.",
        },
        {
            "section": "Accept",
            "instruction": "Keep the proposed group as-is. All member topics will share the proposed_group_id as the final merge group.",
        },
        {
            "section": "Reject",
            "instruction": "Discard the proposed merge. All topics in that proposed group will become singleton groups.",
        },
        {
            "section": "Split",
            "instruction": "Use manual_split_group_id in group_members to assign every member topic to a subgroup-local split group such as academic_T1_S01.",
        },
        {
            "section": "Evidence",
            "instruction": "Use semantic similarity, Main/POS/Aspect2 overlap, hierarchical proximity, and representative evidence. Do not treat temporal resemblance as a merge criterion in this workbook.",
        },
        {
            "section": "Blank decisions",
            "instruction": "If group_decision is left blank when the mapping is materialized, the default behavior is reject for safety.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    roster = safe_read_csv(args.input_root / "microtopic_roster.csv")
    pairs = safe_read_csv(args.input_root / "merge_candidate_pairs.csv")
    roster, pairs = cast_inputs(roster, pairs)
    filtered_edges = filter_edges(pairs, min_similarity=args.min_similarity)
    group_summary, group_members, group_edges, singletons, manifest = build_components(roster, filtered_edges)

    group_summary_csv = args.output_root / "proposed_group_summary.csv"
    group_members_csv = args.output_root / "proposed_group_members.csv"
    group_edges_csv = args.output_root / "proposed_group_internal_edges.csv"
    singletons_csv = args.output_root / "singleton_topics.csv"
    workbook_path = output_path(args.output_root, "microtopic_merge_first_group_review", ".xlsx")
    manifest_path = args.output_root / "proposed_group_component_manifest.json"
    readme_path = output_path(args.output_root, "README_microtopic_merge_first_group_review", ".md")

    group_summary.to_csv(group_summary_csv, index=False)
    group_members.to_csv(group_members_csv, index=False)
    group_edges.to_csv(group_edges_csv, index=False)
    singletons.to_csv(singletons_csv, index=False)

    instructions = instructions_frame()
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sanitize_frame_for_excel(group_summary).to_excel(writer, sheet_name="group_summary", index=False)
        sanitize_frame_for_excel(group_members).to_excel(writer, sheet_name="group_members", index=False)
        sanitize_frame_for_excel(group_edges).to_excel(writer, sheet_name="group_edges", index=False)
        sanitize_frame_for_excel(singletons).to_excel(writer, sheet_name="singletons", index=False)
        sanitize_frame_for_excel(instructions).to_excel(writer, sheet_name="instructions", index=False)
        workbook_autofit(writer, "group_summary", sanitize_frame_for_excel(group_summary))
        workbook_autofit(writer, "group_members", sanitize_frame_for_excel(group_members))
        workbook_autofit(writer, "group_edges", sanitize_frame_for_excel(group_edges))
        workbook_autofit(writer, "singletons", sanitize_frame_for_excel(singletons))
        workbook_autofit(writer, "instructions", sanitize_frame_for_excel(instructions))

    readme_lines = [
        "# Merge-First Group Review",
        "",
        f"- raw topics in scope: `{int(roster.shape[0])}`",
        f"- filtered candidate edges (`mutual` and `profile_similarity >= {args.min_similarity:.2f}`): `{int(filtered_edges.shape[0])}`",
        f"- evidence mode: `semantic multiaspect (Main + POS + Aspect2 + hierarchy)`",
        f"- total connected components across all topics: `{manifest['row_counts']['total_components_all_topics']}`",
        f"- non-singleton proposed groups to review: `{manifest['row_counts']['non_singleton_proposed_groups']}`",
        f"- topics inside proposed groups: `{manifest['row_counts']['topics_in_non_singletons']}`",
        f"- singleton topics with no group review needed: `{manifest['row_counts']['singleton_topic_count']}`",
        "",
        "## Review flow",
        "1. Review `group_summary` first.",
        "2. Prioritize semantic similarity and Main/POS/Aspect2 agreement; ignore temporal resemblance as merge evidence.",
        "3. Use `accept`, `reject`, or `split` in `group_decision`.",
        "4. Only fill `manual_split_group_id` in `group_members` when `group_decision = split`.",
        "5. Materialize the approved mapping and continue with the existing merge-first downstream pipeline.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    manifest.update(
        {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "group_rule": {
                "selected_from_both_directions": True,
                "profile_similarity_min": args.min_similarity,
                "connected_components_within_subgroup": True,
            },
            "review_mode": "semantic_multiaspect",
            "outputs": {
                "proposed_group_summary_csv": str(group_summary_csv),
                "proposed_group_members_csv": str(group_members_csv),
                "proposed_group_internal_edges_csv": str(group_edges_csv),
                "singleton_topics_csv": str(singletons_csv),
                "review_workbook": str(workbook_path),
                "readme": str(readme_path),
            },
        }
    )
    write_json(manifest_path, manifest)
    print(args.output_root)


if __name__ == "__main__":
    main()
