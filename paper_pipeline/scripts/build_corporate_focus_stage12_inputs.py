#!/usr/bin/env python3
"""Build Stage 1/2 inputs for the corporate-focused subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_micro_topic_evolution_inputs import build_year_evidence
from micro_topic_evolution_common import SOURCE_ORDER, configure_logging, ensure_directory, read_json, write_json
from microtopic_posthoc_merge_common import (
    CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    CORPORATE_FOCUS_STAGE12_INPUT_ROOT,
    MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-root", type=Path, default=MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED)
    parser.add_argument("--review-root", type=Path, default=CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=CORPORATE_FOCUS_STAGE12_INPUT_ROOT)
    parser.add_argument("--max-chunks-per-year", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def coalesce_merge_columns(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Collapse pandas merge suffix variants back into one canonical column."""
    x_name = f"{column_name}_x"
    y_name = f"{column_name}_y"
    if x_name not in frame.columns and y_name not in frame.columns:
        return frame

    if column_name in frame.columns:
        canonical = frame[column_name].copy()
    else:
        canonical = pd.Series([pd.NA] * len(frame), index=frame.index)

    if x_name in frame.columns:
        canonical = canonical.where(canonical.notna(), frame[x_name])
    if y_name in frame.columns:
        canonical = canonical.where(canonical.notna(), frame[y_name])

    frame[column_name] = canonical
    drop_cols = [name for name in [x_name, y_name] if name in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    return frame


def load_allowed_groups(review_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    corporate_path = review_root / "included_corporate_groups.csv"
    noncorporate_path = review_root / "included_noncorporate_groups.csv"
    if not corporate_path.exists() or not noncorporate_path.exists():
        raise SystemExit(
            "Corporate focus review outputs are missing. Expected: "
            f"{corporate_path} and {noncorporate_path}"
        )
    corporate = pd.read_csv(corporate_path)
    noncorporate = pd.read_csv(noncorporate_path)
    allowed = set(corporate.get("final_merge_group_id", pd.Series(dtype=str)).astype(str).tolist())
    allowed.update(noncorporate.get("final_merge_group_id", pd.Series(dtype=str)).astype(str).tolist())
    if not allowed:
        raise SystemExit("No allowed groups found in the corporate focus review outputs.")
    return corporate, noncorporate, allowed


def select_subset(topic_info: pd.DataFrame, allowed_groups: set[str]) -> pd.DataFrame:
    valid = topic_info.loc[topic_info["Topic"] != -1].copy()
    valid = valid.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
    total = float(valid["Count"].sum())
    if total > 0:
        valid["share_of_non_outlier"] = valid["Count"] / total
    else:
        valid["share_of_non_outlier"] = 0.0
    selected = valid.loc[valid["final_merge_group_id"].astype(str).isin(allowed_groups)].copy()
    selected = selected.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
    selected["cumulative_share"] = selected["share_of_non_outlier"].cumsum()
    selected["approved_for_corporate_focus"] = True
    return selected


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    included_corporate, included_noncorporate, allowed_groups = load_allowed_groups(args.review_root)
    selected_topic_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    yearly_rows: list[dict] = []

    subgroup_dirs = sorted(path for path in args.micro_root.iterdir() if path.is_dir() and path.name != "summary")
    for subgroup_dir in subgroup_dirs:
        manifest_path = subgroup_dir / "manifest.json"
        topic_info_path = subgroup_dir / "topic_info.csv"
        topics_over_time_path = subgroup_dir / "topics_over_time.csv"
        document_topics_path = subgroup_dir / "document_topics.parquet"
        if not all(path.exists() for path in [manifest_path, topic_info_path, topics_over_time_path, document_topics_path]):
            continue

        manifest = read_json(manifest_path)
        topic_info = pd.read_csv(topic_info_path)
        topics_over_time = pd.read_csv(topics_over_time_path)
        document_topics = pd.read_parquet(document_topics_path)
        selected = select_subset(topic_info, allowed_groups)
        if selected.empty:
            subgroup_rows.append(
                {
                    "subgroup": subgroup_dir.name,
                    "source": manifest["source"],
                    "assigned_label": manifest["assigned_label"],
                    "status": manifest["status"],
                    "warning": manifest.get("warning", ""),
                    "row_count": int(manifest["row_count"]),
                    "selected_topic_count": 0,
                    "selected_topic_coverage": 0.0,
                    "selected_topic_ids": json.dumps([]),
                    "selected_topic_names": json.dumps([]),
                    "selected_final_merge_group_ids": json.dumps([]),
                }
            )
            continue

        topic_rows, group_year_rows = build_year_evidence(
            subgroup=subgroup_dir.name,
            manifest=manifest,
            selected_topics=selected,
            topics_over_time=topics_over_time,
            document_topics=document_topics,
            max_chunks_per_year=args.max_chunks_per_year,
        )

        approved_meta = selected[
            [
                "Topic",
                "Name",
                "Count",
                "share_of_non_outlier",
                "cumulative_share",
                "approved_for_corporate_focus",
                "final_merge_group_id",
                "group_size",
                "is_singleton_group",
                "member_micro_topic_ids_json",
            ]
        ].copy()
        approved_meta = approved_meta.rename(columns={"Topic": "micro_topic_id", "Name": "topic_name_original"})
        topic_rows_frame = pd.DataFrame(topic_rows).merge(
            approved_meta,
            on=["micro_topic_id", "topic_name_original"],
            how="left",
        )
        for canonical_name in ["share_of_non_outlier", "cumulative_share"]:
            topic_rows_frame = coalesce_merge_columns(topic_rows_frame, canonical_name)
        selected_topic_rows.extend(topic_rows_frame.to_dict(orient="records"))
        yearly_rows.extend(group_year_rows)

        subgroup_rows.append(
            {
                "subgroup": subgroup_dir.name,
                "source": manifest["source"],
                "assigned_label": manifest["assigned_label"],
                "status": manifest["status"],
                "warning": manifest.get("warning", ""),
                "row_count": int(manifest["row_count"]),
                "selected_topic_count": int(len(selected)),
                "selected_topic_coverage": float(selected["share_of_non_outlier"].sum()),
                "selected_topic_ids": json.dumps(selected["Topic"].astype(int).tolist()),
                "selected_topic_names": json.dumps(selected["Name"].astype(str).tolist(), ensure_ascii=False),
                "selected_final_merge_group_ids": json.dumps(selected["final_merge_group_id"].astype(str).tolist(), ensure_ascii=False),
            }
        )

    selected_topics_frame = pd.DataFrame(selected_topic_rows)
    subgroup_frame = pd.DataFrame(subgroup_rows)
    yearly_frame = pd.DataFrame(yearly_rows)

    if not selected_topics_frame.empty:
        selected_topics_frame["source"] = pd.Categorical(selected_topics_frame["source"], categories=SOURCE_ORDER, ordered=True)
        selected_topics_frame = selected_topics_frame.sort_values(["source", "assigned_label", "micro_topic_id"]).reset_index(drop=True)
    if not subgroup_frame.empty:
        subgroup_frame["source"] = pd.Categorical(subgroup_frame["source"], categories=SOURCE_ORDER, ordered=True)
        subgroup_frame = subgroup_frame.sort_values(["source", "assigned_label"]).reset_index(drop=True)
    if not yearly_frame.empty:
        yearly_frame["source"] = pd.Categorical(yearly_frame["source"], categories=SOURCE_ORDER, ordered=True)
        yearly_frame = yearly_frame.sort_values(["source", "assigned_label", "micro_topic_id", "year"]).reset_index(drop=True)

    selected_topics_csv = args.output_root / "selected_micro_topics.csv"
    subgroup_csv = args.output_root / "selected_micro_topics_by_subgroup.csv"
    year_evidence_csv = args.output_root / "micro_topic_year_evidence.csv"
    year_evidence_jsonl = args.output_root / "micro_topic_year_evidence.jsonl"
    included_corporate_copy = args.output_root / "included_corporate_groups.csv"
    included_noncorporate_copy = args.output_root / "included_noncorporate_groups.csv"

    selected_topics_frame.to_csv(selected_topics_csv, index=False)
    subgroup_frame.to_csv(subgroup_csv, index=False)
    yearly_frame.to_csv(year_evidence_csv, index=False)
    write_jsonl(year_evidence_jsonl, yearly_frame)
    included_corporate.to_csv(included_corporate_copy, index=False)
    included_noncorporate.to_csv(included_noncorporate_copy, index=False)

    write_json(
        args.output_root / "selection_manifest.json",
        {
            "micro_root": str(args.micro_root),
            "review_root": str(args.review_root),
            "selection_mode": "corporate_focused_direct_pairs_no_mnn",
            "selection_rule": {
                "include_all_corporate_groups": True,
                "include_noncorporate_groups_with_direct_pair_to_corporate": True,
                "use_mutual_nearest_neighbor": False,
                "allow_indirect_academic_media_inclusion": False,
            },
            "max_chunks_per_year": args.max_chunks_per_year,
            "subgroup_count": int(subgroup_frame.shape[0]),
            "selected_micro_topic_count": int(selected_topics_frame.shape[0]),
            "annual_evidence_row_count": int(yearly_frame.shape[0]),
            "included_corporate_group_count": int(included_corporate.shape[0]),
            "included_noncorporate_group_count": int(included_noncorporate.shape[0]),
            "outputs": {
                "selected_micro_topics": str(selected_topics_csv),
                "selected_micro_topics_by_subgroup": str(subgroup_csv),
                "micro_topic_year_evidence": str(year_evidence_csv),
                "micro_topic_year_evidence_jsonl": str(year_evidence_jsonl),
                "included_corporate_groups": str(included_corporate_copy),
                "included_noncorporate_groups": str(included_noncorporate_copy),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
