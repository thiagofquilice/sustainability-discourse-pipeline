#!/usr/bin/env python3
"""Build an auditable corporate-focused review layer from direct cross-source semantic pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from cross_source_microtopic_common import SOURCE_ORDER, build_pair_id, configure_logging, write_json
from microtopic_posthoc_merge_common import (
    CORPORATE_FOCUS_PAIR_OUTPUT_ROOT,
    CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
    ensure_directory,
    json_dumps,
    normalize_text,
    sanitize_frame_for_excel,
    workbook_autofit,
)


DIRECT_DYADS = {"academic__corporate", "media__corporate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-micro-root", type=Path, default=MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED)
    parser.add_argument("--pair-root", type=Path, default=CORPORATE_FOCUS_PAIR_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT)
    parser.add_argument("--similarity-threshold", type=float, default=0.60)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_group_catalog(merged_micro_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subgroup_dirs = sorted(path for path in merged_micro_root.iterdir() if path.is_dir() and path.name != "summary")
    for subgroup_dir in subgroup_dirs:
        manifest_path = subgroup_dir / "manifest.json"
        topic_info_path = subgroup_dir / "topic_info.csv"
        if not manifest_path.exists() or not topic_info_path.exists():
            continue
        manifest = pd.read_json(manifest_path, typ="series")
        topic_info = pd.read_csv(topic_info_path)
        if topic_info.empty:
            continue
        required = {"Topic", "Name", "Count", "final_merge_group_id"}
        missing = required - set(topic_info.columns)
        if missing:
            raise SystemExit(
                f"{topic_info_path} is missing required merged-root columns: {sorted(missing)}"
            )
        frame = topic_info.copy()
        frame["Topic"] = pd.to_numeric(frame["Topic"], errors="coerce")
        frame = frame.dropna(subset=["Topic"]).copy()
        frame["Topic"] = frame["Topic"].astype(int)
        frame = frame.rename(
            columns={
                "Topic": "micro_topic_id",
                "Name": "topic_name_original",
                "Count": "topic_size",
            }
        )
        frame["subgroup"] = subgroup_dir.name
        frame["source"] = str(manifest.get("source", ""))
        frame["assigned_label"] = str(manifest.get("assigned_label", ""))
        frame["macro_topic_name"] = str(manifest.get("topic_name", ""))
        for _, row in frame.iterrows():
            rows.append(
                {
                    "subgroup": str(row["subgroup"]),
                    "source": str(row["source"]),
                    "assigned_label": str(row["assigned_label"]),
                    "macro_topic_name": str(row["macro_topic_name"]),
                    "micro_topic_id": int(row["micro_topic_id"]),
                    "topic_name_original": str(row["topic_name_original"]),
                    "topic_size": int(row["topic_size"]),
                    "final_merge_group_id": str(row["final_merge_group_id"]),
                    "group_size": int(row["group_size"]) if "group_size" in frame.columns else 1,
                    "is_singleton_group": bool(row["is_singleton_group"]) if "is_singleton_group" in frame.columns else True,
                    "member_micro_topic_ids_json": str(row.get("member_micro_topic_ids_json", "[]")),
                    "member_topic_names_json": str(row.get("member_topic_names_json", "[]")),
                }
            )
    catalog = pd.DataFrame(rows)
    if not catalog.empty:
        catalog["source"] = pd.Categorical(catalog["source"], categories=SOURCE_ORDER, ordered=True)
        catalog = catalog.sort_values(["source", "assigned_label", "micro_topic_id"]).reset_index(drop=True)
    return catalog


def merge_side_metadata(frame: pd.DataFrame, catalog: pd.DataFrame, side: str) -> pd.DataFrame:
    side_catalog = catalog.rename(
        columns={
            "subgroup": f"subgroup_{side}",
            "micro_topic_id": f"microtopic_id_{side}",
            "source": f"source_{side}",
            "assigned_label": f"assigned_label_{side}",
            "macro_topic_name": f"macro_topic_name_{side}",
            "topic_name_original": f"topic_name_original_{side}",
            "topic_size": f"topic_size_{side}",
            "final_merge_group_id": f"final_merge_group_id_{side}",
            "group_size": f"group_size_{side}",
            "is_singleton_group": f"is_singleton_group_{side}",
            "member_micro_topic_ids_json": f"member_micro_topic_ids_json_{side}",
            "member_topic_names_json": f"member_topic_names_json_{side}",
        }
    )
    merged = frame.merge(
        side_catalog,
        on=[f"subgroup_{side}", f"microtopic_id_{side}", f"source_{side}"],
        how="left",
    )
    missing = merged.loc[merged[f"final_merge_group_id_{side}"].isna()]
    if not missing.empty:
        raise SystemExit(
            f"Failed to map merged-group metadata for side {side}; missing rows: "
            + ", ".join(
                f"{row.subgroup_a}:{row.microtopic_id_a} vs {row.subgroup_b}:{row.microtopic_id_b}"
                for row in missing.head(5).itertuples(index=False)
            )
        )
    return merged


def build_pair_evidence(pair_root: Path, catalog: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_path = pair_root / "all_pairwise_similarities.csv"
    if not pairwise_path.exists():
        raise SystemExit(f"Missing all pairwise similarities file: {pairwise_path}")
    all_pairs = pd.read_csv(pairwise_path)
    if all_pairs.empty:
        raise SystemExit(f"No pairwise similarities found in {pairwise_path}")

    direct_pairs = all_pairs.loc[all_pairs["dyad"].isin(DIRECT_DYADS)].copy()
    direct_pairs["pair_id"] = direct_pairs.apply(
        lambda row: build_pair_id(
            dyad=str(row["dyad"]),
            macro_topic=str(row["macro_topic"]),
            source_a=str(row["source_a"]),
            microtopic_id_a=int(row["microtopic_id_a"]),
            source_b=str(row["source_b"]),
            microtopic_id_b=int(row["microtopic_id_b"]),
        ),
        axis=1,
    )
    direct_pairs = merge_side_metadata(direct_pairs, catalog, "a")
    direct_pairs = merge_side_metadata(direct_pairs, catalog, "b")

    direct_pairs["noncorporate_subgroup"] = direct_pairs["subgroup_a"]
    direct_pairs["noncorporate_source"] = direct_pairs["source_a"]
    direct_pairs["noncorporate_assigned_label"] = direct_pairs["assigned_label_a"]
    direct_pairs["noncorporate_micro_topic_id"] = direct_pairs["microtopic_id_a"]
    direct_pairs["noncorporate_topic_name_original"] = direct_pairs["topic_name_original_a"]
    direct_pairs["noncorporate_topic_size"] = direct_pairs["topic_size_a"]
    direct_pairs["noncorporate_final_merge_group_id"] = direct_pairs["final_merge_group_id_a"]
    direct_pairs["noncorporate_group_size"] = direct_pairs["group_size_a"]
    direct_pairs["matched_corporate_subgroup"] = direct_pairs["subgroup_b"]
    direct_pairs["matched_corporate_micro_topic_id"] = direct_pairs["microtopic_id_b"]
    direct_pairs["matched_corporate_topic_name_original"] = direct_pairs["topic_name_original_b"]
    direct_pairs["matched_corporate_topic_size"] = direct_pairs["topic_size_b"]
    direct_pairs["matched_corporate_group_id"] = direct_pairs["final_merge_group_id_b"]

    qualifying = direct_pairs.loc[direct_pairs["cosine_similarity"] >= threshold].copy()
    qualifying = qualifying.sort_values(
        [
            "noncorporate_source",
            "noncorporate_assigned_label",
            "noncorporate_micro_topic_id",
            "cosine_similarity",
            "matched_corporate_micro_topic_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    return direct_pairs, qualifying


def build_included_noncorporate(qualifying: pd.DataFrame) -> pd.DataFrame:
    if qualifying.empty:
        return pd.DataFrame()
    key_cols = [
        "noncorporate_subgroup",
        "noncorporate_source",
        "noncorporate_assigned_label",
        "noncorporate_micro_topic_id",
        "noncorporate_topic_name_original",
        "noncorporate_final_merge_group_id",
        "noncorporate_topic_size",
        "noncorporate_group_size",
    ]
    best = (
        qualifying.sort_values(["cosine_similarity", "matched_corporate_group_id"], ascending=[False, True])
        .groupby(key_cols, dropna=False)
        .head(1)
        .copy()
    )
    agg = (
        qualifying.groupby(key_cols, dropna=False)
        .agg(
            direct_pair_count=("pair_id", "nunique"),
            max_cosine_similarity=("cosine_similarity", "max"),
            matched_corporate_group_ids_json=(
                "matched_corporate_group_id",
                lambda values: json_dumps(sorted({str(value) for value in values})),
            ),
            matched_corporate_subgroups_json=(
                "matched_corporate_subgroup",
                lambda values: json_dumps(sorted({str(value) for value in values})),
            ),
            justifying_pair_ids_json=(
                "pair_id",
                lambda values: json_dumps(sorted({str(value) for value in values})),
            ),
            dyads_json=("dyad", lambda values: json_dumps(sorted({str(value) for value in values}))),
        )
        .reset_index()
    )
    frame = agg.merge(
        best[
            key_cols
            + [
                "pair_id",
                "dyad",
                "cosine_similarity",
                "matched_corporate_group_id",
                "matched_corporate_subgroup",
                "matched_corporate_micro_topic_id",
                "matched_corporate_topic_name_original",
            ]
        ].rename(
            columns={
                "pair_id": "best_pair_id",
                "dyad": "best_dyad",
                "cosine_similarity": "best_cosine_similarity",
                "matched_corporate_group_id": "best_matched_corporate_group_id",
                "matched_corporate_subgroup": "best_matched_corporate_subgroup",
                "matched_corporate_micro_topic_id": "best_matched_corporate_micro_topic_id",
                "matched_corporate_topic_name_original": "best_matched_corporate_topic_name_original",
            }
        ),
        on=key_cols,
        how="left",
    )
    frame["inclusion_reason"] = "direct_pair_with_corporate_above_threshold"
    return frame.rename(
        columns={
            "noncorporate_subgroup": "subgroup",
            "noncorporate_source": "source",
            "noncorporate_assigned_label": "assigned_label",
            "noncorporate_micro_topic_id": "micro_topic_id",
            "noncorporate_topic_name_original": "topic_name_original",
            "noncorporate_final_merge_group_id": "final_merge_group_id",
            "noncorporate_topic_size": "topic_size",
            "noncorporate_group_size": "group_size",
        }
    )


def build_included_corporate(catalog: pd.DataFrame) -> pd.DataFrame:
    frame = catalog.loc[catalog["source"] == "corporate"].copy()
    frame["approved_for_corporate_focus"] = True
    frame["inclusion_reason"] = "all_corporate_groups_included"
    return frame.reset_index(drop=True)


def build_excluded_noncorporate(catalog: pd.DataFrame, direct_pairs: pd.DataFrame, included: pd.DataFrame, threshold: float) -> pd.DataFrame:
    noncorporate = catalog.loc[catalog["source"].isin(["academic", "media"])].copy()
    include_keys = (
        included[["subgroup", "micro_topic_id"]].assign(_include=True)
        if not included.empty
        else pd.DataFrame(columns=["subgroup", "micro_topic_id", "_include"])
    )
    frame = noncorporate.merge(include_keys, on=["subgroup", "micro_topic_id"], how="left")
    frame = frame.loc[frame["_include"] != True].drop(columns=["_include"]).copy()  # noqa: E712

    if direct_pairs.empty:
        frame["best_corporate_similarity"] = pd.NA
        frame["best_pair_id"] = ""
        frame["best_matched_corporate_group_id"] = ""
        frame["best_matched_corporate_subgroup"] = ""
    else:
        best = (
            direct_pairs.sort_values(["cosine_similarity", "matched_corporate_group_id"], ascending=[False, True])
            .groupby(["noncorporate_subgroup", "noncorporate_micro_topic_id"], dropna=False)
            .head(1)
            .rename(
                columns={
                    "noncorporate_subgroup": "subgroup",
                    "noncorporate_micro_topic_id": "micro_topic_id",
                    "pair_id": "best_pair_id",
                    "cosine_similarity": "best_corporate_similarity",
                    "matched_corporate_group_id": "best_matched_corporate_group_id",
                    "matched_corporate_subgroup": "best_matched_corporate_subgroup",
                }
            )[
                [
                    "subgroup",
                    "micro_topic_id",
                    "best_pair_id",
                    "best_corporate_similarity",
                    "best_matched_corporate_group_id",
                    "best_matched_corporate_subgroup",
                ]
            ]
        )
        frame = frame.merge(best, on=["subgroup", "micro_topic_id"], how="left")
    frame["exclusion_reason"] = frame["best_corporate_similarity"].apply(
        lambda value: (
            "no_direct_pair_with_corporate_above_threshold"
            if pd.isna(value) or float(value) < threshold
            else "review_needed"
        )
    )
    return frame.reset_index(drop=True)


def write_workbook(
    output_root: Path,
    included_corporate: pd.DataFrame,
    included_noncorporate: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    excluded_noncorporate: pd.DataFrame,
    threshold: float,
) -> Path:
    workbook_path = output_root / "corporate_focus_review.xlsx"
    instructions = pd.DataFrame(
        [
            {
                "rule": "corporate groups",
                "detail": "All corporate merged groups are included automatically.",
            },
            {
                "rule": "academic/media groups",
                "detail": (
                    "A non-corporate group is included when it has at least one direct pair with corporate "
                    f"and cosine_similarity >= {threshold:.2f}."
                ),
            },
            {
                "rule": "MNN",
                "detail": "Mutual nearest neighbor is intentionally not used in this review.",
            },
            {
                "rule": "indirect links",
                "detail": "Academic-media links do not justify inclusion here; only direct ties to corporate count.",
            },
        ]
    )
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sheets = {
            "included_corporate": included_corporate,
            "included_noncorporate": included_noncorporate,
            "pair_evidence": pair_evidence,
            "excluded_noncorporate": excluded_noncorporate,
            "instructions": instructions,
        }
        for sheet_name, frame in sheets.items():
            cleaned = sanitize_frame_for_excel(frame)
            cleaned.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook_autofit(writer, sheet_name, cleaned)
    return workbook_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)

    catalog = load_group_catalog(args.merged_micro_root)
    if catalog.empty:
        raise SystemExit(f"No merged microtopic groups found under {args.merged_micro_root}")

    direct_pairs, qualifying = build_pair_evidence(
        pair_root=args.pair_root,
        catalog=catalog,
        threshold=args.similarity_threshold,
    )

    pair_evidence = qualifying[
        [
            "pair_id",
            "dyad",
            "macro_topic",
            "macro_topic_name",
            "cosine_similarity",
            "noncorporate_source",
            "noncorporate_subgroup",
            "noncorporate_assigned_label",
            "noncorporate_micro_topic_id",
            "noncorporate_topic_name_original",
            "noncorporate_final_merge_group_id",
            "noncorporate_topic_size",
            "matched_corporate_subgroup",
            "matched_corporate_micro_topic_id",
            "matched_corporate_topic_name_original",
            "matched_corporate_group_id",
            "matched_corporate_topic_size",
        ]
    ].rename(
        columns={
            "noncorporate_source": "source",
            "noncorporate_subgroup": "subgroup",
            "noncorporate_assigned_label": "assigned_label",
            "noncorporate_micro_topic_id": "micro_topic_id",
            "noncorporate_topic_name_original": "topic_name_original",
            "noncorporate_final_merge_group_id": "final_merge_group_id",
            "noncorporate_topic_size": "topic_size",
        }
    )

    included_corporate = build_included_corporate(catalog)
    included_noncorporate = build_included_noncorporate(qualifying)
    excluded_noncorporate = build_excluded_noncorporate(
        catalog=catalog,
        direct_pairs=direct_pairs,
        included=included_noncorporate,
        threshold=args.similarity_threshold,
    )

    if not included_corporate.empty:
        included_corporate["source"] = pd.Categorical(included_corporate["source"], categories=SOURCE_ORDER, ordered=True)
        included_corporate = included_corporate.sort_values(["source", "assigned_label", "micro_topic_id"]).reset_index(drop=True)
    if not included_noncorporate.empty:
        included_noncorporate["source"] = pd.Categorical(included_noncorporate["source"], categories=SOURCE_ORDER, ordered=True)
        included_noncorporate = included_noncorporate.sort_values(["source", "assigned_label", "best_cosine_similarity"], ascending=[True, True, False]).reset_index(drop=True)
    if not pair_evidence.empty:
        pair_evidence["source"] = pd.Categorical(pair_evidence["source"], categories=SOURCE_ORDER, ordered=True)
        pair_evidence = pair_evidence.sort_values(["source", "assigned_label", "cosine_similarity"], ascending=[True, True, False]).reset_index(drop=True)
    if not excluded_noncorporate.empty:
        excluded_noncorporate["source"] = pd.Categorical(excluded_noncorporate["source"], categories=SOURCE_ORDER, ordered=True)
        excluded_noncorporate = excluded_noncorporate.sort_values(["source", "assigned_label", "best_corporate_similarity"], ascending=[True, True, False]).reset_index(drop=True)

    included_corporate_path = args.output_root / "included_corporate_groups.csv"
    included_noncorporate_path = args.output_root / "included_noncorporate_groups.csv"
    pair_evidence_path = args.output_root / "corporate_focus_pair_evidence.csv"
    excluded_noncorporate_path = args.output_root / "excluded_noncorporate_groups.csv"
    manifest_path = args.output_root / "corporate_focus_review_manifest.json"

    included_corporate.to_csv(included_corporate_path, index=False)
    included_noncorporate.to_csv(included_noncorporate_path, index=False)
    pair_evidence.to_csv(pair_evidence_path, index=False)
    excluded_noncorporate.to_csv(excluded_noncorporate_path, index=False)
    workbook_path = write_workbook(
        output_root=args.output_root,
        included_corporate=included_corporate,
        included_noncorporate=included_noncorporate,
        pair_evidence=pair_evidence,
        excluded_noncorporate=excluded_noncorporate,
        threshold=args.similarity_threshold,
    )

    write_json(
        manifest_path,
        {
            "merged_micro_root": str(args.merged_micro_root),
            "pair_root": str(args.pair_root),
            "similarity_threshold": args.similarity_threshold,
            "review_rule": {
                "include_all_corporate": True,
                "include_noncorporate_if_any_direct_pair_to_corporate_at_or_above_threshold": True,
                "use_mutual_nearest_neighbor": False,
                "allow_indirect_academic_media_inclusion": False,
            },
            "row_counts": {
                "group_catalog_rows": int(catalog.shape[0]),
                "included_corporate_groups": int(included_corporate.shape[0]),
                "included_noncorporate_groups": int(included_noncorporate.shape[0]),
                "pair_evidence_rows": int(pair_evidence.shape[0]),
                "excluded_noncorporate_groups": int(excluded_noncorporate.shape[0]),
            },
            "outputs": {
                "included_corporate_groups_csv": str(included_corporate_path),
                "included_noncorporate_groups_csv": str(included_noncorporate_path),
                "corporate_focus_pair_evidence_csv": str(pair_evidence_path),
                "excluded_noncorporate_groups_csv": str(excluded_noncorporate_path),
                "corporate_focus_review_workbook": str(workbook_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
