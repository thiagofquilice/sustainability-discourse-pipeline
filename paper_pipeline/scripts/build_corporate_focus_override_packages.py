#!/usr/bin/env python3
"""Build full and delta Stage 1/2 packages from manual excluded-group overrides."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from microtopic_posthoc_merge_common import (
    CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    CORPORATE_FOCUS_STAGE12_INPUT_ROOT,
    MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
    ensure_directory,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--excluded-workbook",
        type=Path,
        default=Path("paper_pipeline/outputs/corporate_focus_excluded_review/excluded_noncorporate_postmerge_review.xlsx"),
    )
    parser.add_argument(
        "--micro-root",
        type=Path,
        default=MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
    )
    parser.add_argument(
        "--full-review-root",
        type=Path,
        default=Path("paper_pipeline/outputs/corporate_focus_review_with_overrides"),
    )
    parser.add_argument(
        "--delta-review-root",
        type=Path,
        default=Path("paper_pipeline/outputs/corporate_focus_review_override_delta"),
    )
    parser.add_argument(
        "--full-output-root",
        type=Path,
        default=Path("paper_pipeline/outputs/corporate_focus_stage12_input_with_overrides"),
    )
    parser.add_argument(
        "--delta-output-root",
        type=Path,
        default=Path("paper_pipeline/outputs/corporate_focus_stage12_input_override_delta"),
    )
    parser.add_argument("--max-chunks-per-year", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_override_groups(workbook_path: Path) -> pd.DataFrame:
    summary = pd.read_excel(workbook_path, sheet_name="excluded_group_summary")
    yes_mask = summary["include_in_stage12_override"].fillna("").astype(str).str.strip().str.lower() == "yes"
    overrides = summary.loc[yes_mask].copy()
    if overrides.empty:
        raise SystemExit(f"No include_in_stage12_override = yes rows found in {workbook_path}")
    return overrides


def aggregate_override_pairs(overrides: pd.DataFrame, pair_evidence: pd.DataFrame) -> pd.DataFrame:
    frames: list[dict] = []
    for _, row in overrides.iterrows():
        group_id = str(row["final_merge_group_id"])
        matches = pair_evidence.loc[pair_evidence["final_merge_group_id"].astype(str) == group_id].copy()
        if matches.empty:
            best_pair_id = row.get("best_pair_id", "")
            best_similarity = row.get("best_corporate_similarity", "")
            best_group = row.get("best_matched_corporate_group_id", "")
            best_subgroup = row.get("best_matched_corporate_subgroup", "")
            frame_row = {
                "subgroup": row["subgroup"],
                "source": row["source"],
                "assigned_label": row["assigned_label"],
                "micro_topic_id": int(row["micro_topic_id"]),
                "topic_name_original": row["topic_name_original"],
                "final_merge_group_id": group_id,
                "topic_size": int(row["topic_size"]),
                "group_size": int(row["group_size"]),
                "direct_pair_count": 0,
                "max_cosine_similarity": float(best_similarity) if pd.notna(best_similarity) else None,
                "matched_corporate_group_ids_json": json.dumps([best_group] if best_group else []),
                "matched_corporate_subgroups_json": json.dumps([best_subgroup] if best_subgroup else []),
                "justifying_pair_ids_json": json.dumps([best_pair_id] if best_pair_id else []),
                "dyads_json": json.dumps([]),
                "best_pair_id": best_pair_id,
                "best_dyad": "",
                "best_cosine_similarity": float(best_similarity) if pd.notna(best_similarity) else None,
                "best_matched_corporate_group_id": best_group,
                "best_matched_corporate_subgroup": best_subgroup,
                "best_matched_corporate_micro_topic_id": None,
                "best_matched_corporate_topic_name_original": "",
                "inclusion_reason": "manual_override_from_excluded_review",
                "manual_review_notes": row.get("manual_review_notes", ""),
            }
            frames.append(frame_row)
            continue

        matches = matches.sort_values(["cosine_similarity", "pair_id"], ascending=[False, True]).reset_index(drop=True)
        best = matches.iloc[0]
        frames.append(
            {
                "subgroup": row["subgroup"],
                "source": row["source"],
                "assigned_label": row["assigned_label"],
                "micro_topic_id": int(row["micro_topic_id"]),
                "topic_name_original": row["topic_name_original"],
                "final_merge_group_id": group_id,
                "topic_size": int(row["topic_size"]),
                "group_size": int(row["group_size"]),
                "direct_pair_count": int(len(matches)),
                "max_cosine_similarity": float(matches["cosine_similarity"].max()),
                "matched_corporate_group_ids_json": json.dumps(
                    sorted(matches["matched_corporate_group_id"].dropna().astype(str).unique().tolist()),
                    ensure_ascii=False,
                ),
                "matched_corporate_subgroups_json": json.dumps(
                    sorted(matches["matched_corporate_subgroup"].dropna().astype(str).unique().tolist()),
                    ensure_ascii=False,
                ),
                "justifying_pair_ids_json": json.dumps(matches["pair_id"].astype(str).tolist(), ensure_ascii=False),
                "dyads_json": json.dumps(sorted(matches["dyad"].dropna().astype(str).unique().tolist()), ensure_ascii=False),
                "best_pair_id": str(best["pair_id"]),
                "best_dyad": str(best["dyad"]),
                "best_cosine_similarity": float(best["cosine_similarity"]),
                "best_matched_corporate_group_id": str(best["matched_corporate_group_id"]),
                "best_matched_corporate_subgroup": str(best["matched_corporate_subgroup"]),
                "best_matched_corporate_micro_topic_id": int(best["matched_corporate_micro_topic_id"]),
                "best_matched_corporate_topic_name_original": str(best["matched_corporate_topic_name_original"]),
                "inclusion_reason": "manual_override_from_excluded_review",
                "manual_review_notes": row.get("manual_review_notes", ""),
            }
        )
    return pd.DataFrame(frames)


def write_review_root(
    target_root: Path,
    included_corporate: pd.DataFrame,
    included_noncorporate: pd.DataFrame,
    override_noncorporate: pd.DataFrame,
    mode: str,
) -> None:
    ensure_directory(target_root)
    included_corporate.to_csv(target_root / "included_corporate_groups.csv", index=False)
    included_noncorporate.to_csv(target_root / "included_noncorporate_groups.csv", index=False)
    override_noncorporate.to_csv(target_root / "manual_override_noncorporate_groups.csv", index=False)
    write_json(
        target_root / "manual_override_manifest.json",
        {
            "mode": mode,
            "included_corporate_group_count": int(len(included_corporate)),
            "included_noncorporate_group_count": int(len(included_noncorporate)),
            "override_noncorporate_group_count": int(len(override_noncorporate)),
        },
    )


def run_script(script_name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    print(f"[override-packages] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    included_corporate = pd.read_csv(args.review_root / "included_corporate_groups.csv")
    included_noncorporate = pd.read_csv(args.review_root / "included_noncorporate_groups.csv")
    pair_evidence = pd.read_csv(args.review_root / "corporate_focus_pair_evidence.csv")
    overrides = load_override_groups(args.excluded_workbook)
    override_noncorporate = aggregate_override_pairs(overrides, pair_evidence)

    combined_noncorporate = pd.concat([included_noncorporate, override_noncorporate], ignore_index=True)
    combined_noncorporate = combined_noncorporate.drop_duplicates(subset=["final_merge_group_id"], keep="first")
    combined_noncorporate = combined_noncorporate.sort_values(
        ["source", "assigned_label", "subgroup", "micro_topic_id"]
    ).reset_index(drop=True)

    override_noncorporate = override_noncorporate.sort_values(
        ["source", "assigned_label", "subgroup", "micro_topic_id"]
    ).reset_index(drop=True)

    write_review_root(
        args.full_review_root,
        included_corporate=included_corporate,
        included_noncorporate=combined_noncorporate,
        override_noncorporate=override_noncorporate,
        mode="full_with_overrides",
    )
    write_review_root(
        args.delta_review_root,
        included_corporate=included_corporate.iloc[0:0].copy(),
        included_noncorporate=override_noncorporate,
        override_noncorporate=override_noncorporate,
        mode="delta_overrides_only",
    )

    run_script(
        "build_corporate_focus_stage12_inputs.py",
        [
            "--micro-root",
            str(args.micro_root),
            "--review-root",
            str(args.full_review_root),
            "--output-root",
            str(args.full_output_root),
            "--max-chunks-per-year",
            str(args.max_chunks_per_year),
            "--log-level",
            args.log_level,
        ],
    )
    run_script(
        "build_corporate_focus_colab_package.py",
        [
            "--input-root",
            str(args.full_output_root),
            "--zip-name",
            "corporate_focus_stage12_input_with_overrides_upload.zip",
        ],
    )

    run_script(
        "build_corporate_focus_stage12_inputs.py",
        [
            "--micro-root",
            str(args.micro_root),
            "--review-root",
            str(args.delta_review_root),
            "--output-root",
            str(args.delta_output_root),
            "--max-chunks-per-year",
            str(args.max_chunks_per_year),
            "--log-level",
            args.log_level,
        ],
    )
    run_script(
        "build_corporate_focus_colab_package.py",
        [
            "--input-root",
            str(args.delta_output_root),
            "--zip-name",
            "corporate_focus_stage12_input_override_delta_upload.zip",
        ],
    )

    write_json(
        args.full_output_root.parent / "corporate_focus_override_package_manifest.json",
        {
            "base_review_root": str(args.review_root),
            "excluded_workbook": str(args.excluded_workbook),
            "override_group_count": int(len(override_noncorporate)),
            "full_review_root": str(args.full_review_root),
            "delta_review_root": str(args.delta_review_root),
            "full_output_root": str(args.full_output_root),
            "delta_output_root": str(args.delta_output_root),
            "full_zip": str(args.full_output_root.parent / "corporate_focus_stage12_input_with_overrides_upload.zip"),
            "delta_zip": str(args.delta_output_root.parent / "corporate_focus_stage12_input_override_delta_upload.zip"),
        },
    )

    print(args.delta_output_root)


if __name__ == "__main__":
    main()
