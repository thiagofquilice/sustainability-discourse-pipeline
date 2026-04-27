#!/usr/bin/env python3
"""Build a publication-oriented audit sample from the 6-topic Gemma pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_INPUT = WORKFLOW_ROOT / "outputs" / "pilot_1000" / "input" / "six_topic_pilot_1000.parquet"
DEFAULT_PILOT_OUTPUT = WORKFLOW_ROOT / "outputs" / "pilot_1000" / "output" / "validation_output.csv"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "manual_review" / "paper_audit_pilot1000"

REVIEW_COLUMNS = [
    "sample_id",
    "review_order",
    "audit_panel",
    "sampling_reason",
    "source",
    "year",
    "chunk_id",
    "source_doc_id",
    "assigned_label",
    "topic_name",
    "assigned_label_text",
    "best_score",
    "best_score_band",
    "second_best_score",
    "score_gap",
    "score_gap_band",
    "n_candidates_above_threshold",
    "candidate_complexity",
    "v_gemma",
    "text_full",
    "paper_manual_decision",
    "paper_manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-input", type=Path, default=DEFAULT_PILOT_INPUT)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_bands(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    best_bins = result["best_score"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    best_bins = np.unique(best_bins)
    if len(best_bins) < 4:
        best_bins = np.array([
            result["best_score"].min() - 1e-9,
            result["best_score"].quantile(0.5),
            result["best_score"].quantile(0.9),
            result["best_score"].max() + 1e-9,
        ])
    else:
        best_bins[0] -= 1e-9
        best_bins[-1] += 1e-9
    result["best_score_band"] = pd.cut(
        result["best_score"],
        bins=best_bins,
        labels=["lower_score", "mid_score", "higher_score"],
        include_lowest=True,
    ).astype(str)

    gap_bins = result["score_gap"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    gap_bins = np.unique(gap_bins)
    if len(gap_bins) < 4:
        gap_bins = np.array([
            result["score_gap"].min() - 1e-9,
            result["score_gap"].quantile(0.5),
            result["score_gap"].quantile(0.9),
            result["score_gap"].max() + 1e-9,
        ])
    else:
        gap_bins[0] -= 1e-9
        gap_bins[-1] += 1e-9
    result["score_gap_band"] = pd.cut(
        result["score_gap"],
        bins=gap_bins,
        labels=["ambiguous_gap", "mid_gap", "clear_gap"],
        include_lowest=True,
    ).astype(str)

    def candidate_band(value: float) -> str:
        if pd.isna(value):
            return "unknown_candidates"
        value = int(value)
        if value <= 1:
            return "single_candidate"
        if value == 2:
            return "two_candidates"
        return "three_plus_candidates"

    result["candidate_complexity"] = result["n_candidates_above_threshold"].apply(candidate_band)
    return result


def hard_positive_rank(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    ranked = frame.sample(frac=1.0, random_state=seed).copy()
    return ranked.sort_values(
        ["n_candidates_above_threshold", "score_gap", "best_score", "year"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )


def select_yes_rows(yes_frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or yes_frame.empty:
        return yes_frame.head(0).copy()
    ranked = hard_positive_rank(yes_frame, seed)
    return ranked.head(min(n, len(ranked))).copy()


def build_sample(merged: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    parts: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {
        "total_rows": int(len(merged)),
        "total_yes": int((merged["v_gemma"] == "yes").sum()),
        "total_no": int((merged["v_gemma"] == "no").sum()),
        "unmatched_no_cells": [],
        "coverage_yes_cells": [],
    }

    no_rows = merged.loc[merged["v_gemma"] == "no"].copy()
    no_rows["audit_panel"] = "all_no"
    no_rows["sampling_reason"] = "Include every Gemma no from the pilot to preserve the full set of rejections."
    parts.append(no_rows)

    grouped = merged.groupby(["source", "assigned_label"], dropna=False)
    matched_yes_parts: list[pd.DataFrame] = []
    coverage_yes_parts: list[pd.DataFrame] = []

    for idx, ((source, label), group) in enumerate(grouped):
        no_group = group.loc[group["v_gemma"] == "no"].copy()
        yes_group = group.loc[group["v_gemma"] == "yes"].copy()
        cell_seed = seed + idx * 17
        if len(no_group) > 0:
            chosen_yes = select_yes_rows(yes_group, len(no_group), cell_seed)
            if not chosen_yes.empty:
                chosen_yes["audit_panel"] = "matched_yes"
                chosen_yes["sampling_reason"] = "Matched Gemma yes from the same source-topic cell, prioritizing harder positives."
                matched_yes_parts.append(chosen_yes)
            if len(chosen_yes) < len(no_group):
                diagnostics["unmatched_no_cells"].append(
                    {
                        "source": source,
                        "assigned_label": label,
                        "no_rows": int(len(no_group)),
                        "available_yes_rows": int(len(yes_group)),
                        "matched_yes_rows": int(len(chosen_yes)),
                    }
                )
        elif len(yes_group) > 0:
            chosen_yes = select_yes_rows(yes_group, 1, cell_seed)
            if not chosen_yes.empty:
                chosen_yes["audit_panel"] = "coverage_yes"
                chosen_yes["sampling_reason"] = "Add one Gemma yes from a source-topic cell with no Gemma no so the audit still covers that cell."
                coverage_yes_parts.append(chosen_yes)
                diagnostics["coverage_yes_cells"].append({"source": source, "assigned_label": label, "rows": int(len(chosen_yes))})

    if matched_yes_parts:
        parts.append(pd.concat(matched_yes_parts, ignore_index=True))
    if coverage_yes_parts:
        parts.append(pd.concat(coverage_yes_parts, ignore_index=True))

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.rename(columns={"text": "text_full"})
    sample["sample_id"] = [f"six_topic_paper_{i + 1:03d}" for i in range(len(sample))]
    review_order = sample.sample(frac=1.0, random_state=seed + 9000).reset_index()[["index"]]
    review_order["review_order"] = np.arange(1, len(review_order) + 1)
    sample = sample.reset_index().merge(review_order, on="index", how="left").drop(columns=["index"])
    sample["paper_manual_decision"] = pd.NA
    sample["paper_manual_notes"] = pd.NA
    sample = sample[REVIEW_COLUMNS].sort_values(["review_order"]).reset_index(drop=True)

    diagnostics["final_rows"] = int(len(sample))
    diagnostics["panel_counts"] = {key: int(value) for key, value in sample["audit_panel"].value_counts().to_dict().items()}
    diagnostics["source_panel_counts"] = (
        sample.groupby(["source", "audit_panel"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
    )
    return sample, diagnostics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pilot_input = pd.read_parquet(args.pilot_input)
    pilot_output = pd.read_csv(args.pilot_output)
    merged = pilot_input.merge(pilot_output[["chunk_id", "v_gemma"]], on="chunk_id", how="inner", validate="one_to_one")
    merged = add_bands(merged)

    sample, diagnostics = build_sample(merged, args.seed)

    csv_path = args.output_root / "six_topic_paper_audit_sample.csv"
    xlsx_path = args.output_root / "six_topic_paper_audit_sample.xlsx"
    manifest_path = args.output_root / "six_topic_paper_audit_manifest.json"
    summary_path = args.output_root / "six_topic_paper_audit_summary.csv"
    by_cell_path = args.output_root / "six_topic_paper_audit_by_source_label.csv"
    method_path = args.output_root / "six_topic_paper_audit_method.txt"

    sample.to_csv(csv_path, index=False)

    summary = (
        sample.groupby(["audit_panel", "source", "v_gemma"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["audit_panel", "source", "v_gemma"])
    )
    summary.to_csv(summary_path, index=False)

    by_cell = (
        sample.groupby(["audit_panel", "source", "assigned_label", "topic_name", "v_gemma"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["audit_panel", "source", "assigned_label", "v_gemma"])
    )
    by_cell.to_csv(by_cell_path, index=False)

    method_text = (
        "Paper-oriented audit design for the 6-topic 1000-row Gemma pilot\n\n"
        "1. Include every Gemma no from the pilot.\n"
        "2. For each source-topic cell with Gemma no, add the same number of Gemma yes from that same cell when available, prioritizing harder positives.\n"
        "3. For source-topic cells that have Gemma yes but zero Gemma no, add one coverage Gemma yes so the audit still spans those cells.\n"
        "4. Randomize review order to reduce sequence effects during manual coding.\n"
    )
    method_path.write_text(method_text, encoding="utf-8")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"sheet_name": "paper_audit_review", "description": "Main review sheet in randomized review order."},
                {"sheet_name": "summary", "description": "Counts by panel, source, and Gemma decision."},
                {"sheet_name": "by_source_label", "description": "Counts by panel, source, topic, and Gemma decision."},
                {"sheet_name": "method", "description": "Sampling logic for the audit sample."},
            ]
        ).to_excel(writer, sheet_name="readme", index=False)
        sample.to_excel(writer, sheet_name="paper_audit_review", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_cell.to_excel(writer, sheet_name="by_source_label", index=False)
        pd.DataFrame({"method": method_text.splitlines()}).to_excel(writer, sheet_name="method", index=False)

    manifest_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
