#!/usr/bin/env python3
"""Build a 150-row post-Gemma manual review workbook from the 6-topic pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCE_ORDER = ["academic", "corporate", "media"]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_INPUT = WORKFLOW_ROOT / "outputs" / "pilot_1000" / "input" / "six_topic_pilot_1000.parquet"
DEFAULT_PILOT_OUTPUT = WORKFLOW_ROOT / "outputs" / "pilot_1000" / "output" / "validation_output.csv"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs" / "manual_review" / "pilot_1000_post_gemma"
ROWS_PER_SOURCE = 50

COLUMNS = [
    "sample_id",
    "source",
    "year",
    "chunk_id",
    "source_doc_id",
    "assigned_label",
    "topic_name",
    "assigned_label_text",
    "best_score",
    "second_best_score",
    "score_gap",
    "n_candidates_above_threshold",
    "text_full",
    "v_gemma",
    "assistant_eval",
    "assistant_notes",
    "user_eval",
    "user_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-input", type=Path, default=DEFAULT_PILOT_INPUT)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_uniform_by_topic(frame: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    if frame.empty or quota <= 0:
        return frame.head(0).copy()
    frame = frame.sample(frac=1.0, random_state=seed).copy()
    groups = {label: group.reset_index(drop=True) for label, group in frame.groupby("assigned_label", dropna=False)}
    selected_parts: list[pd.DataFrame] = []
    pointers = {label: 0 for label in groups}
    labels = sorted(groups)
    selected = 0
    while selected < quota:
        progressed = False
        for label in labels:
            idx = pointers[label]
            group = groups[label]
            if idx >= len(group):
                continue
            selected_parts.append(group.iloc[[idx]])
            pointers[label] += 1
            selected += 1
            progressed = True
            if selected >= quota:
                break
        if not progressed:
            break
    sampled = pd.concat(selected_parts, ignore_index=True) if selected_parts else frame.head(0).copy()
    if len(sampled) < quota:
        remainder = frame.loc[~frame["chunk_id"].astype(str).isin(sampled["chunk_id"].astype(str))].copy()
        if not remainder.empty:
            extra = remainder.sample(n=min(quota - len(sampled), len(remainder)), random_state=seed + 1000)
            sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.head(quota).copy()


def provisional_assistant_eval(row: pd.Series) -> tuple[str, str]:
    source = str(row["source"])
    best_score = float(row["best_score"])
    n_candidates = int(row["n_candidates_above_threshold"]) if not pd.isna(row["n_candidates_above_threshold"]) else 0
    if source == "corporate" and (best_score < 0.70 or n_candidates > 1):
        return "review_carefully", "Corporate Gemma-yes with lower score or multiple cosine candidates; worth checking first."
    if n_candidates > 1:
        return "review_carefully", "Gemma-yes but upstream cosine had multiple candidates above threshold."
    if best_score >= 0.75:
        return "plausible_yes", "Gemma-yes and cosine score is relatively strong in the 6-topic setup."
    return "plausible_yes", "Gemma-yes under the 6-topic setup, but still worth validating against topic specificity."


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pilot_input = pd.read_parquet(args.pilot_input)
    pilot_output = pd.read_csv(args.pilot_output)
    merged = pilot_input.merge(pilot_output[["chunk_id", "v_gemma"]], on="chunk_id", how="inner", validate="one_to_one")
    accepted = merged.loc[merged["v_gemma"] == "yes"].copy()

    samples: list[pd.DataFrame] = []
    for idx, source in enumerate(SOURCE_ORDER):
        source_frame = accepted.loc[accepted["source"] == source].copy()
        source_frame = source_frame.sort_values(["n_candidates_above_threshold", "best_score"], ascending=[False, True])
        samples.append(sample_uniform_by_topic(source_frame, ROWS_PER_SOURCE, args.seed + idx))
    sample = pd.concat(samples, ignore_index=True)
    sample["sample_id"] = [f"six_topic_pilot_{i + 1:03d}" for i in range(len(sample))]
    sample = sample.rename(columns={"text": "text_full"})
    opinions = sample.apply(provisional_assistant_eval, axis=1, result_type="expand")
    sample["assistant_eval"] = opinions[0]
    sample["assistant_notes"] = opinions[1]
    sample["user_eval"] = pd.NA
    sample["user_notes"] = pd.NA
    sample = sample[COLUMNS].sort_values(["source", "assigned_label", "sample_id"]).reset_index(drop=True)

    csv_path = args.output_root / "six_topic_post_gemma_manual_review_150.csv"
    xlsx_path = args.output_root / "six_topic_post_gemma_manual_review_150.xlsx"
    summary_path = args.output_root / "six_topic_post_gemma_manual_review_150_summary.csv"
    manifest_path = args.output_root / "six_topic_post_gemma_manual_review_150_manifest.json"

    sample.to_csv(csv_path, index=False)
    summary = (
        sample.groupby(["source", "assigned_label", "topic_name", "assistant_eval"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source", "assigned_label", "assistant_eval"])
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"sheet_name": "manual_review", "description": "150 post-Gemma yes rows sampled from the 6-topic pilot."},
                {"sheet_name": "summary", "description": "Counts by source, topic, and provisional assistant screen."},
            ]
        ).to_excel(writer, sheet_name="readme", index=False)
        sample.to_excel(writer, sheet_name="manual_review", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps({"rows": int(len(sample)), "source_rows": {source: int((sample["source"] == source).sum()) for source in SOURCE_ORDER}}, indent=2),
        encoding="utf-8",
    )
    print(csv_path)


if __name__ == "__main__":
    main()
