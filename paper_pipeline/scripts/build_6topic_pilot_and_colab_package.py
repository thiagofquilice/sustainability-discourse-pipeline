#!/usr/bin/env python3
"""Build the 6-topic pilot package and pre-Gemma manual-review workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ORDER = ["academic", "corporate", "media"]
PILOT_SOURCE_QUOTAS = {"academic": 334, "corporate": 333, "media": 333}
PRE_REVIEW_PER_SOURCE = 50
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = WORKFLOW_ROOT / "outputs" / "cosine"
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def proportional_topic_quota(frame: pd.DataFrame, quota: int) -> dict[str, int]:
    counts = frame["assigned_label"].value_counts().sort_index()
    raw = (counts / counts.sum()) * quota
    base = pd.Series(np.floor(raw), index=raw.index).astype(int)
    remainder = quota - int(base.sum())
    if remainder > 0:
        order = (raw - base).sort_values(ascending=False).index.tolist()
        for topic in order[:remainder]:
            base.loc[topic] += 1
    return base.to_dict()


def sample_source_proportional(frame: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.head(0).copy()
    quotas = proportional_topic_quota(frame, quota)
    pieces: list[pd.DataFrame] = []
    used_ids: set[str] = set()
    for topic, topic_quota in quotas.items():
        subset = frame.loc[(frame["assigned_label"] == topic) & (~frame["chunk_id"].astype(str).isin(used_ids))].copy()
        if subset.empty or topic_quota <= 0:
            continue
        chosen = subset.sample(n=min(topic_quota, len(subset)), random_state=seed)
        pieces.append(chosen)
        used_ids.update(chosen["chunk_id"].astype(str).tolist())
    sampled = pd.concat(pieces, ignore_index=True) if pieces else frame.head(0).copy()
    if len(sampled) < quota:
        remainder = frame.loc[~frame["chunk_id"].astype(str).isin(sampled["chunk_id"].astype(str))].copy()
        if not remainder.empty:
            extra = remainder.sample(n=min(quota - len(sampled), len(remainder)), random_state=seed + 1000)
            sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.head(quota).copy()


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


def add_score_band(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["score_band"] = pd.cut(
        result["best_score"],
        bins=[-np.inf, 0.68, 0.74, np.inf],
        labels=["near_threshold", "mid_score", "high_score"],
    ).astype(str)
    return result


def provisional_assistant_eval(row: pd.Series) -> tuple[str, str]:
    source = str(row["source"])
    best_score = float(row["best_score"])
    n_candidates = int(row["n_candidates_above_threshold"]) if not pd.isna(row["n_candidates_above_threshold"]) else 0
    if source == "corporate" and (best_score < 0.70 or n_candidates > 1):
        return "review_carefully", "Corporate candidate with lower score or multiple candidates above threshold."
    if n_candidates > 1:
        return "review_carefully", "Multiple cosine candidates above threshold for this document."
    if best_score >= 0.75:
        return "plausible_yes", "Strong cosine score for the 6-topic pilot sample."
    return "plausible_yes", "Accepted for the pilot, but still close enough to the threshold to merit checking."


def build_manual_pre_gemma(best_only: pd.DataFrame, output_root: Path, seed: int) -> None:
    manual_root = output_root / "manual_review" / "pilot_1000_pre_gemma"
    manual_root.mkdir(parents=True, exist_ok=True)
    samples: list[pd.DataFrame] = []
    for idx, source in enumerate(SOURCE_ORDER):
        source_frame = best_only.loc[best_only["source"] == source].copy()
        samples.append(sample_uniform_by_topic(source_frame, PRE_REVIEW_PER_SOURCE, seed + idx + 500))
    sample = pd.concat(samples, ignore_index=True)
    sample["sample_id"] = [f"six_topic_pre_{i + 1:03d}" for i in range(len(sample))]
    opinions = sample.apply(provisional_assistant_eval, axis=1, result_type="expand")
    sample["assistant_eval_pre_gemma"] = opinions[0]
    sample["assistant_notes_pre_gemma"] = opinions[1]
    sample["user_eval"] = pd.NA
    sample["user_notes"] = pd.NA
    sample = sample.rename(columns={"text": "text_full"})[
        [
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
            "assistant_eval_pre_gemma",
            "assistant_notes_pre_gemma",
            "user_eval",
            "user_notes",
        ]
    ].sort_values(["source", "assigned_label", "sample_id"])
    sample.to_csv(manual_root / "six_topic_pilot_1000_pre_gemma_manual_review.csv", index=False)
    with pd.ExcelWriter(manual_root / "six_topic_pilot_1000_pre_gemma_manual_review.xlsx", engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="manual_review", index=False)


def build_colab_package(run_root: Path, output_root: Path, seed: int) -> None:
    best_only_path = run_root / "best_only_positive.parquet"
    candidates_path = run_root / "multi_candidate_positive.parquet"

    pilot_input_dir = output_root / "pilot_1000" / "input"
    full_input_dir = output_root / "full_run" / "input"
    extra_input_dir = output_root / "full_run" / "extra_candidates_input"
    for path in [pilot_input_dir, full_input_dir, extra_input_dir]:
        path.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "chunk_id",
        "text",
        "assigned_label",
        "assigned_label_id",
        "assigned_label_text",
        "topic_name",
        "topic_definition",
        "topic_summary_text",
        "sdg_crosswalk",
        "source",
        "year",
        "source_doc_id",
        "best_score",
        "second_best_score",
        "score_gap",
        "score_margin",
        "n_candidates_above_threshold",
    ]
    best_only = add_score_band(pd.read_parquet(best_only_path, columns=keep_cols)).copy()
    best_only["sample_order"] = np.arange(len(best_only))

    pilot_parts: list[pd.DataFrame] = []
    for idx, (source, quota) in enumerate(PILOT_SOURCE_QUOTAS.items()):
        source_frame = best_only.loc[best_only["source"] == source].copy()
        pilot_parts.append(sample_source_proportional(source_frame, quota, seed + idx))
    pilot = pd.concat(pilot_parts, ignore_index=True)
    pilot = pilot.sort_values(["source", "assigned_label", "sample_order"]).reset_index(drop=True)

    remaining = best_only.loc[~best_only["chunk_id"].astype(str).isin(pilot["chunk_id"].astype(str))].copy()
    remaining = remaining.sort_values(["source", "year", "assigned_label", "chunk_id"]).reset_index(drop=True)

    pilot_csv = pilot_input_dir / "six_topic_pilot_1000.csv"
    pilot_parquet = pilot_input_dir / "six_topic_pilot_1000.parquet"
    pilot.to_csv(pilot_csv, index=False)
    pilot.to_parquet(pilot_parquet, index=False)
    (
        pilot.groupby(["source", "assigned_label", "topic_name", "score_band"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source", "assigned_label", "score_band"])
        .to_csv(pilot_input_dir / "six_topic_pilot_1000_summary.csv", index=False)
    )
    (pilot_input_dir / "six_topic_pilot_1000_manifest.json").write_text(
        json.dumps({"rows": int(len(pilot)), "source_quotas": PILOT_SOURCE_QUOTAS, "seed": seed}, indent=2),
        encoding="utf-8",
    )

    remaining.to_parquet(full_input_dir / "six_topic_best_only_remaining.parquet", index=False)
    (
        remaining.groupby(["source", "assigned_label", "topic_name"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source", "assigned_label"])
        .to_csv(full_input_dir / "six_topic_best_only_remaining_summary.csv", index=False)
    )
    (full_input_dir / "six_topic_best_only_remaining_manifest.json").write_text(
        json.dumps({"rows": int(len(remaining)), "excluded_pilot_rows": int(len(pilot))}, indent=2),
        encoding="utf-8",
    )

    (extra_input_dir / "six_topic_extra_candidates_manifest.json").write_text(
        json.dumps(
            {
                "status": "prepared_but_not_materialized",
                "source_parquet": str(candidates_path),
                "note": "Extra candidates stay optional for a second-stage Colab run."
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    build_manual_pre_gemma(pilot, output_root, seed)


def main() -> None:
    args = parse_args()
    build_colab_package(args.run_root, args.output_root, args.seed)
    print(args.output_root / "pilot_1000" / "input" / "six_topic_pilot_1000.parquet")


if __name__ == "__main__":
    main()
