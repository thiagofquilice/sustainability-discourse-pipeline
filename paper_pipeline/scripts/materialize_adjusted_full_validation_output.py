#!/usr/bin/env python3
"""Materialize the canonical adjusted full validation output for the 6-topic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path("paper_pipeline")
DEFAULT_FULL_OUTPUT = WORKFLOW_ROOT / "outputs" / "full_run" / "best_only" / "validation_output.csv"
DEFAULT_BEST_ONLY_POSITIVE = WORKFLOW_ROOT / "outputs" / "cosine" / "best_only_positive.parquet"
DEFAULT_REMAP_FILE = WORKFLOW_ROOT / "outputs" / "t2_secondary_recovery_cosine_round" / "old_t2_under_secondary_recovery_variant.parquet"
DEFAULT_CHANGED_GEMMA = WORKFLOW_ROOT / "outputs" / "t2_secondary_recovery_cosine_round" / "gemma_local_full" / "validation_output.csv"
DEFAULT_CATALOG = WORKFLOW_ROOT / "catalog" / "six_topic_discourse_catalog.csv"
DEFAULT_OUTPUT_DIR = WORKFLOW_ROOT / "outputs" / "full_run" / "adjusted_with_t2_secondary_recovery"


REFRESH_FROM_CATALOG_COLUMNS = [
    "assigned_label_id",
    "assigned_label_text",
    "topic_name",
    "topic_definition",
    "topic_summary_text",
    "sdg_crosswalk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_OUTPUT)
    parser.add_argument("--best-only-positive", type=Path, default=DEFAULT_BEST_ONLY_POSITIVE)
    parser.add_argument("--remap-file", type=Path, default=DEFAULT_REMAP_FILE)
    parser.add_argument("--changed-gemma", type=Path, default=DEFAULT_CHANGED_GEMMA)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    catalog = catalog.rename(
        columns={
            "topic_code": "assigned_label",
            "topic_id": "assigned_label_id",
            "topic_elements_bullets": "assigned_label_text",
        }
    )
    return catalog[
        [
            "assigned_label",
            "assigned_label_id",
            "assigned_label_text",
            "topic_name",
            "topic_definition",
            "topic_summary_text",
            "sdg_crosswalk",
        ]
    ].drop_duplicates(subset=["assigned_label"])


def enrich_from_best_only(frame: pd.DataFrame, best_only_positive: pd.DataFrame) -> pd.DataFrame:
    extra_cols = [c for c in best_only_positive.columns if c not in frame.columns]
    merge_cols = ["chunk_id", *extra_cols]
    enriched = frame.merge(best_only_positive[merge_cols], on="chunk_id", how="left")
    return enriched


def refresh_topic_columns(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in frame.columns if c not in REFRESH_FROM_CATALOG_COLUMNS]
    refreshed = frame[base_cols].merge(catalog, on="assigned_label", how="left")
    refreshed["assigned_label_id"] = pd.to_numeric(refreshed["assigned_label_id"], errors="coerce").astype("Int64")
    return refreshed


def update_scores_for_remain_t2(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "new_best_score": "best_score",
        "new_second_best_score": "second_best_score",
        "new_score_gap": "score_gap",
        "new_n_candidates_above_threshold": "n_candidates_above_threshold",
    }
    for src, dest in rename_map.items():
        if src in frame.columns:
            frame[dest] = frame[src]
    return frame


def coalesce_t2_embedding_index(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = [c for c in ["embedding_row_index", "embedding_row_index_x", "embedding_row_index_y"] if c in frame.columns]
    if not candidates:
        return frame
    series = None
    for col in candidates:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if series is None:
            series = numeric
        else:
            series = series.fillna(numeric)
    frame["embedding_row_index"] = series
    return frame


def materialize_adjusted(full_enriched: pd.DataFrame, remap: pd.DataFrame, changed_gemma: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    remap_lookup = remap[
        [
            "chunk_id",
            "transition",
            "embedding_row_index",
            "new_best_score",
            "new_second_best_score",
            "new_score_gap",
            "new_n_candidates_above_threshold",
        ]
    ].drop_duplicates(subset=["chunk_id"])

    full_t2 = full_enriched.loc[full_enriched["assigned_label"] == "T2"].merge(remap_lookup, on="chunk_id", how="left")
    full_non_t2 = full_enriched.loc[full_enriched["assigned_label"] != "T2"].copy()

    remain_t2 = full_t2.loc[full_t2["transition"] == "T2_to_T2"].copy()
    remain_t2 = coalesce_t2_embedding_index(remain_t2)
    remain_t2 = update_scores_for_remain_t2(remain_t2)
    to_outside = full_t2.loc[full_t2["transition"] == "T2_to_outside"].copy()
    changed_old = full_t2.loc[full_t2["transition"] == "T2_to_other"].copy()

    adjusted = pd.concat(
        [
            full_non_t2,
            remain_t2[[c for c in remain_t2.columns if c in full_enriched.columns]],
            changed_gemma[[c for c in changed_gemma.columns if c in full_enriched.columns]],
        ],
        ignore_index=True,
    )

    manifest = {
        "full_rows_original": int(len(full_enriched)),
        "adjusted_rows": int(len(adjusted)),
        "t2_rows_original": int(len(full_t2)),
        "t2_remain_t2": int(len(remain_t2)),
        "t2_to_other": int(len(changed_old)),
        "t2_to_outside": int(len(to_outside)),
        "changed_gemma_rows_used": int(len(changed_gemma)),
    }
    return adjusted, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.full_output)
    best_only_positive = pd.read_parquet(args.best_only_positive)
    remap = pd.read_parquet(args.remap_file)
    changed_gemma = pd.read_csv(args.changed_gemma)
    catalog = load_catalog(args.catalog)

    full_enriched = enrich_from_best_only(full, best_only_positive)
    changed_gemma_enriched = enrich_from_best_only(changed_gemma, best_only_positive)
    changed_chunk_ids = set(changed_gemma_enriched["chunk_id"].astype(str))
    remap = remap.loc[remap["chunk_id"].astype(str).isin(changed_chunk_ids | set(full_enriched["chunk_id"].astype(str)))].copy()

    adjusted, manifest = materialize_adjusted(full_enriched, remap, changed_gemma_enriched)
    adjusted = refresh_topic_columns(adjusted, catalog)

    yes_frame = adjusted.loc[adjusted["v_gemma"].astype(str).str.lower() == "yes"].copy().reset_index(drop=True)

    adjusted_csv = args.output_dir / "adjusted_full_validation_output.csv"
    adjusted_parquet = args.output_dir / "adjusted_full_validation_output.parquet"
    yes_csv = args.output_dir / "adjusted_full_yes.csv"
    yes_parquet = args.output_dir / "adjusted_full_yes.parquet"

    adjusted.to_csv(adjusted_csv, index=False)
    yes_frame.to_csv(yes_csv, index=False)

    parquet_status = {"adjusted_full_validation_output_parquet": "not_written", "adjusted_full_yes_parquet": "not_written"}
    try:
        adjusted.to_parquet(adjusted_parquet, index=False)
        parquet_status["adjusted_full_validation_output_parquet"] = "written"
    except Exception as exc:  # noqa: BLE001
        parquet_status["adjusted_full_validation_output_parquet"] = f"failed: {exc}"
    try:
        yes_frame.to_parquet(yes_parquet, index=False)
        parquet_status["adjusted_full_yes_parquet"] = "written"
    except Exception as exc:  # noqa: BLE001
        parquet_status["adjusted_full_yes_parquet"] = f"failed: {exc}"

    by_source = (
        yes_frame.groupby("source", dropna=False)
        .agg(chunk_rows=("chunk_id", "size"), unique_docs=("source_doc_id", "nunique"), years=("year", "nunique"))
        .reset_index()
    )
    by_source_topic = (
        yes_frame.groupby(["source", "assigned_label"], dropna=False)
        .agg(chunk_rows=("chunk_id", "size"), unique_docs=("source_doc_id", "nunique"), years=("year", "nunique"))
        .reset_index()
    )
    by_source.to_csv(args.output_dir / "adjusted_full_yes_by_source.csv", index=False)
    by_source_topic.to_csv(args.output_dir / "adjusted_full_yes_by_source_topic.csv", index=False)

    manifest.update(
        {
            "adjusted_yes_rows": int(len(yes_frame)),
            "adjusted_yes_unique_docs": int(yes_frame["source_doc_id"].nunique()),
            "outputs": {
                "adjusted_full_validation_output_csv": str(adjusted_csv),
                "adjusted_full_validation_output_parquet": str(adjusted_parquet),
                "adjusted_full_yes_csv": str(yes_csv),
                "adjusted_full_yes_parquet": str(yes_parquet),
            },
            "parquet_status": parquet_status,
        }
    )
    (args.output_dir / "adjusted_full_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(args.output_dir)


if __name__ == "__main__":
    main()
