#!/usr/bin/env python3
"""Build descriptive and validation-ready outputs for matched microtopic pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import OUTPUT_ROOT, configure_logging, load_stage1_enrichment, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--profiles-csv", type=Path, default=None)
    parser.add_argument("--series-csv", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def active_year_stats(series: pd.DataFrame, numerator_col: str, share_col: str) -> dict[str, float | int | None]:
    active = series.loc[series[numerator_col].fillna(0) > 0].copy()
    if active.empty:
        return {
            "first_year": None,
            "last_year": None,
            "years_active": 0,
            "peak_year": None,
            "peak_share": np.nan,
            "mean_share": np.nan,
            "total_volume": 0,
        }
    peak_idx = active[share_col].astype(float).fillna(-np.inf).idxmax()
    peak_year = int(active.loc[peak_idx, "year"]) if pd.notna(peak_idx) else None
    peak_share = float(active.loc[peak_idx, share_col]) if pd.notna(peak_idx) else np.nan
    return {
        "first_year": int(active["year"].min()),
        "last_year": int(active["year"].max()),
        "years_active": int(active["year"].nunique()),
        "peak_year": peak_year,
        "peak_share": peak_share,
        "mean_share": float(active[share_col].astype(float).mean()),
        "total_volume": int(active[numerator_col].fillna(0).sum()),
    }


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    diagnostics_csv = args.diagnostics_csv or (args.output_root / "semantic_pairs_with_temporal_diagnostics.csv")
    profiles_csv = args.profiles_csv or (args.output_root / "microtopic_profiles.csv")
    series_csv = args.series_csv or (args.output_root / "pair_annual_series_combined.csv")

    diagnostics = safe_read_csv(diagnostics_csv)
    profiles = safe_read_csv(profiles_csv)
    series = safe_read_csv(series_csv)
    if diagnostics.empty:
        pair_summary_path = args.output_root / "pair_descriptive_summary.csv"
        validation_path = args.output_root / "validation_ready_pair_table.csv"
        summary_by_dyad_path = args.output_root / "summary_by_dyad.csv"
        summary_by_macro_path = args.output_root / "summary_by_macro_topic.csv"
        summary_by_source_path = args.output_root / "summary_by_source.csv"
        for path in [
            pair_summary_path,
            validation_path,
            summary_by_dyad_path,
            summary_by_macro_path,
            summary_by_source_path,
        ]:
            pd.DataFrame().to_csv(path, index=False)
        write_json(
            args.output_root / "descriptive_manifest.json",
            {
                "diagnostics_csv": str(diagnostics_csv),
                "profiles_csv": str(profiles_csv),
                "series_csv": str(series_csv),
                "pair_descriptive_summary_count": 0,
                "validation_ready_row_count": 0,
                "stage1_enrichment_available": False,
                "outputs": {
                    "pair_descriptive_summary": str(pair_summary_path),
                    "validation_ready_pair_table": str(validation_path),
                    "summary_by_dyad": str(summary_by_dyad_path),
                    "summary_by_macro_topic": str(summary_by_macro_path),
                    "summary_by_source": str(summary_by_source_path),
                },
            },
        )
        print(args.output_root)
        return

    profiles["microtopic_id"] = pd.to_numeric(profiles["microtopic_id"], errors="coerce").astype(int)
    profile_map = {
        (str(row.subgroup), int(row.microtopic_id)): row._asdict()
        for row in profiles.itertuples(index=False)
    }

    stage1 = load_stage1_enrichment()
    stage1_map = {}
    if not stage1.empty:
        stage1_map = {
            (str(row.subgroup), int(row.micro_topic_id)): row._asdict()
            for row in stage1.itertuples(index=False)
        }

    descriptive_rows: list[dict] = []
    validation_rows: list[dict] = []

    for pair in diagnostics.itertuples(index=False):
        pair_series = series.loc[series["pair_id"] == pair.pair_id].copy().sort_values("year")
        profile_a = profile_map[(str(pair.subgroup_a), int(pair.microtopic_id_a))]
        profile_b = profile_map[(str(pair.subgroup_b), int(pair.microtopic_id_b))]
        stage1_a = stage1_map.get((str(pair.subgroup_a), int(pair.microtopic_id_a)), {})
        stage1_b = stage1_map.get((str(pair.subgroup_b), int(pair.microtopic_id_b)), {})

        a_doc = active_year_stats(pair_series, "source_a_doc_numerator", "source_a_share_docs")
        b_doc = active_year_stats(pair_series, "source_b_doc_numerator", "source_b_share_docs")
        a_chunk = active_year_stats(pair_series, "source_a_chunk_numerator", "source_a_share_chunks")
        b_chunk = active_year_stats(pair_series, "source_b_chunk_numerator", "source_b_share_chunks")

        overlap = pair_series.dropna(subset=["source_a_share_docs", "source_b_share_docs"]).copy()
        descriptive_record = {
            **pair._asdict(),
            "source_a_first_year_docs": a_doc["first_year"],
            "source_b_first_year_docs": b_doc["first_year"],
            "source_a_peak_year_docs": a_doc["peak_year"],
            "source_b_peak_year_docs": b_doc["peak_year"],
            "source_a_last_year_docs": a_doc["last_year"],
            "source_b_last_year_docs": b_doc["last_year"],
            "source_a_years_active_docs": a_doc["years_active"],
            "source_b_years_active_docs": b_doc["years_active"],
            "source_a_total_doc_volume": a_doc["total_volume"],
            "source_b_total_doc_volume": b_doc["total_volume"],
            "source_a_mean_share_docs": a_doc["mean_share"],
            "source_b_mean_share_docs": b_doc["mean_share"],
            "source_a_peak_share_docs": a_doc["peak_share"],
            "source_b_peak_share_docs": b_doc["peak_share"],
            "source_a_peak_year_chunks": a_chunk["peak_year"],
            "source_b_peak_year_chunks": b_chunk["peak_year"],
            "source_a_peak_share_chunks": a_chunk["peak_share"],
            "source_b_peak_share_chunks": b_chunk["peak_share"],
            "source_a_total_chunk_volume": a_chunk["total_volume"],
            "source_b_total_chunk_volume": b_chunk["total_volume"],
            "source_a_mean_share_chunks": a_chunk["mean_share"],
            "source_b_mean_share_chunks": b_chunk["mean_share"],
            "mean_abs_doc_share_gap": float(
                (overlap["source_a_share_docs"] - overlap["source_b_share_docs"]).abs().mean()
            )
            if not overlap.empty
            else np.nan,
            "mean_abs_chunk_share_gap": float(
                (pair_series["source_a_share_chunks"] - pair_series["source_b_share_chunks"]).abs().mean()
            )
            if pair_series["source_a_share_chunks"].notna().any() and pair_series["source_b_share_chunks"].notna().any()
            else np.nan,
            "doc_volume_ratio_a_to_b": float(a_doc["total_volume"] / b_doc["total_volume"])
            if b_doc["total_volume"]
            else np.nan,
            "chunk_volume_ratio_a_to_b": float(a_chunk["total_volume"] / b_chunk["total_volume"])
            if b_chunk["total_volume"]
            else np.nan,
        }
        descriptive_rows.append(descriptive_record)

        validation_rows.append(
            {
                **descriptive_record,
                "profile_text_a": profile_a["profile_text"],
                "profile_text_b": profile_b["profile_text"],
                "top_terms_a": profile_a["top_terms"],
                "top_terms_b": profile_b["top_terms"],
                "representative_chunks_a": profile_a["representative_chunks"],
                "representative_chunks_b": profile_b["representative_chunks"],
                "stage1_enrichment_available_a": bool(stage1_a.get("stage1_summary_available", False)),
                "stage1_enrichment_available_b": bool(stage1_b.get("stage1_summary_available", False)),
                "stage1_summary_year_count_a": stage1_a.get("stage1_summary_year_count"),
                "stage1_summary_year_count_b": stage1_b.get("stage1_summary_year_count"),
                "stage1_focus_primary_examples_json_a": stage1_a.get("stage1_focus_primary_examples_json", ""),
                "stage1_focus_primary_examples_json_b": stage1_b.get("stage1_focus_primary_examples_json", ""),
                "stage1_frame_examples_json_a": stage1_a.get("stage1_frame_examples_json", ""),
                "stage1_frame_examples_json_b": stage1_b.get("stage1_frame_examples_json", ""),
            }
        )

    descriptive = pd.DataFrame(descriptive_rows)
    validation = pd.DataFrame(validation_rows)

    pair_summary_path = args.output_root / "pair_descriptive_summary.csv"
    validation_path = args.output_root / "validation_ready_pair_table.csv"
    descriptive.to_csv(pair_summary_path, index=False)
    validation.to_csv(validation_path, index=False)

    summary_by_dyad = (
        descriptive.groupby(["dyad", "dyad_label"], dropna=False)
        .agg(
            semantic_pair_count=("pair_id", "nunique"),
            viable_lenient_count=("lenient_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            viable_balanced_count=("balanced_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            viable_strict_count=("strict_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_cosine_similarity=("cosine_similarity", "mean"),
        )
        .reset_index()
    )
    summary_by_macro = (
        descriptive.groupby(["macro_topic", "macro_topic_name"], dropna=False)
        .agg(
            semantic_pair_count=("pair_id", "nunique"),
            viable_lenient_count=("lenient_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            viable_balanced_count=("balanced_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            viable_strict_count=("strict_is_viable", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_cosine_similarity=("cosine_similarity", "mean"),
        )
        .reset_index()
    )

    source_rows: list[dict] = []
    for source_col, id_col, viable_col in [
        ("source_a", "microtopic_id_a", "balanced_is_viable"),
        ("source_b", "microtopic_id_b", "balanced_is_viable"),
    ]:
        group = (
            descriptive.groupby(source_col, dropna=False)
            .agg(
                pair_participation_count=("pair_id", "nunique"),
                balanced_viable_pair_count=(viable_col, lambda s: int(pd.Series(s).fillna(False).sum())),
                mean_cosine_similarity=("cosine_similarity", "mean"),
            )
            .reset_index()
            .rename(columns={source_col: "source"})
        )
        uniq = (
            descriptive[[source_col, id_col]]
            .drop_duplicates()
            .groupby(source_col, dropna=False)
            .size()
            .reset_index(name="unique_microtopics_in_pairs")
            .rename(columns={source_col: "source"})
        )
        group = group.merge(uniq, on="source", how="left")
        source_rows.append(group)
    summary_by_source = (
        pd.concat(source_rows, ignore_index=True)
        .groupby("source", dropna=False)
        .agg(
            pair_participation_count=("pair_participation_count", "sum"),
            balanced_viable_pair_count=("balanced_viable_pair_count", "sum"),
            unique_microtopics_in_pairs=("unique_microtopics_in_pairs", "sum"),
            mean_cosine_similarity=("mean_cosine_similarity", "mean"),
        )
        .reset_index()
    )

    summary_by_dyad_path = args.output_root / "summary_by_dyad.csv"
    summary_by_macro_path = args.output_root / "summary_by_macro_topic.csv"
    summary_by_source_path = args.output_root / "summary_by_source.csv"
    summary_by_dyad.to_csv(summary_by_dyad_path, index=False)
    summary_by_macro.to_csv(summary_by_macro_path, index=False)
    summary_by_source.to_csv(summary_by_source_path, index=False)

    write_json(
        args.output_root / "descriptive_manifest.json",
        {
            "diagnostics_csv": str(diagnostics_csv),
            "profiles_csv": str(profiles_csv),
            "series_csv": str(series_csv),
            "pair_descriptive_summary_count": int(descriptive.shape[0]),
            "validation_ready_row_count": int(validation.shape[0]),
            "stage1_enrichment_available": bool(stage1_map),
            "outputs": {
                "pair_descriptive_summary": str(pair_summary_path),
                "validation_ready_pair_table": str(validation_path),
                "summary_by_dyad": str(summary_by_dyad_path),
                "summary_by_macro_topic": str(summary_by_macro_path),
                "summary_by_source": str(summary_by_source_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
