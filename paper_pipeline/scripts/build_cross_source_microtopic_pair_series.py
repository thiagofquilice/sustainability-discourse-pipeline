#!/usr/bin/env python3
"""Build annual time series for retained cross-source microtopic pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import OUTPUT_ROOT, configure_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--pair-input-csv", type=Path, default=None)
    parser.add_argument("--annual-counts-csv", type=Path, default=None)
    parser.add_argument("--denominators-csv", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    if denominator is None or pd.isna(denominator):
        return np.nan
    denominator = float(denominator)
    if denominator == 0:
        return np.nan
    numerator = float(numerator or 0.0)
    return numerator / denominator


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    pair_input_csv = args.pair_input_csv or (args.output_root / "semantic_pairs_with_temporal_diagnostics.csv")
    annual_counts_csv = args.annual_counts_csv or (args.output_root / "microtopic_annual_counts.csv")
    denominators_csv = args.denominators_csv or (args.output_root / "source_topic_year_denominators.csv")

    pairs = safe_read_csv(pair_input_csv)
    annual_counts = safe_read_csv(annual_counts_csv)
    denominators = safe_read_csv(denominators_csv)
    if pairs.empty:
        combined_path = args.output_root / "pair_annual_series_combined.csv"
        docs_path = args.output_root / "pair_annual_series_docs.csv"
        chunks_path = args.output_root / "pair_annual_series_chunks.csv"
        pd.DataFrame().to_csv(combined_path, index=False)
        pd.DataFrame().to_csv(docs_path, index=False)
        pd.DataFrame().to_csv(chunks_path, index=False)
        write_json(
            args.output_root / "series_manifest.json",
            {
                "pair_input_csv": str(pair_input_csv),
                "annual_counts_csv": str(annual_counts_csv),
                "denominators_csv": str(denominators_csv),
                "pair_count": 0,
                "series_row_count": 0,
                "outputs": {
                    "pair_annual_series_combined": str(combined_path),
                    "pair_annual_series_docs": str(docs_path),
                    "pair_annual_series_chunks": str(chunks_path),
                },
            },
        )
        print(args.output_root)
        return

    annual_counts["micro_topic_id"] = pd.to_numeric(annual_counts["micro_topic_id"], errors="coerce").astype(int)
    annual_counts["year"] = pd.to_numeric(annual_counts["year"], errors="coerce").astype(int)
    denominators["year"] = pd.to_numeric(denominators["year"], errors="coerce").astype(int)

    numerator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.micro_topic_id), int(row.year)): (
            int(row.microtopic_doc_n),
            int(row.microtopic_chunk_n),
        )
        for row in annual_counts.itertuples(index=False)
    }
    denominator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.year)): (
            int(row.yes_doc_n),
            int(row.yes_chunk_n),
        )
        for row in denominators.itertuples(index=False)
    }
    years_by_source_topic: dict[tuple[str, str], list[int]] = (
        denominators.groupby(["source", "assigned_label"], dropna=False)["year"]
        .apply(lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()))
        .to_dict()
    )

    rows: list[dict] = []
    for pair in pairs.itertuples(index=False):
        years_a = years_by_source_topic.get((str(pair.source_a), str(pair.macro_topic)), [])
        years_b = years_by_source_topic.get((str(pair.source_b), str(pair.macro_topic)), [])
        series_years = sorted(set(years_a).union(years_b))
        for year in series_years:
            denom_a = denominator_lookup.get((str(pair.source_a), str(pair.macro_topic), int(year)))
            denom_b = denominator_lookup.get((str(pair.source_b), str(pair.macro_topic), int(year)))
            num_a = numerator_lookup.get(
                (str(pair.source_a), str(pair.macro_topic), int(pair.microtopic_id_a), int(year)),
                (0, 0),
            )
            num_b = numerator_lookup.get(
                (str(pair.source_b), str(pair.macro_topic), int(pair.microtopic_id_b), int(year)),
                (0, 0),
            )
            row = {
                "pair_id": str(pair.pair_id),
                "macro_topic": str(pair.macro_topic),
                "macro_topic_name": str(pair.macro_topic_name),
                "dyad": str(pair.dyad),
                "dyad_label": str(pair.dyad_label),
                "source_a": str(pair.source_a),
                "microtopic_id_a": int(pair.microtopic_id_a),
                "source_b": str(pair.source_b),
                "microtopic_id_b": int(pair.microtopic_id_b),
                "year": int(year),
                "source_a_doc_numerator": int(num_a[0]) if denom_a is not None else np.nan,
                "source_b_doc_numerator": int(num_b[0]) if denom_b is not None else np.nan,
                "source_a_chunk_numerator": int(num_a[1]) if denom_a is not None else np.nan,
                "source_b_chunk_numerator": int(num_b[1]) if denom_b is not None else np.nan,
                "source_a_doc_denominator": int(denom_a[0]) if denom_a is not None else np.nan,
                "source_b_doc_denominator": int(denom_b[0]) if denom_b is not None else np.nan,
                "source_a_chunk_denominator": int(denom_a[1]) if denom_a is not None else np.nan,
                "source_b_chunk_denominator": int(denom_b[1]) if denom_b is not None else np.nan,
            }
            row["source_a_share_docs"] = safe_ratio(
                row["source_a_doc_numerator"], row["source_a_doc_denominator"]
            )
            row["source_b_share_docs"] = safe_ratio(
                row["source_b_doc_numerator"], row["source_b_doc_denominator"]
            )
            row["source_a_share_chunks"] = safe_ratio(
                row["source_a_chunk_numerator"], row["source_a_chunk_denominator"]
            )
            row["source_b_share_chunks"] = safe_ratio(
                row["source_b_chunk_numerator"], row["source_b_chunk_denominator"]
            )
            rows.append(row)

    combined = pd.DataFrame(rows).sort_values(["pair_id", "year"]).reset_index(drop=True)
    docs_only = combined[
        [
            "pair_id",
            "macro_topic",
            "macro_topic_name",
            "dyad",
            "dyad_label",
            "source_a",
            "microtopic_id_a",
            "source_b",
            "microtopic_id_b",
            "year",
            "source_a_doc_numerator",
            "source_b_doc_numerator",
            "source_a_doc_denominator",
            "source_b_doc_denominator",
            "source_a_share_docs",
            "source_b_share_docs",
        ]
    ].copy()
    chunks_only = combined[
        [
            "pair_id",
            "macro_topic",
            "macro_topic_name",
            "dyad",
            "dyad_label",
            "source_a",
            "microtopic_id_a",
            "source_b",
            "microtopic_id_b",
            "year",
            "source_a_chunk_numerator",
            "source_b_chunk_numerator",
            "source_a_chunk_denominator",
            "source_b_chunk_denominator",
            "source_a_share_chunks",
            "source_b_share_chunks",
        ]
    ].copy()

    combined_path = args.output_root / "pair_annual_series_combined.csv"
    docs_path = args.output_root / "pair_annual_series_docs.csv"
    chunks_path = args.output_root / "pair_annual_series_chunks.csv"
    combined.to_csv(combined_path, index=False)
    docs_only.to_csv(docs_path, index=False)
    chunks_only.to_csv(chunks_path, index=False)

    write_json(
        args.output_root / "series_manifest.json",
        {
            "pair_input_csv": str(pair_input_csv),
            "annual_counts_csv": str(annual_counts_csv),
            "denominators_csv": str(denominators_csv),
            "pair_count": int(pairs.shape[0]),
            "series_row_count": int(combined.shape[0]),
            "outputs": {
                "pair_annual_series_combined": str(combined_path),
                "pair_annual_series_docs": str(docs_path),
                "pair_annual_series_chunks": str(chunks_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
