#!/usr/bin/env python3
"""Aggregate no-MNN corporate-focus matches and analyze temporal precedence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microtopic_posthoc_merge_common import (
    CORPORATE_FOCUS_AGGREGATED_TEMPORAL_OUTPUT_ROOT,
    CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT,
    CORPORATE_FOCUS_PAIR_OUTPUT_ROOT,
    ensure_directory,
    json_dumps,
    write_json,
)


MIN_OVERLAP = 4
CORRELATION_FLOOR = 0.30
BALANCED_RULE = {
    "min_active_years_per_side": 4,
    "min_overlap_years": 4,
    "min_docs_per_side": 10,
    "min_chunks_per_side": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-evidence-csv",
        type=Path,
        default=CORPORATE_FOCUS_REVIEW_OUTPUT_ROOT / "corporate_focus_pair_evidence.csv",
    )
    parser.add_argument(
        "--annual-counts-csv",
        type=Path,
        default=CORPORATE_FOCUS_PAIR_OUTPUT_ROOT / "microtopic_annual_counts.csv",
    )
    parser.add_argument(
        "--denominators-csv",
        type=Path,
        default=CORPORATE_FOCUS_PAIR_OUTPUT_ROOT / "source_topic_year_denominators.csv",
    )
    parser.add_argument("--output-root", type=Path, default=CORPORATE_FOCUS_AGGREGATED_TEMPORAL_OUTPUT_ROOT)
    parser.add_argument("--similarity-threshold", type=float, default=0.60)
    parser.add_argument("--lag-min", type=int, default=-2)
    parser.add_argument("--lag-max", type=int, default=2)
    return parser.parse_args()


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    if denominator is None or pd.isna(denominator):
        return np.nan
    denominator = float(denominator)
    if denominator == 0:
        return np.nan
    numerator = float(numerator or 0.0)
    return numerator / denominator


def correlation_for_lag(series: pd.DataFrame, col_a: str, col_b: str, lag: int) -> tuple[float | None, int]:
    left = series[["year", col_a]].rename(columns={col_a: "x"})
    right = series[["year", col_b]].rename(columns={col_b: "y"}).copy()
    right["year"] = right["year"] - lag
    aligned = left.merge(right, on="year", how="inner").dropna()
    overlap = len(aligned)
    if overlap < MIN_OVERLAP:
        return None, overlap
    if aligned["x"].nunique() <= 1 or aligned["y"].nunique() <= 1:
        return None, overlap
    return float(np.corrcoef(aligned["x"], aligned["y"])[0, 1]), overlap


def precedence_type_label(source_a: str, source_b: str, classification: str) -> str:
    if classification == "synchronous_or_unclear":
        return classification
    if classification == "source_a_leads":
        return f"{source_a}_leads_{source_b}"
    return f"{source_b}_leads_{source_a}"


def classify(best_lag: float | int | None, best_corr: float | None, low_info: bool, unstable: bool) -> tuple[str, str]:
    if pd.isna(best_lag) or pd.isna(best_corr) or low_info or unstable or abs(float(best_corr)) < CORRELATION_FLOOR:
        return "synchronous_or_unclear", "low_information_or_unclear"
    if int(best_lag) > 0:
        return "source_a_leads", "positive_best_lag_docs"
    if int(best_lag) < 0:
        return "source_b_leads", "negative_best_lag_docs"
    return "synchronous_or_unclear", "zero_best_lag_docs"


def summarize_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            aggregated_pair_count=("aggregate_pair_id", "nunique"),
            corporate_microtopic_count=("matched_corporate_group_id", "nunique"),
            stable_leading_pair_count=("is_stable_leading_pair", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_best_correlation_docs=("best_correlation_docs", "mean"),
            median_best_correlation_docs=("best_correlation_docs", "median"),
        )
        .reset_index()
    )
    return grouped


def build_aggregate_pairs(pair_evidence: pd.DataFrame, threshold: float) -> pd.DataFrame:
    qualifying = pair_evidence.loc[pair_evidence["cosine_similarity"] >= threshold].copy()
    if qualifying.empty:
        return pd.DataFrame()
    group_cols = [
        "dyad",
        "macro_topic",
        "macro_topic_name",
        "source",
        "assigned_label",
        "matched_corporate_subgroup",
        "matched_corporate_micro_topic_id",
        "matched_corporate_topic_name_original",
        "matched_corporate_group_id",
    ]
    rows: list[dict] = []
    for key, group in qualifying.groupby(group_cols, dropna=False):
        (
            dyad,
            macro_topic,
            macro_topic_name,
            source_a,
            assigned_label,
            corporate_subgroup,
            corporate_micro_topic_id,
            corporate_topic_name_original,
            corporate_group_id,
        ) = key
        noncorp_topic_ids = sorted(group["micro_topic_id"].astype(int).unique().tolist())
        rows.append(
            {
                "aggregate_pair_id": f"{dyad}__{macro_topic}__agg__{source_a}__{corporate_group_id}",
                "dyad": str(dyad),
                "macro_topic": str(macro_topic),
                "macro_topic_name": str(macro_topic_name),
                "source_a": str(source_a),
                "subgroup_a": f"{source_a}_{assigned_label}",
                "source_b": "corporate",
                "subgroup_b": str(corporate_subgroup),
                "assigned_label": str(assigned_label),
                "matched_corporate_group_id": str(corporate_group_id),
                "corporate_micro_topic_id": int(corporate_micro_topic_id),
                "corporate_topic_name_original": str(corporate_topic_name_original),
                "constituent_noncorporate_micro_topic_ids_json": json_dumps(noncorp_topic_ids),
                "constituent_noncorporate_group_ids_json": json_dumps(
                    sorted(group["final_merge_group_id"].astype(str).unique().tolist())
                ),
                "constituent_noncorporate_topic_names_json": json_dumps(
                    sorted(group["topic_name_original"].astype(str).unique().tolist())
                ),
                "constituent_noncorporate_count": int(len(noncorp_topic_ids)),
                "direct_pair_count": int(group["pair_id"].nunique()),
                "max_cosine_similarity": float(group["cosine_similarity"].max()),
                "mean_cosine_similarity": float(group["cosine_similarity"].mean()),
                "best_pair_ids_json": json_dumps(
                    group.sort_values("cosine_similarity", ascending=False)["pair_id"].astype(str).head(10).tolist()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_topic", "dyad", "matched_corporate_group_id"]).reset_index(drop=True)


def build_series(aggregate_pairs: pd.DataFrame, annual_counts: pd.DataFrame, denominators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = annual_counts.copy()
    annual["micro_topic_id"] = pd.to_numeric(annual["micro_topic_id"], errors="coerce")
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    annual = annual.dropna(subset=["micro_topic_id", "year"]).copy()
    annual["micro_topic_id"] = annual["micro_topic_id"].astype(int)
    annual["year"] = annual["year"].astype(int)

    denominators = denominators.copy()
    denominators["year"] = pd.to_numeric(denominators["year"], errors="coerce")
    denominators = denominators.dropna(subset=["year"]).copy()
    denominators["year"] = denominators["year"].astype(int)

    numerator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.micro_topic_id), int(row.year)): (
            int(row.microtopic_doc_n),
            int(row.microtopic_chunk_n),
        )
        for row in annual.itertuples(index=False)
    }
    denominator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.year)): (
            int(row.yes_doc_n),
            int(row.yes_chunk_n),
        )
        for row in denominators.itertuples(index=False)
    }
    years_by_source_topic = (
        denominators.groupby(["source", "assigned_label"], dropna=False)["year"]
        .apply(lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()))
        .to_dict()
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for pair in aggregate_pairs.itertuples(index=False):
        noncorp_ids = (
            [int(value) for value in json.loads(pair.constituent_noncorporate_micro_topic_ids_json)]
            if str(pair.constituent_noncorporate_micro_topic_ids_json).strip()
            else []
        )
        years_a = years_by_source_topic.get((str(pair.source_a), str(pair.assigned_label)), [])
        years_b = years_by_source_topic.get(("corporate", str(pair.assigned_label)), [])
        series_years = sorted(set(years_a).union(years_b))
        source_a_active_years: list[int] = []
        source_b_active_years: list[int] = []
        total_docs_a = 0
        total_chunks_a = 0
        total_docs_b = 0
        total_chunks_b = 0

        for year in series_years:
            denom_a = denominator_lookup.get((str(pair.source_a), str(pair.assigned_label), int(year)))
            denom_b = denominator_lookup.get(("corporate", str(pair.assigned_label), int(year)))
            docs_a = sum(
                numerator_lookup.get((str(pair.source_a), str(pair.assigned_label), topic_id, int(year)), (0, 0))[0]
                for topic_id in noncorp_ids
            )
            chunks_a = sum(
                numerator_lookup.get((str(pair.source_a), str(pair.assigned_label), topic_id, int(year)), (0, 0))[1]
                for topic_id in noncorp_ids
            )
            docs_b, chunks_b = numerator_lookup.get(
                ("corporate", str(pair.assigned_label), int(pair.corporate_micro_topic_id), int(year)),
                (0, 0),
            )
            if docs_a > 0:
                source_a_active_years.append(int(year))
            if docs_b > 0:
                source_b_active_years.append(int(year))
            total_docs_a += int(docs_a)
            total_chunks_a += int(chunks_a)
            total_docs_b += int(docs_b)
            total_chunks_b += int(chunks_b)

            rows.append(
                {
                    "aggregate_pair_id": str(pair.aggregate_pair_id),
                    "dyad": str(pair.dyad),
                    "macro_topic": str(pair.macro_topic),
                    "macro_topic_name": str(pair.macro_topic_name),
                    "source_a": str(pair.source_a),
                    "subgroup_a": str(pair.subgroup_a),
                    "source_b": "corporate",
                    "subgroup_b": str(pair.subgroup_b),
                    "matched_corporate_group_id": str(pair.matched_corporate_group_id),
                    "corporate_micro_topic_id": int(pair.corporate_micro_topic_id),
                    "year": int(year),
                    "source_a_doc_numerator": int(docs_a) if denom_a is not None else np.nan,
                    "source_b_doc_numerator": int(docs_b) if denom_b is not None else np.nan,
                    "source_a_chunk_numerator": int(chunks_a) if denom_a is not None else np.nan,
                    "source_b_chunk_numerator": int(chunks_b) if denom_b is not None else np.nan,
                    "source_a_doc_denominator": int(denom_a[0]) if denom_a is not None else np.nan,
                    "source_b_doc_denominator": int(denom_b[0]) if denom_b is not None else np.nan,
                    "source_a_chunk_denominator": int(denom_a[1]) if denom_a is not None else np.nan,
                    "source_b_chunk_denominator": int(denom_b[1]) if denom_b is not None else np.nan,
                    "source_a_share_docs": safe_ratio(docs_a, denom_a[0] if denom_a is not None else np.nan),
                    "source_b_share_docs": safe_ratio(docs_b, denom_b[0] if denom_b is not None else np.nan),
                    "source_a_share_chunks": safe_ratio(chunks_a, denom_a[1] if denom_a is not None else np.nan),
                    "source_b_share_chunks": safe_ratio(chunks_b, denom_b[1] if denom_b is not None else np.nan),
                }
            )

        overlap_years = sorted(set(source_a_active_years).intersection(source_b_active_years))
        diagnostics.append(
            {
                "aggregate_pair_id": str(pair.aggregate_pair_id),
                "dyad": str(pair.dyad),
                "macro_topic": str(pair.macro_topic),
                "macro_topic_name": str(pair.macro_topic_name),
                "source_a": str(pair.source_a),
                "subgroup_a": str(pair.subgroup_a),
                "source_b": "corporate",
                "subgroup_b": str(pair.subgroup_b),
                "matched_corporate_group_id": str(pair.matched_corporate_group_id),
                "corporate_micro_topic_id": int(pair.corporate_micro_topic_id),
                "corporate_topic_name_original": str(pair.corporate_topic_name_original),
                "constituent_noncorporate_micro_topic_ids_json": str(pair.constituent_noncorporate_micro_topic_ids_json),
                "constituent_noncorporate_group_ids_json": str(pair.constituent_noncorporate_group_ids_json),
                "constituent_noncorporate_topic_names_json": str(pair.constituent_noncorporate_topic_names_json),
                "constituent_noncorporate_count": int(pair.constituent_noncorporate_count),
                "direct_pair_count": int(pair.direct_pair_count),
                "max_cosine_similarity": float(pair.max_cosine_similarity),
                "mean_cosine_similarity": float(pair.mean_cosine_similarity),
                "source_a_active_year_count": int(len(source_a_active_years)),
                "source_b_active_year_count": int(len(source_b_active_years)),
                "source_a_first_year": int(min(source_a_active_years)) if source_a_active_years else np.nan,
                "source_b_first_year": int(min(source_b_active_years)) if source_b_active_years else np.nan,
                "source_a_last_year": int(max(source_a_active_years)) if source_a_active_years else np.nan,
                "source_b_last_year": int(max(source_b_active_years)) if source_b_active_years else np.nan,
                "source_a_document_count": int(total_docs_a),
                "source_b_document_count": int(total_docs_b),
                "source_a_chunk_count": int(total_chunks_a),
                "source_b_chunk_count": int(total_chunks_b),
                "overlap_year_count": int(len(overlap_years)),
                "overlap_years_json": json_dumps(overlap_years),
                "balanced_is_viable": bool(
                    len(source_a_active_years) >= BALANCED_RULE["min_active_years_per_side"]
                    and len(source_b_active_years) >= BALANCED_RULE["min_active_years_per_side"]
                    and len(overlap_years) >= BALANCED_RULE["min_overlap_years"]
                    and total_docs_a >= BALANCED_RULE["min_docs_per_side"]
                    and total_docs_b >= BALANCED_RULE["min_docs_per_side"]
                    and total_chunks_a >= BALANCED_RULE["min_chunks_per_side"]
                    and total_chunks_b >= BALANCED_RULE["min_chunks_per_side"]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["aggregate_pair_id", "year"]).reset_index(drop=True), pd.DataFrame(diagnostics)


def build_lag_summary(series: pd.DataFrame, diagnostics: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for diag in diagnostics.itertuples(index=False):
        if not bool(diag.balanced_is_viable):
            summary_rows.append(
                {
                    **diag._asdict(),
                    "best_lag_docs": np.nan,
                    "best_correlation_docs": np.nan,
                    "best_overlap_docs": 0,
                    "low_information_flag_docs": True,
                    "unstable_series_flag_docs": True,
                }
            )
            continue
        pair_series = series.loc[series["aggregate_pair_id"] == diag.aggregate_pair_id].copy().sort_values("year")
        metric_rows: list[dict] = []
        for lag in lags:
            corr, overlap = correlation_for_lag(pair_series, "source_a_share_docs", "source_b_share_docs", lag)
            metric_rows.append({"lag": int(lag), "correlation": corr, "abs_correlation": abs(corr) if corr is not None else np.nan, "n_overlap_years": int(overlap)})
        metric_df = pd.DataFrame(metric_rows).dropna(subset=["correlation"]).copy()
        if metric_df.empty:
            summary_rows.append(
                {
                    **diag._asdict(),
                    "best_lag_docs": np.nan,
                    "best_correlation_docs": np.nan,
                    "best_overlap_docs": 0,
                    "low_information_flag_docs": True,
                    "unstable_series_flag_docs": True,
                }
            )
            continue
        metric_df = metric_df.sort_values(["abs_correlation", "n_overlap_years"], ascending=[False, False]).reset_index(drop=True)
        best = metric_df.iloc[0]
        second_abs = float(metric_df.iloc[1]["abs_correlation"]) if len(metric_df) > 1 else np.nan
        unstable = bool(len(metric_df) < 3 or (pd.notna(second_abs) and abs(float(best["abs_correlation"]) - second_abs) < 0.05))
        summary_rows.append(
            {
                **diag._asdict(),
                "best_lag_docs": int(best["lag"]),
                "best_correlation_docs": float(best["correlation"]),
                "best_overlap_docs": int(best["n_overlap_years"]),
                "low_information_flag_docs": bool(int(best["n_overlap_years"]) < MIN_OVERLAP),
                "unstable_series_flag_docs": unstable,
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_root)
    if args.lag_min > args.lag_max:
        raise SystemExit("--lag-min must be less than or equal to --lag-max")
    lags = list(range(args.lag_min, args.lag_max + 1))

    pair_evidence = pd.read_csv(args.pair_evidence_csv)
    annual_counts = pd.read_csv(args.annual_counts_csv)
    denominators = pd.read_csv(args.denominators_csv)

    aggregate_pairs = build_aggregate_pairs(pair_evidence, threshold=args.similarity_threshold)
    series, diagnostics = build_series(aggregate_pairs, annual_counts, denominators) if not aggregate_pairs.empty else (pd.DataFrame(), pd.DataFrame())
    lag_summary = build_lag_summary(series, diagnostics, lags=lags) if not diagnostics.empty else pd.DataFrame()

    if not lag_summary.empty:
        classified = lag_summary.apply(
            lambda row: classify(
                row.get("best_lag_docs"),
                row.get("best_correlation_docs"),
                bool(row.get("low_information_flag_docs", True)),
                bool(row.get("unstable_series_flag_docs", True)),
            ),
            axis=1,
            result_type="expand",
        )
        lag_summary["precedence_class"] = classified[0]
        lag_summary["precedence_reason"] = classified[1]
        lag_summary["precedence_type_label"] = lag_summary.apply(
            lambda row: precedence_type_label(str(row["source_a"]), str(row["source_b"]), str(row["precedence_class"])),
            axis=1,
        )
        lag_summary["is_stable_leading_pair"] = lag_summary["precedence_type_label"] != "synchronous_or_unclear"
    else:
        lag_summary = pd.DataFrame(
            columns=[
                "aggregate_pair_id",
                "precedence_class",
                "precedence_reason",
                "precedence_type_label",
                "is_stable_leading_pair",
            ]
        )

    aggregate_pairs_path = args.output_root / "aggregate_pairs.csv"
    series_path = args.output_root / "aggregate_pair_annual_series.csv"
    diagnostics_path = args.output_root / "aggregate_pair_temporal_diagnostics.csv"
    lag_summary_path = args.output_root / "aggregate_pair_precedence_classification.csv"
    by_macro_path = args.output_root / "aggregate_pair_precedence_summary_by_macro_topic.csv"
    by_macro_dyad_path = args.output_root / "aggregate_pair_precedence_summary_by_macro_topic_dyad.csv"

    aggregate_pairs.to_csv(aggregate_pairs_path, index=False)
    series.to_csv(series_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    lag_summary.to_csv(lag_summary_path, index=False)
    summarize_groups(lag_summary, ["macro_topic", "macro_topic_name"]).to_csv(by_macro_path, index=False)
    summarize_groups(lag_summary, ["macro_topic", "macro_topic_name", "dyad", "precedence_type_label"]).to_csv(
        by_macro_dyad_path, index=False
    )

    write_json(
        args.output_root / "aggregate_temporal_manifest.json",
        {
            "pair_evidence_csv": str(args.pair_evidence_csv),
            "annual_counts_csv": str(args.annual_counts_csv),
            "denominators_csv": str(args.denominators_csv),
            "similarity_threshold": args.similarity_threshold,
            "temporal_rule": {"name": "balanced", **BALANCED_RULE},
            "lags": lags,
            "min_overlap": MIN_OVERLAP,
            "correlation_floor": CORRELATION_FLOOR,
            "row_counts": {
                "aggregate_pairs": int(aggregate_pairs.shape[0]),
                "aggregate_series_rows": int(series.shape[0]),
                "aggregate_diagnostics_rows": int(diagnostics.shape[0]),
                "aggregate_precedence_rows": int(lag_summary.shape[0]),
                "stable_leading_pairs": int(lag_summary["is_stable_leading_pair"].fillna(False).sum()) if not lag_summary.empty else 0,
            },
            "outputs": {
                "aggregate_pairs": str(aggregate_pairs_path),
                "aggregate_pair_annual_series": str(series_path),
                "aggregate_pair_temporal_diagnostics": str(diagnostics_path),
                "aggregate_pair_precedence_classification": str(lag_summary_path),
                "aggregate_pair_precedence_summary_by_macro_topic": str(by_macro_path),
                "aggregate_pair_precedence_summary_by_macro_topic_dyad": str(by_macro_dyad_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
