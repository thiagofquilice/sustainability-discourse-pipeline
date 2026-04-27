#!/usr/bin/env python3
"""Analyze lagged temporal relations between corporate anchors and external topics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PIPELINE_ROOT = Path("paper_pipeline")
RELATIVE_ROOT = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_relative_longitudinal_series"
DEFAULT_SERIES_CSV = RELATIVE_ROOT / "corporate_topic_external_relative_longitudinal.csv"
DEFAULT_TOPIC_TABLES_DIR = RELATIVE_ROOT / "topic_tables"
DEFAULT_MERGED_ROOT = PIPELINE_ROOT / "outputs" / "bertopic_micro_merged_multiaspect_reviewed"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_source_topic_relations"

YEARS = list(range(2000, 2026))
MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
CORRELATION_FLOOR = 0.30
AMBIGUOUS_BEST_LAG_MARGIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-csv", type=Path, default=DEFAULT_SERIES_CSV)
    parser.add_argument("--topic-tables-dir", type=Path, default=DEFAULT_TOPIC_TABLES_DIR)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag-min", type=int, default=-3)
    parser.add_argument("--lag-max", type=int, default=3)
    parser.add_argument("--min-overlap", type=int, default=4)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def safe_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def detect_document_id_column(columns: pd.Index) -> str:
    for name in ["source_doc_id", "document_id", "doc_id"]:
        if name in columns:
            return name
    raise KeyError("Could not find source_doc_id, document_id, or doc_id in document_topics.csv")


def bh_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype="float64")
    valid = pd.to_numeric(p_values, errors="coerce").dropna().clip(lower=0, upper=1)
    if valid.empty:
        return adjusted
    order = valid.sort_values(kind="mergesort").index
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)
    q_values = ranked * m / np.arange(1, m + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    adjusted.loc[order] = np.clip(q_values, 0.0, 1.0)
    return adjusted


def apply_q_values(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    result = details.copy()
    result["spearman_q_relation"] = result.groupby("relation_id", group_keys=False)["spearman_p"].transform(bh_adjust)
    result["pearson_q_relation"] = result.groupby("relation_id", group_keys=False)["pearson_p"].transform(bh_adjust)
    result["spearman_q_table"] = bh_adjust(result["spearman_p"])
    result["pearson_q_table"] = bh_adjust(result["pearson_p"])
    return result


def load_aligned_topic_tables(topic_tables_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        path = topic_tables_dir / f"{macro_topic}_aligned_aggregate_topics.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    aligned = pd.concat(frames, ignore_index=True)
    for column in [
        "final_merge_group_id_noncorporate",
        "final_merge_group_id_corporate",
        "subgroup_noncorporate",
        "subgroup_corporate",
        "source_noncorporate",
    ]:
        aligned[column] = aligned[column].astype(str)
    return aligned


def annual_value_series(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "year_document_count",
        "annual_source_macro_document_count",
        "annual_document_prevalence",
    ]
    series = frame[columns].copy()
    series["year"] = pd.to_numeric(series["year"], errors="coerce")
    series = series.dropna(subset=["year"]).copy()
    series["year"] = series["year"].astype(int)
    for column in columns[1:]:
        series[column] = pd.to_numeric(series[column], errors="coerce").fillna(0.0)
    series["value_for_correlation"] = series["annual_document_prevalence"].where(
        series["annual_source_macro_document_count"] > 0,
        np.nan,
    )
    return series.sort_values("year").reset_index(drop=True)


def series_stats(series: pd.DataFrame) -> dict[str, Any]:
    if series.empty:
        return {
            "first_active_year": np.nan,
            "peak_year": np.nan,
            "active_year_count": 0,
            "total_year_document_count": 0,
            "peak_annual_document_prevalence": np.nan,
        }
    active = series.loc[series["year_document_count"] > 0].copy()
    if active.empty:
        return {
            "first_active_year": np.nan,
            "peak_year": np.nan,
            "active_year_count": 0,
            "total_year_document_count": 0,
            "peak_annual_document_prevalence": np.nan,
        }
    peak = active.sort_values(
        ["annual_document_prevalence", "year_document_count", "year"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "first_active_year": int(active["year"].min()),
        "peak_year": int(peak["year"]),
        "active_year_count": int(active["year"].nunique()),
        "total_year_document_count": int(active["year_document_count"].sum()),
        "peak_annual_document_prevalence": float(peak["annual_document_prevalence"]),
    }


def correlation_for_lag(
    external_series: pd.DataFrame,
    corporate_series: pd.DataFrame,
    lag: int,
    min_overlap: int,
) -> dict[str, Any]:
    left = external_series[["year", "value_for_correlation"]].rename(columns={"value_for_correlation": "external"})
    right = corporate_series[["year", "value_for_correlation"]].rename(columns={"value_for_correlation": "corporate"})
    right = right.copy()
    right["year"] = right["year"] - lag
    aligned = left.merge(right, on="year", how="inner").dropna(subset=["external", "corporate"])
    overlap = int(len(aligned))
    external_constant = bool(overlap > 0 and aligned["external"].nunique(dropna=True) <= 1)
    corporate_constant = bool(overlap > 0 and aligned["corporate"].nunique(dropna=True) <= 1)
    low_information = bool(overlap < min_overlap or external_constant or corporate_constant)

    row: dict[str, Any] = {
        "lag": int(lag),
        "n_overlap_years": overlap,
        "external_series_constant_flag": external_constant,
        "corporate_series_constant_flag": corporate_constant,
        "low_information_flag": low_information,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }
    if low_information:
        return row

    spearman_stat, spearman_p = spearmanr(aligned["external"], aligned["corporate"])
    pearson_stat, pearson_p = pearsonr(aligned["external"], aligned["corporate"])
    row.update(
        {
            "spearman_r": float(spearman_stat),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_stat),
            "pearson_p": float(pearson_p),
        }
    )
    return row


def build_lag_details(
    relation_rows: list[dict[str, Any]],
    lags: list[int],
    min_overlap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relation in relation_rows:
        external_series = relation.pop("_external_series")
        corporate_series = relation.pop("_corporate_series")
        for lag in lags:
            metric = correlation_for_lag(external_series, corporate_series, lag, min_overlap)
            rows.append(
                {
                    **relation,
                    **metric,
                    "abs_spearman_r": abs(metric["spearman_r"]) if pd.notna(metric["spearman_r"]) else np.nan,
                    "abs_pearson_r": abs(metric["pearson_r"]) if pd.notna(metric["pearson_r"]) else np.nan,
                    "lag_sign_convention": "positive lag means external source leads corporate by that many years",
                }
            )
    return apply_q_values(pd.DataFrame(rows))


def classify_temporal_relation(best: pd.Series | None, ambiguous: bool) -> tuple[str, str]:
    if best is None:
        return "synchronous_or_unclear", "no_valid_lag"
    if bool(best.get("low_information_flag", True)):
        return "synchronous_or_unclear", "low_information"
    if pd.isna(best.get("spearman_r")) or abs(float(best["spearman_r"])) < CORRELATION_FLOOR:
        return "synchronous_or_unclear", "correlation_below_floor"
    lag = int(best["lag"])
    if lag > 0:
        reason = "positive_best_lag_external_leads"
        return "external_leads", f"{reason}_ambiguous" if ambiguous else reason
    if lag < 0:
        reason = "negative_best_lag_corporate_leads"
        return "corporate_leads", f"{reason}_ambiguous" if ambiguous else reason
    return "synchronous_or_unclear", "zero_best_lag"


def summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    metadata_columns = [
        column
        for column in details.columns
        if column
        not in {
            "lag",
            "n_overlap_years",
            "external_series_constant_flag",
            "corporate_series_constant_flag",
            "low_information_flag",
            "spearman_r",
            "spearman_p",
            "pearson_r",
            "pearson_p",
            "abs_spearman_r",
            "abs_pearson_r",
            "spearman_q_relation",
            "pearson_q_relation",
            "spearman_q_table",
            "pearson_q_table",
            "lag_sign_convention",
        }
    ]

    for relation_id, group in details.groupby("relation_id", sort=False):
        metadata = group.iloc[0][metadata_columns].to_dict()
        valid = group.dropna(subset=["spearman_r"]).copy()
        if valid.empty:
            best = None
            ambiguous = False
        else:
            valid["abs_lag"] = valid["lag"].abs()
            valid = valid.sort_values(
                ["abs_spearman_r", "n_overlap_years", "abs_lag", "lag"],
                ascending=[False, False, True, False],
                kind="mergesort",
            )
            best = valid.iloc[0]
            second_abs = safe_float(valid.iloc[1]["abs_spearman_r"]) if len(valid) > 1 else np.nan
            ambiguous = bool(
                pd.notna(second_abs)
                and abs(float(best["abs_spearman_r"]) - second_abs) < AMBIGUOUS_BEST_LAG_MARGIN
            )

        label, reason = classify_temporal_relation(best, ambiguous)
        summary_rows.append(
            {
                **metadata,
                "best_lag_spearman": int(best["lag"]) if best is not None else np.nan,
                "best_spearman_r": float(best["spearman_r"]) if best is not None else np.nan,
                "best_spearman_p": float(best["spearman_p"]) if best is not None else np.nan,
                "best_spearman_q_relation": float(best["spearman_q_relation"]) if best is not None else np.nan,
                "best_spearman_q_table": float(best["spearman_q_table"]) if best is not None else np.nan,
                "pearson_r_at_best_spearman_lag": float(best["pearson_r"]) if best is not None else np.nan,
                "pearson_p_at_best_spearman_lag": float(best["pearson_p"]) if best is not None else np.nan,
                "pearson_q_table_at_best_spearman_lag": float(best["pearson_q_table"]) if best is not None else np.nan,
                "best_overlap_years": int(best["n_overlap_years"]) if best is not None else 0,
                "ambiguous_best_lag_flag": bool(ambiguous),
                "temporal_relation_label": label,
                "temporal_relation_reason": reason,
            }
        )
    return pd.DataFrame(summary_rows)


def load_individual_external_series(relations: pd.DataFrame, merged_root: Path) -> pd.DataFrame:
    requested = relations[
        ["source_noncorporate", "subgroup_noncorporate", "final_merge_group_id_noncorporate", "macro_topic"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for subgroup, subgroup_requests in requested.groupby("subgroup_noncorporate", sort=True):
        path = merged_root / subgroup / "document_topics.csv"
        header = pd.read_csv(path, nrows=0)
        doc_col = detect_document_id_column(header.columns)
        doc_topics = pd.read_csv(path, usecols=[doc_col, "year", "final_merge_group_id"])
        doc_topics = doc_topics.rename(columns={doc_col: "stable_doc_id"})
        doc_topics = doc_topics.dropna(subset=["stable_doc_id", "year", "final_merge_group_id"]).copy()
        doc_topics["stable_doc_id"] = doc_topics["stable_doc_id"].astype(str)
        doc_topics["final_merge_group_id"] = doc_topics["final_merge_group_id"].astype(str)
        doc_topics["year"] = pd.to_numeric(doc_topics["year"], errors="coerce")
        doc_topics = doc_topics.dropna(subset=["year"]).copy()
        doc_topics["year"] = doc_topics["year"].astype(int)

        denominator = doc_topics.groupby("year", sort=False)["stable_doc_id"].nunique().astype(int).to_dict()
        numerator = (
            doc_topics.groupby(["final_merge_group_id", "year"], sort=False)["stable_doc_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        group_ids = sorted(subgroup_requests["final_merge_group_id_noncorporate"].astype(str).unique().tolist())
        source = str(subgroup_requests["source_noncorporate"].iloc[0])
        macro_topic = str(subgroup_requests["macro_topic"].iloc[0])
        rows = []
        for group_id in group_ids:
            for year in YEARS:
                denom = int(denominator.get(year, 0))
                docs = int(numerator.get((group_id, year), 0))
                rows.append(
                    {
                        "source": source,
                        "macro_topic": macro_topic,
                        "subgroup": subgroup,
                        "final_merge_group_id": group_id,
                        "year": int(year),
                        "year_document_count": docs,
                        "annual_source_macro_document_count": denom,
                        "annual_document_prevalence": float(docs) / float(denom) if denom else 0.0,
                        "value_for_correlation": float(docs) / float(denom) if denom else np.nan,
                    }
                )
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_series(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column].astype(str) == str(value)]
    return annual_value_series(selected)


def build_aggregate_relation_rows(series: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corporate_rows = series[series["series_role"] == "corporate"].copy()
    for anchor in corporate_rows[
        [
            "macro_topic",
            "macro_topic_name",
            "corporate_subgroup",
            "corporate_final_merge_group_id",
            "corporate_topic_label",
            "corporate_unique_document_count",
        ]
    ].drop_duplicates().itertuples(index=False):
        corporate_series = get_series(
            corporate_rows,
            {
                "macro_topic": anchor.macro_topic,
                "corporate_final_merge_group_id": anchor.corporate_final_merge_group_id,
            },
        )
        corporate_stats = series_stats(corporate_series)
        for source, role in [("academic", "academic_aggregate"), ("media", "media_aggregate")]:
            external_raw = series[
                (series["series_role"] == role)
                & (series["macro_topic"].astype(str) == str(anchor.macro_topic))
                & (
                    series["corporate_final_merge_group_id"].astype(str)
                    == str(anchor.corporate_final_merge_group_id)
                )
            ].copy()
            if external_raw.empty or safe_int(external_raw["n_contributing_external_topics"].max()) <= 0:
                continue
            external_series = annual_value_series(external_raw)
            external_stats = series_stats(external_series)
            first_gap, peak_gap = temporal_gaps(external_stats, corporate_stats)
            rows.append(
                {
                    "relation_id": f"aggregate::{anchor.macro_topic}::{anchor.corporate_final_merge_group_id}::{source}",
                    "relation_level": "aggregate",
                    "macro_topic": anchor.macro_topic,
                    "macro_topic_name": anchor.macro_topic_name,
                    "corporate_subgroup": anchor.corporate_subgroup,
                    "corporate_final_merge_group_id": anchor.corporate_final_merge_group_id,
                    "corporate_topic_label": anchor.corporate_topic_label,
                    "external_source": source,
                    "external_relation_label": f"{source} aggregate",
                    "external_topic_count": safe_int(external_raw["n_contributing_external_topics"].max()),
                    "corporate_unique_document_count": safe_int(anchor.corporate_unique_document_count),
                    "external_unique_document_count": safe_int(external_raw["series_unique_document_count"].max()),
                    "corporate_first_active_year": corporate_stats["first_active_year"],
                    "external_first_active_year": external_stats["first_active_year"],
                    "first_active_year_gap": first_gap,
                    "corporate_peak_year": corporate_stats["peak_year"],
                    "external_peak_year": external_stats["peak_year"],
                    "peak_year_gap": peak_gap,
                    "corporate_active_year_count": corporate_stats["active_year_count"],
                    "external_active_year_count": external_stats["active_year_count"],
                    "corporate_peak_annual_document_prevalence": corporate_stats["peak_annual_document_prevalence"],
                    "external_peak_annual_document_prevalence": external_stats["peak_annual_document_prevalence"],
                    "_external_series": external_series,
                    "_corporate_series": corporate_series,
                }
            )
    return rows


def temporal_gaps(external_stats: dict[str, Any], corporate_stats: dict[str, Any]) -> tuple[float, float]:
    external_first = external_stats.get("first_active_year")
    corporate_first = corporate_stats.get("first_active_year")
    external_peak = external_stats.get("peak_year")
    corporate_peak = corporate_stats.get("peak_year")
    first_gap = (
        float(corporate_first) - float(external_first)
        if pd.notna(external_first) and pd.notna(corporate_first)
        else np.nan
    )
    peak_gap = (
        float(corporate_peak) - float(external_peak)
        if pd.notna(external_peak) and pd.notna(corporate_peak)
        else np.nan
    )
    return first_gap, peak_gap


def build_individual_relation_rows(
    series: pd.DataFrame,
    relations: pd.DataFrame,
    individual_series: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corporate_rows = series[series["series_role"] == "corporate"].copy()
    corporate_series_cache: dict[tuple[str, str], pd.DataFrame] = {}
    corporate_stats_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for relation in relations.itertuples(index=False):
        corp_key = (str(relation.macro_topic), str(relation.final_merge_group_id_corporate))
        if corp_key not in corporate_series_cache:
            corporate_series_cache[corp_key] = get_series(
                corporate_rows,
                {
                    "macro_topic": corp_key[0],
                    "corporate_final_merge_group_id": corp_key[1],
                },
            )
            corporate_stats_cache[corp_key] = series_stats(corporate_series_cache[corp_key])
        external_series = individual_series[
            (individual_series["subgroup"].astype(str) == str(relation.subgroup_noncorporate))
            & (
                individual_series["final_merge_group_id"].astype(str)
                == str(relation.final_merge_group_id_noncorporate)
            )
        ].copy()
        external_stats = series_stats(external_series)
        corporate_stats = corporate_stats_cache[corp_key]
        first_gap, peak_gap = temporal_gaps(external_stats, corporate_stats)
        rows.append(
            {
                "relation_id": (
                    f"individual::{relation.macro_topic}::{relation.final_merge_group_id_corporate}"
                    f"::{relation.source_noncorporate}::{relation.final_merge_group_id_noncorporate}"
                ),
                "relation_level": "individual",
                "macro_topic": relation.macro_topic,
                "macro_topic_name": relation.macro_topic_name,
                "corporate_subgroup": relation.subgroup_corporate,
                "corporate_final_merge_group_id": relation.final_merge_group_id_corporate,
                "corporate_topic_label": relation.topic_label_refined_corporate,
                "external_source": relation.source_noncorporate,
                "external_subgroup": relation.subgroup_noncorporate,
                "external_final_merge_group_id": relation.final_merge_group_id_noncorporate,
                "external_micro_topic_id": relation.micro_topic_id_noncorporate,
                "external_topic_label": relation.topic_label_refined_noncorporate,
                "external_relation_label": relation.topic_label_refined_noncorporate,
                "external_topic_count": 1,
                "corporate_unique_document_count": safe_int(relation.corporate_unique_document_count),
                "external_unique_document_count": safe_int(relation.external_unique_document_count),
                "best_cosine_similarity": safe_float(relation.best_cosine_similarity),
                "direct_pair_count": safe_int(relation.direct_pair_count),
                "best_pair_id": clean_text(relation.best_pair_id),
                "inclusion_reason": clean_text(relation.inclusion_reason),
                "corporate_first_active_year": corporate_stats["first_active_year"],
                "external_first_active_year": external_stats["first_active_year"],
                "first_active_year_gap": first_gap,
                "corporate_peak_year": corporate_stats["peak_year"],
                "external_peak_year": external_stats["peak_year"],
                "peak_year_gap": peak_gap,
                "corporate_active_year_count": corporate_stats["active_year_count"],
                "external_active_year_count": external_stats["active_year_count"],
                "corporate_peak_annual_document_prevalence": corporate_stats["peak_annual_document_prevalence"],
                "external_peak_annual_document_prevalence": external_stats["peak_annual_document_prevalence"],
                "_external_series": external_series,
                "_corporate_series": corporate_series_cache[corp_key],
            }
        )
    return rows


def reorder_summary_columns(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    common = [
        "relation_id",
        "macro_topic",
        "macro_topic_name",
        "corporate_final_merge_group_id",
        "corporate_topic_label",
        "external_source",
        "external_relation_label",
        "external_topic_count",
        "corporate_unique_document_count",
        "external_unique_document_count",
        "best_lag_spearman",
        "best_spearman_r",
        "best_spearman_p",
        "best_spearman_q_relation",
        "best_spearman_q_table",
        "pearson_r_at_best_spearman_lag",
        "pearson_p_at_best_spearman_lag",
        "pearson_q_table_at_best_spearman_lag",
        "best_overlap_years",
        "first_active_year_gap",
        "peak_year_gap",
        "temporal_relation_label",
        "temporal_relation_reason",
        "ambiguous_best_lag_flag",
        "corporate_first_active_year",
        "external_first_active_year",
        "corporate_peak_year",
        "external_peak_year",
        "corporate_active_year_count",
        "external_active_year_count",
        "corporate_peak_annual_document_prevalence",
        "external_peak_annual_document_prevalence",
    ]
    individual_extra = [
        "external_final_merge_group_id",
        "external_micro_topic_id",
        "external_topic_label",
        "best_cosine_similarity",
        "direct_pair_count",
        "best_pair_id",
        "inclusion_reason",
    ]
    columns = common + (individual_extra if level == "individual" else [])
    columns += [column for column in frame.columns if column not in columns]
    return frame[columns]


def write_outputs(
    aggregate_summary: pd.DataFrame,
    aggregate_details: pd.DataFrame,
    individual_summary: pd.DataFrame,
    individual_details: pd.DataFrame,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate_summary": str(output_dir / "aggregate_source_relations_summary.csv"),
        "aggregate_lag_details": str(output_dir / "aggregate_source_relations_lag_details.csv"),
        "individual_summary": str(output_dir / "individual_source_relations_summary.csv"),
        "individual_lag_details": str(output_dir / "individual_source_relations_lag_details.csv"),
        "excel": str(output_dir / "corporate_focus_source_topic_relations.xlsx"),
        "manifest": str(output_dir / "source_topic_relations_manifest.json"),
    }
    aggregate_summary.to_csv(paths["aggregate_summary"], index=False)
    aggregate_details.to_csv(paths["aggregate_lag_details"], index=False)
    individual_summary.to_csv(paths["individual_summary"], index=False)
    individual_details.to_csv(paths["individual_lag_details"], index=False)

    readme = pd.DataFrame(
        [
            {
                "item": "metric",
                "description": "annual_document_prevalence: annual topic documents divided by source-domain documents in the same year.",
            },
            {
                "item": "lag convention",
                "description": "Positive lag means the external academic/media series leads the corporate series by that many years.",
            },
            {
                "item": "primary statistic",
                "description": "Spearman correlation is used to choose the best lag; Pearson is reported as a robustness check.",
            },
            {
                "item": "gap convention",
                "description": "first_active_year_gap and peak_year_gap equal corporate year minus external year; positive values mean external comes earlier.",
            },
            {
                "item": "interpretation",
                "description": "These are exploratory temporal association diagnostics, not causal estimates.",
            },
        ]
    )
    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        aggregate_summary.to_excel(writer, sheet_name="Aggregate relations", index=False)
        individual_summary.to_excel(writer, sheet_name="Individual topic relations", index=False)
        aggregate_details.to_excel(writer, sheet_name="Aggregate lag details", index=False)
        individual_details.to_excel(writer, sheet_name="Individual lag details", index=False)

    manifest["outputs"] = paths
    Path(paths["manifest"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def main() -> None:
    args = parse_args()
    if args.lag_min > args.lag_max:
        raise SystemExit("--lag-min must be less than or equal to --lag-max")
    lags = list(range(args.lag_min, args.lag_max + 1))

    series = pd.read_csv(args.series_csv)
    required = {"annual_document_prevalence", "year_document_count", "annual_source_macro_document_count"}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise KeyError(f"Missing required annual prevalence columns in {args.series_csv}: {missing}")

    relations = load_aligned_topic_tables(args.topic_tables_dir)
    individual_series = load_individual_external_series(relations, args.merged_root)

    aggregate_relation_rows = build_aggregate_relation_rows(series)
    individual_relation_rows = build_individual_relation_rows(series, relations, individual_series)

    aggregate_details = build_lag_details(aggregate_relation_rows, lags, args.min_overlap)
    individual_details = build_lag_details(individual_relation_rows, lags, args.min_overlap)
    aggregate_summary = reorder_summary_columns(summarize_details(aggregate_details), "aggregate")
    individual_summary = reorder_summary_columns(summarize_details(individual_details), "individual")

    manifest = {
        "inputs": {
            "series_csv": str(args.series_csv),
            "topic_tables_dir": str(args.topic_tables_dir),
            "merged_root": str(args.merged_root),
        },
        "method": {
            "metric": "annual_document_prevalence",
            "lags": lags,
            "min_overlap": int(args.min_overlap),
            "primary_statistic": "spearman",
            "secondary_statistic": "pearson",
            "correlation_floor_for_direction_label": CORRELATION_FLOOR,
            "ambiguous_best_lag_margin": AMBIGUOUS_BEST_LAG_MARGIN,
            "lag_sign_convention": "positive lag means academic/media external series leads corporate by that many years",
            "multiple_testing_correction": "Benjamini-Hochberg within each relation and across each full lag-detail table",
        },
        "row_counts": {
            "aggregate_summary": int(len(aggregate_summary)),
            "aggregate_lag_details": int(len(aggregate_details)),
            "individual_summary": int(len(individual_summary)),
            "individual_lag_details": int(len(individual_details)),
            "aligned_external_topic_rows": int(len(relations)),
            "unique_individual_external_topics": int(
                relations[
                    ["source_noncorporate", "subgroup_noncorporate", "final_merge_group_id_noncorporate"]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        },
    }
    paths = write_outputs(
        aggregate_summary,
        aggregate_details,
        individual_summary,
        individual_details,
        args.output_dir,
        manifest,
    )

    print(json.dumps(manifest["row_counts"], indent=2, ensure_ascii=False))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
