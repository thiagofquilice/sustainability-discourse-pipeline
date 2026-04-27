#!/usr/bin/env python3
"""Add temporal grouping outputs to the broad same-issue discourse branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import (
    DEFAULT_DISCOURSE_FUNCTION_WITH_ACADEMIC_MEDIA_OUTPUT_ROOT,
    DEFAULT_THRESHOLD_065_OUTPUT_ROOT,
    configure_logging,
    write_json,
)


LAGS = [-2, -1, 0, 1, 2]
MIN_OVERLAP = 4
CORRELATION_FLOOR = 0.30
README_START_MARKER = "<!-- temporal-grouping:start -->"
README_END_MARKER = "<!-- temporal-grouping:end -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DISCOURSE_FUNCTION_WITH_ACADEMIC_MEDIA_OUTPUT_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_THRESHOLD_065_OUTPUT_ROOT)
    parser.add_argument("--analyst-csv", type=Path, default=None)
    parser.add_argument("--annual-counts-csv", type=Path, default=None)
    parser.add_argument("--denominators-csv", type=Path, default=None)
    parser.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    parser.add_argument("--correlation-floor", type=float, default=CORRELATION_FLOOR)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def pick_column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float:
    if denominator is None or pd.isna(denominator):
        return np.nan
    denominator = float(denominator)
    if denominator == 0:
        return np.nan
    numerator = float(numerator or 0.0)
    return numerator / denominator


def correlation_for_lag(series: pd.DataFrame, col_a: str, col_b: str, lag: int, min_overlap: int) -> tuple[float | None, int]:
    left = series[["year", col_a]].rename(columns={col_a: "x"})
    right = series[["year", col_b]].rename(columns={col_b: "y"}).copy()
    right["year"] = right["year"] - lag
    aligned = left.merge(right, on="year", how="inner").dropna()
    overlap = len(aligned)
    if overlap < min_overlap:
        return None, overlap
    if aligned["x"].nunique() <= 1 or aligned["y"].nunique() <= 1:
        return None, overlap
    return float(np.corrcoef(aligned["x"], aligned["y"])[0, 1]), overlap


def build_temporal_pair_input(analyst: pd.DataFrame) -> pd.DataFrame:
    source_a_col = pick_column(analyst, "source_a", "source_a_x", "source_a_y")
    source_b_col = pick_column(analyst, "source_b", "source_b_x", "source_b_y")
    filtered = analyst.loc[
        (analyst["same_issue_assessment"] == "same_issue")
        & (analyst["keep_for_discourse_analysis"].fillna(False))
    ].copy()
    filtered = filtered.rename(columns={source_a_col: "source_a", source_b_col: "source_b"})
    keep_cols = [
        "pair_id",
        "macro_topic",
        "macro_topic_name",
        "dyad",
        "dyad_label",
        "source_a",
        "subgroup_a",
        "microtopic_id_a",
        "microtopic_label_a",
        "source_b",
        "subgroup_b",
        "microtopic_id_b",
        "microtopic_label_b",
        "cosine_similarity",
        "same_issue_assessment",
        "discourse_function_relation",
        "function_label_a",
        "function_label_b",
        "difference_summary",
        "temporal_interpretation_note",
        "keep_for_discourse_analysis",
        "topic_label_refined_a",
        "topic_label_refined_b",
        "overall_summary_a",
        "overall_summary_b",
    ]
    keep_cols = [col for col in keep_cols if col in filtered.columns]
    return filtered[keep_cols].drop_duplicates(subset=["pair_id"]).reset_index(drop=True)


def build_doc_share_series(
    pairs: pd.DataFrame,
    annual_counts: pd.DataFrame,
    denominators: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    annual_counts = annual_counts.copy()
    denominators = denominators.copy()
    annual_counts["micro_topic_id"] = pd.to_numeric(annual_counts["micro_topic_id"], errors="coerce").astype(int)
    annual_counts["year"] = pd.to_numeric(annual_counts["year"], errors="coerce").astype(int)
    denominators["year"] = pd.to_numeric(denominators["year"], errors="coerce").astype(int)

    numerator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.micro_topic_id), int(row.year)): int(row.microtopic_doc_n)
        for row in annual_counts.itertuples(index=False)
    }
    denominator_lookup = {
        (str(row.source), str(row.assigned_label), int(row.year)): int(row.yes_doc_n)
        for row in denominators.itertuples(index=False)
    }
    years_by_source_topic: dict[tuple[str, str], list[int]] = (
        denominators.groupby(["source", "assigned_label"], dropna=False)["year"]
        .apply(lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique().tolist()))
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        years_a = years_by_source_topic.get((str(pair.source_a), str(pair.macro_topic)), [])
        years_b = years_by_source_topic.get((str(pair.source_b), str(pair.macro_topic)), [])
        series_years = sorted(set(years_a).union(years_b))
        for year in series_years:
            doc_num_a = numerator_lookup.get((str(pair.source_a), str(pair.macro_topic), int(pair.microtopic_id_a), int(year)), 0)
            doc_num_b = numerator_lookup.get((str(pair.source_b), str(pair.macro_topic), int(pair.microtopic_id_b), int(year)), 0)
            doc_den_a = denominator_lookup.get((str(pair.source_a), str(pair.macro_topic), int(year)))
            doc_den_b = denominator_lookup.get((str(pair.source_b), str(pair.macro_topic), int(year)))
            rows.append(
                {
                    "pair_id": str(pair.pair_id),
                    "macro_topic": str(pair.macro_topic),
                    "macro_topic_name": str(pair.macro_topic_name),
                    "dyad": str(pair.dyad),
                    "dyad_label": str(pair.dyad_label),
                    "source_a": str(pair.source_a),
                    "source_b": str(pair.source_b),
                    "microtopic_id_a": int(pair.microtopic_id_a),
                    "microtopic_id_b": int(pair.microtopic_id_b),
                    "year": int(year),
                    "source_a_doc_numerator": int(doc_num_a) if doc_den_a is not None else np.nan,
                    "source_b_doc_numerator": int(doc_num_b) if doc_den_b is not None else np.nan,
                    "source_a_doc_denominator": int(doc_den_a) if doc_den_a is not None else np.nan,
                    "source_b_doc_denominator": int(doc_den_b) if doc_den_b is not None else np.nan,
                    "source_a_share_docs": safe_ratio(doc_num_a, doc_den_a),
                    "source_b_share_docs": safe_ratio(doc_num_b, doc_den_b),
                }
            )
    return pd.DataFrame(rows).sort_values(["pair_id", "year"]).reset_index(drop=True)


def build_lag_outputs(series_df: pd.DataFrame, pairs: pd.DataFrame, min_overlap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for pair in pairs.itertuples(index=False):
        pair_series = series_df.loc[series_df["pair_id"] == pair.pair_id].copy().sort_values("year")
        metric_rows = []
        for lag in LAGS:
            corr, overlap = correlation_for_lag(pair_series, "source_a_share_docs", "source_b_share_docs", lag, min_overlap)
            row = {
                "pair_id": str(pair.pair_id),
                "macro_topic": str(pair.macro_topic),
                "macro_topic_name": str(pair.macro_topic_name),
                "dyad": str(pair.dyad),
                "dyad_label": str(pair.dyad_label),
                "source_a": str(pair.source_a),
                "source_b": str(pair.source_b),
                "microtopic_id_a": int(pair.microtopic_id_a),
                "microtopic_id_b": int(pair.microtopic_id_b),
                "lag": int(lag),
                "correlation": corr,
                "abs_correlation": abs(corr) if corr is not None else np.nan,
                "n_overlap_years": int(overlap),
                "lag_sign_convention": "positive lag means source_a precedes source_b by that many years",
            }
            lag_rows.append(row)
            metric_rows.append(row)

        metric_df = pd.DataFrame(metric_rows)
        valid = metric_df.dropna(subset=["correlation"]).copy()
        if valid.empty:
            summary_rows.append(
                {
                    "pair_id": str(pair.pair_id),
                    "best_lag_docs": np.nan,
                    "best_correlation_docs": np.nan,
                    "best_overlap_docs": 0,
                    "low_information_flag_docs": True,
                    "unstable_series_flag_docs": True,
                }
            )
            continue

        valid = valid.sort_values(["abs_correlation", "n_overlap_years"], ascending=[False, False]).reset_index(drop=True)
        best = valid.iloc[0]
        second_abs = float(valid.iloc[1]["abs_correlation"]) if len(valid) > 1 else np.nan
        unstable = False
        if len(valid) < 3:
            unstable = True
        elif pd.notna(second_abs) and abs(float(best["abs_correlation"]) - second_abs) < 0.05:
            unstable = True

        summary_rows.append(
            {
                "pair_id": str(pair.pair_id),
                "best_lag_docs": int(best["lag"]),
                "best_correlation_docs": float(best["correlation"]),
                "best_overlap_docs": int(best["n_overlap_years"]),
                "low_information_flag_docs": bool(int(best["n_overlap_years"]) < min_overlap),
                "unstable_series_flag_docs": bool(unstable),
            }
        )

    return pd.DataFrame(lag_rows), pd.DataFrame(summary_rows)


def classify_temporal_row(row: pd.Series, correlation_floor: float) -> tuple[str, str]:
    best_lag = row.get("best_lag_docs")
    best_corr = row.get("best_correlation_docs")
    low_info = bool(row.get("low_information_flag_docs", True))
    unstable = bool(row.get("unstable_series_flag_docs", True))
    if pd.isna(best_lag) or pd.isna(best_corr):
        return "indeterminate", "missing_lag_or_correlation"
    if low_info:
        return "indeterminate", "low_information"
    if unstable:
        return "indeterminate", "unstable_series"
    if float(best_corr) < correlation_floor:
        return "indeterminate", "correlation_below_floor"
    if int(best_lag) > 0:
        return "A_precedes_B", "positive_best_lag_docs"
    if int(best_lag) < 0:
        return "B_precedes_A", "negative_best_lag_docs"
    return "synchronous", "zero_best_lag_docs"


def readable_precedence_label(source_a: str, source_b: str, temporal_group: str) -> str:
    if temporal_group == "A_precedes_B":
        return f"{source_a}_precedes_{source_b}"
    if temporal_group == "B_precedes_A":
        return f"{source_b}_precedes_{source_a}"
    return temporal_group


def summarize_grouped(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            pair_count=("pair_id", "nunique"),
            mean_best_correlation_docs=("best_correlation_docs", "mean"),
            median_best_correlation_docs=("best_correlation_docs", "median"),
            mean_abs_best_lag_docs=(
                "best_lag_docs",
                lambda s: float(pd.Series(s).dropna().abs().mean()) if not pd.Series(s).dropna().empty else np.nan,
            ),
        )
        .reset_index()
    )
    return grouped


def build_dyad_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby(["dyad", "dyad_label"], dropna=False)
        .agg(
            pair_count=("pair_id", "nunique"),
            a_precedes_b_count=("temporal_group", lambda s: int((pd.Series(s) == "A_precedes_B").sum())),
            b_precedes_a_count=("temporal_group", lambda s: int((pd.Series(s) == "B_precedes_A").sum())),
            synchronous_count=("temporal_group", lambda s: int((pd.Series(s) == "synchronous").sum())),
            indeterminate_count=("temporal_group", lambda s: int((pd.Series(s) == "indeterminate").sum())),
            mean_best_correlation_docs=("best_correlation_docs", "mean"),
        )
        .reset_index()
    )
    return summary


def replace_or_append_temporal_section(readme_path: Path, temporal_block: str) -> None:
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Same Issue / Different Discourse Function\n"
    if README_START_MARKER in existing and README_END_MARKER in existing:
        start = existing.index(README_START_MARKER)
        end = existing.index(README_END_MARKER) + len(README_END_MARKER)
        updated = existing[:start].rstrip() + "\n\n" + temporal_block.strip() + "\n"
    else:
        updated = existing.rstrip() + "\n\n" + temporal_block.strip() + "\n"
    readme_path.write_text(updated, encoding="utf-8")


def build_temporal_readme_block(
    pair_count: int,
    dyad_summary: pd.DataFrame,
    by_group: pd.DataFrame,
    correlation_floor: float,
    min_overlap: int,
) -> str:
    lines = [
        README_START_MARKER,
        "## Temporal Grouping",
        "",
        f"- temporal input pairs (`same_issue` + `keep_for_discourse_analysis`): `{pair_count}`",
        "- temporal series use document-share within `source × macro-topic × year`.",
        f"- lags tested: `{LAGS}`",
        f"- minimum overlap for correlation: `{min_overlap}` years",
        f"- positive correlation floor for non-indeterminate classification: `{correlation_floor:.2f}`",
        "- `synchronous` is reserved for robust lag-0 cases.",
        "- `indeterminate` captures low-information, unstable, weak-correlation, or otherwise inconclusive cases.",
        "",
        "### Counts by dyad",
    ]
    if dyad_summary.empty:
        lines.append("- No temporal pairs available.")
    else:
        for row in dyad_summary.itertuples(index=False):
            lines.append(
                f"- `{row.dyad_label}`: `{int(row.pair_count)}` pairs | "
                f"`A_precedes_B={int(row.a_precedes_b_count)}` | "
                f"`B_precedes_A={int(row.b_precedes_a_count)}` | "
                f"`synchronous={int(row.synchronous_count)}` | "
                f"`indeterminate={int(row.indeterminate_count)}`"
            )

    if not by_group.empty:
        lines.extend(["", "### Counts by dyad × group"])
        for row in by_group.itertuples(index=False):
            lines.append(f"- `{row.dyad}` / `{row.temporal_group}`: `{int(row.pair_count)}`")

    lines.append(README_END_MARKER)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.output_root.mkdir(parents=True, exist_ok=True)

    analyst_csv = args.analyst_csv or (args.output_root / "same_issue_discourse_analyst_table.csv")
    annual_counts_csv = args.annual_counts_csv or (args.pair_root / "microtopic_annual_counts.csv")
    denominators_csv = args.denominators_csv or (args.pair_root / "source_topic_year_denominators.csv")

    pair_input_csv = args.output_root / "same_issue_temporal_pair_input.csv"
    annual_series_csv = args.output_root / "same_issue_temporal_pair_annual_series_docs.csv"
    lag_details_csv = args.output_root / "same_issue_temporal_lag_details_docs.csv"
    lag_summary_csv = args.output_root / "same_issue_temporal_lag_summary.csv"
    classification_csv = args.output_root / "same_issue_temporal_classification.csv"
    analyst_temporal_csv = args.output_root / "same_issue_temporal_analyst_table.csv"
    analyst_temporal_xlsx = args.output_root / "same_issue_temporal_analyst_table.xlsx"
    summary_by_dyad_csv = args.output_root / "same_issue_temporal_summary_by_dyad.csv"
    summary_by_dyad_group_csv = args.output_root / "same_issue_temporal_summary_by_dyad_group.csv"
    summary_by_label_csv = args.output_root / "same_issue_temporal_summary_by_precedence_label.csv"
    summary_by_macro_group_csv = args.output_root / "same_issue_temporal_summary_by_macro_topic_group.csv"
    manifest_path = args.output_root / "same_issue_temporal_manifest.json"

    analyst = safe_read_csv(analyst_csv)
    annual_counts = safe_read_csv(annual_counts_csv)
    denominators = safe_read_csv(denominators_csv)

    if analyst.empty or annual_counts.empty or denominators.empty:
        empty = pd.DataFrame()
        for path in [
            pair_input_csv,
            annual_series_csv,
            lag_details_csv,
            lag_summary_csv,
            classification_csv,
            analyst_temporal_csv,
            summary_by_dyad_csv,
            summary_by_dyad_group_csv,
            summary_by_label_csv,
            summary_by_macro_group_csv,
        ]:
            empty.to_csv(path, index=False)
        empty.to_excel(analyst_temporal_xlsx, index=False)
        write_json(
            manifest_path,
            {
                "analyst_csv": str(analyst_csv),
                "annual_counts_csv": str(annual_counts_csv),
                "denominators_csv": str(denominators_csv),
                "row_counts": {
                    "temporal_input_pairs": 0,
                    "annual_series_rows": 0,
                    "classification_rows": 0,
                },
                "outputs": {
                    "pair_input_csv": str(pair_input_csv),
                    "annual_series_csv": str(annual_series_csv),
                    "lag_details_csv": str(lag_details_csv),
                    "lag_summary_csv": str(lag_summary_csv),
                    "classification_csv": str(classification_csv),
                    "analyst_temporal_csv": str(analyst_temporal_csv),
                    "analyst_temporal_xlsx": str(analyst_temporal_xlsx),
                },
            },
        )
        print(args.output_root)
        return

    temporal_pairs = build_temporal_pair_input(analyst)
    annual_series = build_doc_share_series(temporal_pairs, annual_counts, denominators)
    lag_details, lag_summary = build_lag_outputs(annual_series, temporal_pairs, args.min_overlap)

    if temporal_pairs.empty:
        temporal = temporal_pairs.copy()
        summary_by_dyad = pd.DataFrame()
        summary_by_dyad_group = pd.DataFrame()
        summary_by_label = pd.DataFrame()
        summary_by_macro_group = pd.DataFrame()
    else:
        temporal = temporal_pairs.merge(lag_summary, on="pair_id", how="left")
        classified = temporal.apply(
            lambda row: classify_temporal_row(row, args.correlation_floor),
            axis=1,
            result_type="expand",
        )
        temporal["temporal_group"] = classified[0]
        temporal["temporal_reason"] = classified[1]
        temporal["precedence_type_label"] = temporal.apply(
            lambda row: readable_precedence_label(str(row["source_a"]), str(row["source_b"]), str(row["temporal_group"])),
            axis=1,
        )

        summary_by_dyad = build_dyad_summary(temporal)
        summary_by_dyad_group = summarize_grouped(temporal, ["dyad", "dyad_label", "temporal_group"])
        summary_by_label = summarize_grouped(temporal, ["dyad", "dyad_label", "precedence_type_label"])
        summary_by_macro_group = summarize_grouped(temporal, ["macro_topic", "macro_topic_name", "temporal_group"])

    pair_input_csv.parent.mkdir(parents=True, exist_ok=True)
    temporal_pairs.to_csv(pair_input_csv, index=False)
    annual_series.to_csv(annual_series_csv, index=False)
    lag_details.to_csv(lag_details_csv, index=False)
    lag_summary.to_csv(lag_summary_csv, index=False)
    temporal.to_csv(classification_csv, index=False)
    temporal.to_csv(analyst_temporal_csv, index=False)
    summary_by_dyad.to_csv(summary_by_dyad_csv, index=False)
    summary_by_dyad_group.to_csv(summary_by_dyad_group_csv, index=False)
    summary_by_label.to_csv(summary_by_label_csv, index=False)
    summary_by_macro_group.to_csv(summary_by_macro_group_csv, index=False)

    with pd.ExcelWriter(analyst_temporal_xlsx, engine="openpyxl") as writer:
        temporal.to_excel(writer, sheet_name="all_pairs", index=False)
        temporal.loc[temporal["dyad"] == "academic__media"].to_excel(writer, sheet_name="academic_media", index=False)
        temporal.loc[temporal["dyad"] == "academic__corporate"].to_excel(writer, sheet_name="academic_corporate", index=False)
        temporal.loc[temporal["dyad"] == "media__corporate"].to_excel(writer, sheet_name="media_corporate", index=False)
        summary_by_dyad.to_excel(writer, sheet_name="summary_by_dyad", index=False)
        summary_by_dyad_group.to_excel(writer, sheet_name="summary_by_group", index=False)
        summary_by_macro_group.to_excel(writer, sheet_name="summary_by_macro_topic", index=False)

    readme_path = args.output_root / "README_same_issue_discourse_function.md"
    replace_or_append_temporal_section(
        readme_path,
        build_temporal_readme_block(
            pair_count=int(temporal_pairs.shape[0]),
            dyad_summary=summary_by_dyad,
            by_group=summary_by_dyad_group,
            correlation_floor=args.correlation_floor,
            min_overlap=args.min_overlap,
        ),
    )

    write_json(
        manifest_path,
        {
            "analyst_csv": str(analyst_csv),
            "annual_counts_csv": str(annual_counts_csv),
            "denominators_csv": str(denominators_csv),
            "min_overlap": args.min_overlap,
            "correlation_floor": args.correlation_floor,
            "row_counts": {
                "temporal_input_pairs": int(temporal_pairs.shape[0]),
                "annual_series_rows": int(annual_series.shape[0]),
                "lag_detail_rows": int(lag_details.shape[0]),
                "classification_rows": int(temporal.shape[0]),
            },
            "outputs": {
                "pair_input_csv": str(pair_input_csv),
                "annual_series_csv": str(annual_series_csv),
                "lag_details_csv": str(lag_details_csv),
                "lag_summary_csv": str(lag_summary_csv),
                "classification_csv": str(classification_csv),
                "analyst_temporal_csv": str(analyst_temporal_csv),
                "analyst_temporal_xlsx": str(analyst_temporal_xlsx),
                "summary_by_dyad_csv": str(summary_by_dyad_csv),
                "summary_by_dyad_group_csv": str(summary_by_dyad_group_csv),
                "summary_by_label_csv": str(summary_by_label_csv),
                "summary_by_macro_group_csv": str(summary_by_macro_group_csv),
                "readme": str(readme_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
