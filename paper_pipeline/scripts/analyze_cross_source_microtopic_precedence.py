#!/usr/bin/env python3
"""Analyze restrained lagged precedence for viable cross-source microtopic pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import OUTPUT_ROOT, SIMILARITY_THRESHOLDS, read_json, threshold_suffix, write_json


LAGS = [-2, -1, 0, 1, 2]
MIN_OVERLAP = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--viable-pairs-csv", type=Path, default=None)
    parser.add_argument("--series-csv", type=Path, default=None)
    parser.add_argument("--descriptive-summary-csv", type=Path, default=None)
    parser.add_argument("--matching-summary-csv", type=Path, default=None)
    parser.add_argument("--main-threshold", type=float, default=0.75)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def build_metric_table(
    pairs: pd.DataFrame,
    series: pd.DataFrame,
    col_a: str,
    col_b: str,
    metric_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag_rows: list[dict] = []
    summary_rows: list[dict] = []
    for pair in pairs.itertuples(index=False):
        pair_series = series.loc[series["pair_id"] == pair.pair_id].copy().sort_values("year")
        metric_rows = []
        for lag in LAGS:
            corr, overlap = correlation_for_lag(pair_series, col_a, col_b, lag)
            metric_row = {
                "pair_id": str(pair.pair_id),
                "macro_topic": str(pair.macro_topic),
                "macro_topic_name": str(pair.macro_topic_name),
                "dyad": str(pair.dyad),
                "dyad_label": str(pair.dyad_label),
                "source_a": str(pair.source_a),
                "microtopic_id_a": int(pair.microtopic_id_a),
                "source_b": str(pair.source_b),
                "microtopic_id_b": int(pair.microtopic_id_b),
                "metric": metric_label,
                "lag": int(lag),
                "correlation": corr,
                "abs_correlation": abs(corr) if corr is not None else np.nan,
                "n_overlap_years": int(overlap),
                "lag_sign_convention": "positive lag means source_a leads source_b by that many years",
            }
            lag_rows.append(metric_row)
            metric_rows.append(metric_row)

        metric_df = pd.DataFrame(metric_rows)
        valid = metric_df.dropna(subset=["correlation"]).copy()
        if valid.empty:
            summary_rows.append(
                {
                    "pair_id": str(pair.pair_id),
                    f"best_lag_{metric_label}": np.nan,
                    f"best_correlation_{metric_label}": np.nan,
                    f"best_overlap_{metric_label}": 0,
                    f"low_information_flag_{metric_label}": True,
                    f"unstable_series_flag_{metric_label}": True,
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
                f"best_lag_{metric_label}": int(best["lag"]),
                f"best_correlation_{metric_label}": float(best["correlation"]),
                f"best_overlap_{metric_label}": int(best["n_overlap_years"]),
                f"low_information_flag_{metric_label}": bool(int(best["n_overlap_years"]) < MIN_OVERLAP),
                f"unstable_series_flag_{metric_label}": bool(unstable),
            }
        )

    return pd.DataFrame(lag_rows), pd.DataFrame(summary_rows)


def build_readme(
    output_root: Path,
    descriptive: pd.DataFrame,
    lag_summary: pd.DataFrame,
    matching_summary: pd.DataFrame,
    main_threshold: float,
) -> None:
    def pick_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    retained_counts: dict[float, int] = {}
    for threshold in SIMILARITY_THRESHOLDS:
        column = f"retained_count_threshold_{threshold_suffix(threshold)}"
        retained_counts[threshold] = (
            int(matching_summary[column].sum())
            if not matching_summary.empty and column in matching_summary.columns
            else 0
        )
    lenient = (
        int(descriptive["lenient_is_viable"].fillna(False).sum())
        if not descriptive.empty and "lenient_is_viable" in descriptive.columns
        else 0
    )
    balanced = (
        int(descriptive["balanced_is_viable"].fillna(False).sum())
        if not descriptive.empty and "balanced_is_viable" in descriptive.columns
        else 0
    )
    strict = (
        int(descriptive["strict_is_viable"].fillna(False).sum())
        if not descriptive.empty and "strict_is_viable" in descriptive.columns
        else 0
    )

    if not descriptive.empty and {"macro_topic", "macro_topic_name", "balanced_is_viable"}.issubset(
        descriptive.columns
    ):
        macro_summary = (
            descriptive.groupby(["macro_topic", "macro_topic_name"], dropna=False)["balanced_is_viable"]
            .sum()
            .reset_index()
            .sort_values("balanced_is_viable", ascending=False)
        )
    else:
        macro_summary = pd.DataFrame(columns=["macro_topic", "macro_topic_name", "balanced_is_viable"])
    top_macro_lines = [
        f"- `{row.macro_topic}` {row.macro_topic_name}: {int(row.balanced_is_viable)} balanced viable pairs"
        for row in macro_summary.head(5).itertuples(index=False)
    ]

    merged = descriptive.merge(lag_summary, on="pair_id", how="left", suffixes=("_desc", "_lag")) if not descriptive.empty else pd.DataFrame()
    balanced_col = "balanced_is_viable"
    if balanced_col not in merged.columns:
        if "balanced_is_viable_desc" in merged.columns:
            balanced_col = "balanced_is_viable_desc"
        elif "balanced_is_viable_lag" in merged.columns:
            balanced_col = "balanced_is_viable_lag"
    cosine_col = pick_col(merged, ["cosine_similarity", "cosine_similarity_desc", "cosine_similarity_lag"])
    best_corr_col = pick_col(merged, ["best_correlation_docs"])
    strongest = (
        merged.loc[merged[balanced_col].fillna(False)]
        .sort_values([cosine_col, best_corr_col], ascending=[False, False])
        .head(10)
        if not merged.empty and cosine_col and best_corr_col
        else pd.DataFrame()
    )
    if strongest.empty:
        fallback_path = output_root / "retained_pairs_threshold_070.csv"
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            fallback = pd.read_csv(fallback_path)
            strongest = fallback.sort_values("cosine_similarity", ascending=False).head(10)
    strongest_lines = []
    for row in strongest.itertuples(index=False):
        similarity_value = getattr(row, "cosine_similarity", getattr(row, "cosine_similarity_desc", np.nan))
        if hasattr(row, "best_lag_docs") and pd.notna(getattr(row, "best_lag_docs", np.nan)) and pd.notna(
            getattr(row, "best_correlation_docs", np.nan)
        ):
            strongest_lines.append(
                f"- `{row.pair_id}`: similarity `{similarity_value:.3f}`, "
                f"best doc lag `{int(row.best_lag_docs):+d}` corr `{row.best_correlation_docs:.3f}`"
            )
        else:
            strongest_lines.append(
                f"- `{row.pair_id}`: similarity `{similarity_value:.3f}`"
            )

    sensitivity_lines: list[str] = []
    if retained_counts.get(main_threshold, 0) == 0:
        profiles_path = output_root / "microtopic_profiles.csv"
        retained_070_path = output_root / "retained_pairs_threshold_070.csv"
        if profiles_path.exists() and retained_070_path.exists():
            try:
                profiles = pd.read_csv(profiles_path)
                retained_070_pairs = pd.read_csv(retained_070_path)
                if not profiles.empty and not retained_070_pairs.empty:
                    profiles["microtopic_id"] = pd.to_numeric(profiles["microtopic_id"], errors="coerce").astype(int)
                    profile_map = {
                        (str(row.subgroup), int(row.microtopic_id)): row._asdict()
                        for row in profiles.itertuples(index=False)
                    }
                    counts = {"lenient": 0, "balanced": 0, "strict": 0}
                    for row in retained_070_pairs.itertuples(index=False):
                        a = profile_map[(str(row.subgroup_a), int(row.microtopic_id_a))]
                        b = profile_map[(str(row.subgroup_b), int(row.microtopic_id_b))]
                        years_a = set(json.loads(a["years_present"]))
                        years_b = set(json.loads(b["years_present"]))
                        overlap = len(years_a & years_b)
                        for rule_name, rule in {
                            "lenient": (3, 3, 5, 10),
                            "balanced": (4, 4, 10, 20),
                            "strict": (5, 5, 20, 40),
                        }.items():
                            min_years, min_overlap, min_docs, min_chunks = rule
                            if (
                                int(a["active_year_count"]) >= min_years
                                and int(b["active_year_count"]) >= min_years
                                and overlap >= min_overlap
                                and int(a["document_count"]) >= min_docs
                                and int(b["document_count"]) >= min_docs
                                and int(a["chunk_count"]) >= min_chunks
                                and int(b["chunk_count"]) >= min_chunks
                            ):
                                counts[rule_name] += 1
                    sensitivity_lines = [
                        "## Sensitivity Note",
                        f"- At the looser `0.70` similarity threshold, there are `{len(retained_070_pairs)}` retained semantic pairs.",
                        f"- Of those, `{counts['lenient']}` would satisfy the lenient temporal rule, `{counts['balanced']}` the balanced rule, and `{counts['strict']}` the strict rule.",
                        "- This suggests the cross-source microtopic layer has usable signal, but the requested `0.75` main threshold is too strict for this corpus.",
                        "",
                    ]
            except Exception:
                sensitivity_lines = []

    if balanced > 0:
        promise_text = (
            "This layer looks more analytically promising than the earlier broad topic-level lead-lag view because it "
            "operates on semantically tighter matched units rather than internally heterogeneous macro-topics. The "
            "remaining limitation is that viable pairs are still unevenly distributed across macro-topics and corporate "
            "series remain sparse in some areas."
        )
    else:
        promise_text = (
            "This layer is conceptually better aligned with the lead-lag question than the broad topic-level view, "
            "but the current viable-pair count is too limited to claim a clear empirical improvement yet."
        )

    text = "\n".join(
        [
            "# Cross-Source Matched Microtopic Temporal Analysis",
            "",
            "## What was done",
            "We built a deterministic cross-source matching layer on top of the subgroup-specific unsupervised BERTopic models. Matching was restricted to the same macro-topic, used profile-text embeddings rather than LLM outputs, retained mutual nearest-neighbor pairs, and then evaluated temporal viability and restrained lagged precedence.",
            "",
            "## Thresholds used",
            "- similarity thresholds: "
            + ", ".join(
                [
                    f"`{threshold:.2f}`{' (main)' if abs(threshold - main_threshold) < 1e-9 else ''}"
                    for threshold in SIMILARITY_THRESHOLDS
                ]
            ),
            "- temporal viability rules:",
            "  - `lenient`: at least 3 active years per side, at least 3 overlapping years, at least 5 docs and 10 chunks per side",
            "  - `balanced`: at least 4 active years per side, at least 4 overlapping years, at least 10 docs and 20 chunks per side",
            "  - `strict`: at least 5 active years per side, at least 5 overlapping years, at least 20 docs and 40 chunks per side",
            "- lag set: `-2, -1, 0, +1, +2`",
            "",
            "## Pair counts",
            *[
                f"- retained semantic pairs at `{threshold:.2f}`: `{retained_counts.get(threshold, 0)}`"
                for threshold in SIMILARITY_THRESHOLDS
            ],
            f"- temporally viable pairs, `lenient`: `{lenient}`",
            f"- temporally viable pairs, `balanced`: `{balanced}`",
            f"- temporally viable pairs, `strict`: `{strict}`",
            "",
            *sensitivity_lines,
            "## Macro-topics with the most balanced viable pairs",
            *(top_macro_lines or ["- None found"]),
            "",
            "## Strongest candidate pairs",
            *(strongest_lines or ["- None found"]),
            "",
            "## Warnings and limitations",
            "- Matching is deterministic and reproducible, but MNN is intentionally strict and may exclude some plausible near-neighbor alternatives.",
            "- Temporal precedence results indicate predictive association only; they are not causal claims.",
            "- Sparse corporate series remain an important constraint, especially in smaller macro-topics.",
            "- Low-variance or short-overlap pairs are flagged separately and should be interpreted cautiously.",
            "",
            "## Readout",
            promise_text,
            "",
        ]
    )
    (output_root / "README_microtopic_cross_source_pairs.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    viable_pairs_csv = args.viable_pairs_csv or (args.output_root / "temporally_viable_pairs_balanced.csv")
    series_csv = args.series_csv or (args.output_root / "pair_annual_series_combined.csv")
    descriptive_csv = args.descriptive_summary_csv or (args.output_root / "pair_descriptive_summary.csv")
    matching_summary_csv = args.matching_summary_csv or (args.output_root / "matching_summary_by_macro_topic_dyad.csv")

    viable_pairs = safe_read_csv(viable_pairs_csv)
    series = safe_read_csv(series_csv)
    descriptive = safe_read_csv(descriptive_csv)
    matching_summary = safe_read_csv(matching_summary_csv)

    if viable_pairs.empty:
        docs_path = args.output_root / "pair_lag_analysis_docs.csv"
        chunks_path = args.output_root / "pair_lag_analysis_chunks.csv"
        summary_path = args.output_root / "pair_lag_analysis_summary.csv"
        pd.DataFrame().to_csv(docs_path, index=False)
        pd.DataFrame().to_csv(chunks_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        build_readme(
            output_root=args.output_root,
            descriptive=descriptive,
            lag_summary=pd.DataFrame(),
            matching_summary=matching_summary,
            main_threshold=args.main_threshold,
        )
        manifests = {}
        for name in [
            "profile_manifest.json",
            "embedding_manifest.json",
            "matching_manifest.json",
            "temporal_filter_manifest.json",
            "series_manifest.json",
            "descriptive_manifest.json",
        ]:
            path = args.output_root / name
            if path.exists():
                manifests[name] = read_json(path)
        write_json(
            args.output_root / "cross_source_pairs_manifest.json",
            {
                "default_precedence_rule": "balanced",
                "main_threshold": args.main_threshold,
                "lag_set": LAGS,
                "min_overlap_for_lag": MIN_OVERLAP,
                "row_counts": {
                    "viable_pairs_for_lag": 0,
                    "pair_lag_analysis_docs": 0,
                    "pair_lag_analysis_chunks": 0,
                    "pair_lag_analysis_summary": 0,
                },
                "upstream_manifests": manifests,
                "outputs": {
                    "pair_lag_analysis_docs": str(docs_path),
                    "pair_lag_analysis_chunks": str(chunks_path),
                    "pair_lag_analysis_summary": str(summary_path),
                    "readme": str(args.output_root / "README_microtopic_cross_source_pairs.md"),
                },
            },
        )
        print(args.output_root)
        return

    docs_lag, docs_summary = build_metric_table(
        pairs=viable_pairs,
        series=series,
        col_a="source_a_share_docs",
        col_b="source_b_share_docs",
        metric_label="docs",
    )
    chunks_lag, chunks_summary = build_metric_table(
        pairs=viable_pairs,
        series=series,
        col_a="source_a_share_chunks",
        col_b="source_b_share_chunks",
        metric_label="chunks",
    )

    lag_summary = viable_pairs.merge(docs_summary, on="pair_id", how="left").merge(
        chunks_summary, on="pair_id", how="left"
    )

    docs_path = args.output_root / "pair_lag_analysis_docs.csv"
    chunks_path = args.output_root / "pair_lag_analysis_chunks.csv"
    summary_path = args.output_root / "pair_lag_analysis_summary.csv"
    docs_lag.to_csv(docs_path, index=False)
    chunks_lag.to_csv(chunks_path, index=False)
    lag_summary.to_csv(summary_path, index=False)

    build_readme(
        output_root=args.output_root,
        descriptive=descriptive,
        lag_summary=lag_summary,
        matching_summary=matching_summary,
        main_threshold=args.main_threshold,
    )

    manifests = {}
    for name in [
        "profile_manifest.json",
        "embedding_manifest.json",
        "matching_manifest.json",
        "temporal_filter_manifest.json",
        "series_manifest.json",
        "descriptive_manifest.json",
    ]:
        path = args.output_root / name
        if path.exists():
            manifests[name] = read_json(path)

    write_json(
        args.output_root / "cross_source_pairs_manifest.json",
        {
            "default_precedence_rule": "balanced",
            "main_threshold": args.main_threshold,
            "lag_set": LAGS,
            "min_overlap_for_lag": MIN_OVERLAP,
            "row_counts": {
                "viable_pairs_for_lag": int(viable_pairs.shape[0]),
                "pair_lag_analysis_docs": int(docs_lag.shape[0]),
                "pair_lag_analysis_chunks": int(chunks_lag.shape[0]),
                "pair_lag_analysis_summary": int(lag_summary.shape[0]),
            },
            "upstream_manifests": manifests,
            "outputs": {
                "pair_lag_analysis_docs": str(docs_path),
                "pair_lag_analysis_chunks": str(chunks_path),
                "pair_lag_analysis_summary": str(summary_path),
                "readme": str(args.output_root / "README_microtopic_cross_source_pairs.md"),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
