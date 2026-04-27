#!/usr/bin/env python3
"""Build rolling-window detrended lead-lag analysis between sources."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
INPUT_PATH = PIPELINE_ROOT / "outputs" / "temporal_prep" / "topic_share_within_yes_by_source_year.csv"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "temporal_lead_lag_rolling"
FIGURES_DIR = OUTPUT_DIR / "figures"

TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
ORDERED_PAIRS = [
    ("academic", "media"),
    ("media", "academic"),
    ("academic", "corporate"),
    ("corporate", "academic"),
    ("media", "corporate"),
    ("corporate", "media"),
]
PAIR_LABELS = {pair: f"{pair[0]} -> {pair[1]}" for pair in ORDERED_PAIRS}
METHODS = [
    ("diff_share_within_yes", "First difference"),
    ("linear_detrend_residual", "Linear detrend residual"),
]
WINDOW_SIZE = 10
WINDOW_STEP = 1
LAG_MIN = -2
LAG_MAX = 2
MIN_OVERLAP = 4


def load_series() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["source", "year", "assigned_label", "share_within_yes"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["source", "year", "assigned_label", "topic_name", "share_within_yes"]].copy()


def add_window_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["assigned_label", "source", "year"]).copy()
    df["diff_share_within_yes"] = (
        df.groupby(["assigned_label", "source"], dropna=False)["share_within_yes"].diff()
    )

    def detrend(group: pd.DataFrame) -> pd.DataFrame:
        years = group["year"].to_numpy(dtype=float)
        values = group["share_within_yes"].to_numpy(dtype=float)
        if len(group) < 2 or np.allclose(values, values[0]):
            group["linear_detrend_residual"] = values - values.mean()
            return group
        slope, intercept = np.polyfit(years, values, deg=1)
        trend = slope * years + intercept
        group["linear_detrend_residual"] = values - trend
        return group

    groups = [
        detrend(group.copy())
        for _, group in df.groupby(["assigned_label", "source"], dropna=False, sort=False)
    ]
    return pd.concat(groups, ignore_index=True)


def correlation_for_lag(
    x_series: pd.DataFrame,
    y_series: pd.DataFrame,
    value_col: str,
    lag: int,
) -> tuple[float | None, int]:
    x = x_series[["year", value_col]].rename(columns={value_col: "x"})
    y = y_series[["year", value_col]].rename(columns={value_col: "y"})
    y = y.assign(year=y["year"] - lag)
    aligned = x.merge(y, on="year", how="inner").dropna()
    n = len(aligned)
    if n < MIN_OVERLAP:
        return None, n
    if aligned["x"].nunique() <= 1 or aligned["y"].nunique() <= 1:
        return None, n
    return float(np.corrcoef(aligned["x"], aligned["y"])[0, 1]), n


def build_windows(years: pd.Series) -> list[tuple[int, int]]:
    min_year = int(years.min())
    max_year = int(years.max())
    windows = []
    for start in range(min_year, max_year - WINDOW_SIZE + 2, WINDOW_STEP):
        end = start + WINDOW_SIZE - 1
        windows.append((start, end))
    return windows


def build_rolling_correlations(series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows: list[dict[str, object]] = []
    input_rows: list[pd.DataFrame] = []

    for window_start, window_end in build_windows(series["year"]):
        window_label = f"{window_start}-{window_end}"
        window_df = series.loc[(series["year"] >= window_start) & (series["year"] <= window_end)].copy()
        window_df = add_window_transforms(window_df)
        window_df["window_start"] = window_start
        window_df["window_end"] = window_end
        window_df["window_label"] = window_label
        input_rows.append(window_df)

        for topic, topic_group in window_df.groupby("assigned_label", dropna=False):
            topic_name = topic_group["topic_name"].dropna().iloc[0]
            for source_from, source_to in ORDERED_PAIRS:
                x_series = topic_group.loc[topic_group["source"] == source_from].sort_values("year")
                y_series = topic_group.loc[topic_group["source"] == source_to].sort_values("year")
                for value_col, method_name in METHODS:
                    for lag in range(LAG_MIN, LAG_MAX + 1):
                        corr, n_overlap = correlation_for_lag(x_series, y_series, value_col, lag)
                        corr_rows.append(
                            {
                                "window_start": window_start,
                                "window_end": window_end,
                                "window_label": window_label,
                                "assigned_label": topic,
                                "topic_name": topic_name,
                                "source_from": source_from,
                                "source_to": source_to,
                                "pair_label": PAIR_LABELS[(source_from, source_to)],
                                "transform_method": value_col,
                                "transform_label": method_name,
                                "lag": lag,
                                "correlation": corr,
                                "abs_correlation": abs(corr) if corr is not None else np.nan,
                                "n_overlap_years": n_overlap,
                                "lag_sign_convention": "positive lag means source_from leads source_to by that many years",
                            }
                        )

    return pd.concat(input_rows, ignore_index=True), pd.DataFrame(corr_rows)


def build_summary(all_corr: pd.DataFrame) -> pd.DataFrame:
    valid = all_corr.dropna(subset=["correlation"]).copy()
    valid["abs_correlation_rank"] = (
        valid.groupby(
            ["window_label", "assigned_label", "source_from", "source_to", "transform_method"],
            dropna=False,
        )["abs_correlation"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    valid["is_strongest_for_pair"] = valid["abs_correlation_rank"] == 1
    return (
        valid.loc[valid["is_strongest_for_pair"]]
        .sort_values(["assigned_label", "window_start", "transform_method", "source_from", "source_to"])
        .reset_index(drop=True)
    )


def build_heatmap_input(summary: pd.DataFrame) -> pd.DataFrame:
    heatmap = summary.copy()
    heatmap["cell_label"] = heatmap["lag"].apply(lambda x: f"{int(x):+d}")
    return heatmap[
        [
            "window_start",
            "window_end",
            "window_label",
            "assigned_label",
            "topic_name",
            "source_from",
            "source_to",
            "pair_label",
            "transform_method",
            "transform_label",
            "lag",
            "cell_label",
            "correlation",
            "abs_correlation",
            "n_overlap_years",
        ]
    ].copy()


def write_descriptions() -> Path:
    path = OUTPUT_DIR / "figure_descriptions.md"
    text = f"""# Rolling Lead-Lag Figure Descriptions

## Input metric
The analysis uses annual topic shares within topic-positive documents. For each source-year-topic, the base series equals the number of documents containing the topic divided by all topic-positive documents in that source-year.

## Rolling windows
The series are analyzed in rolling {WINDOW_SIZE}-year windows that move forward one year at a time. This allows the dominant lead-lag relationship to change over time instead of forcing one result for the full 25-year period.

## Detrending transformations
Two transformed series are analyzed in each window. The first is the first difference of the annual share. The second is the residual from a linear detrending regression fitted separately within each source-topic window.

## Lag convention
Positive lag means that the `source_from` series leads the `source_to` series by that many years. In this rolling version, lags are evaluated only within ±2 years.

## Topic heatmaps
Each topic figure contains two heatmaps, one per detrending method. Rows are ordered source pairs and columns are rolling windows. Cell color shows the strongest correlation observed in that window and the annotation shows the winning lag. This makes it possible to see whether relationships between sources shift or reverse over time.
"""
    path.write_text(text, encoding="utf-8")
    return path


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def plot_topic_window_heatmaps(heatmap_input: pd.DataFrame) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    pair_order = [PAIR_LABELS[p] for p in ORDERED_PAIRS]
    methods = [m[0] for m in METHODS]
    method_labels = dict(METHODS)
    note = (
        f"Base calculation: Annual series are topic shares within topic-positive documents. "
        f"Each panel shows the strongest lagged correlation within a {WINDOW_SIZE}-year moving window. "
        f"Annotation = winning lag; positive lag means source_from leads source_to."
    )

    for topic in TOPIC_ORDER:
        topic_df = heatmap_input.loc[heatmap_input["assigned_label"] == topic].copy()
        topic_name = topic_df["topic_name"].dropna().iloc[0]
        windows = (
            topic_df[["window_start", "window_label"]]
            .drop_duplicates()
            .sort_values("window_start")["window_label"]
            .tolist()
        )
        fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
        vmin, vmax = -1.0, 1.0

        for ax, method in zip(axes, methods):
            subset = topic_df.loc[topic_df["transform_method"] == method]
            matrix = (
                subset.pivot(index="pair_label", columns="window_label", values="correlation")
                .reindex(index=pair_order, columns=windows)
            )
            annot = (
                subset.pivot(index="pair_label", columns="window_label", values="cell_label")
                .reindex(index=pair_order, columns=windows)
            )
            im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.set_yticks(range(len(pair_order)))
            ax.set_yticklabels(pair_order)
            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels(windows, rotation=35, ha="right")
            ax.set_title(method_labels[method])
            for i in range(len(pair_order)):
                for j in range(len(windows)):
                    label = annot.iloc[i, j]
                    if pd.notna(label):
                        ax.text(j, i, str(label), ha="center", va="center", fontsize=8, color="black")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Correlation")

        fig.suptitle(f"{topic} {topic_name}: rolling lead-lag summary", fontsize=15, y=1.02)
        fig.text(
            0.01,
            0.01,
            note,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        outputs[f"rolling_heatmap_{topic.lower()}"] = save_figure(fig, f"rolling_heatmap_{topic.lower()}")

    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    series = load_series()
    rolling_series, rolling_corr = build_rolling_correlations(series)
    summary = build_summary(rolling_corr)
    heatmap_input = build_heatmap_input(summary)

    series_path = OUTPUT_DIR / "rolling_lead_lag_series_input.csv"
    corr_path = OUTPUT_DIR / "rolling_lead_lag_correlations_all.csv"
    summary_path = OUTPUT_DIR / "rolling_lead_lag_summary_strongest.csv"
    heatmap_path = OUTPUT_DIR / "rolling_lead_lag_heatmap_input.csv"

    rolling_series.to_csv(series_path, index=False)
    rolling_corr.to_csv(corr_path, index=False)
    summary.to_csv(summary_path, index=False)
    heatmap_input.to_csv(heatmap_path, index=False)

    figure_outputs = plot_topic_window_heatmaps(heatmap_input)
    descriptions_path = write_descriptions()

    manifest = {
        "input_path": str(INPUT_PATH),
        "window_size_years": WINDOW_SIZE,
        "window_step_years": WINDOW_STEP,
        "lag_window": [LAG_MIN, LAG_MAX],
        "min_overlap_years": MIN_OVERLAP,
        "series_rows": int(len(rolling_series)),
        "correlation_rows": int(len(rolling_corr)),
        "summary_rows": int(len(summary)),
        "lag_sign_convention": "positive lag means source_from leads source_to by that many years",
        "outputs": {
            "rolling_lead_lag_series_input": str(series_path),
            "rolling_lead_lag_correlations_all": str(corr_path),
            "rolling_lead_lag_summary_strongest": str(summary_path),
            "rolling_lead_lag_heatmap_input": str(heatmap_path),
            "figure_descriptions": str(descriptions_path),
            "figures": figure_outputs,
        },
    }
    (OUTPUT_DIR / "rolling_lead_lag_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
