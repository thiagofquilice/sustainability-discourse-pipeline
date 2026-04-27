#!/usr/bin/env python3
"""Build detrended lead-lag analysis between sources for the 6-topic pipeline."""

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
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "temporal_lead_lag"
FIGURES_DIR = OUTPUT_DIR / "figures"

TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
SOURCE_ORDER = ["academic", "media", "corporate"]
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
METHOD_COLORS = {
    "diff_share_within_yes": "#1f77b4",
    "linear_detrend_residual": "#d62728",
}
LAG_MIN = -5
LAG_MAX = 5
MIN_OVERLAP = 3


def load_series() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["source", "year", "assigned_label", "share_within_yes"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["source", "year", "assigned_label", "topic_name", "share_within_yes"]].copy()


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
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
    # Positive lag means source_from leads source_to by `lag` years:
    # corr(x_t, y_{t+lag}) -> shift y years backward by lag for alignment on t.
    y = y.assign(year=y["year"] - lag)
    aligned = x.merge(y, on="year", how="inner").dropna()
    n = len(aligned)
    if n < MIN_OVERLAP:
        return None, n
    if aligned["x"].nunique() <= 1 or aligned["y"].nunique() <= 1:
        return None, n
    corr = float(np.corrcoef(aligned["x"], aligned["y"])[0, 1])
    return corr, n


def build_correlations(series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topic, topic_group in series.groupby("assigned_label", dropna=False):
        topic_name = topic_group["topic_name"].dropna().iloc[0]
        for source_from, source_to in ORDERED_PAIRS:
            x_series = topic_group.loc[topic_group["source"] == source_from].sort_values("year")
            y_series = topic_group.loc[topic_group["source"] == source_to].sort_values("year")
            for value_col, method_name in METHODS:
                for lag in range(LAG_MIN, LAG_MAX + 1):
                    corr, n_overlap = correlation_for_lag(x_series, y_series, value_col, lag)
                    rows.append(
                        {
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
    return pd.DataFrame(rows)


def build_summary(all_corr: pd.DataFrame) -> pd.DataFrame:
    valid = all_corr.dropna(subset=["correlation"]).copy()
    valid["abs_correlation_rank"] = (
        valid.groupby(["assigned_label", "source_from", "source_to", "transform_method"], dropna=False)["abs_correlation"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    valid["is_strongest_for_pair"] = valid["abs_correlation_rank"] == 1
    summary = (
        valid.loc[valid["is_strongest_for_pair"]]
        .sort_values(["assigned_label", "transform_method", "source_from", "source_to"])
        .reset_index(drop=True)
    )
    return summary


def build_heatmap_input(summary: pd.DataFrame) -> pd.DataFrame:
    heatmap = summary.copy()
    heatmap["cell_label"] = heatmap["lag"].apply(lambda x: f"lag {int(x):+d}")
    return heatmap[
        [
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
    text = """# Lead-Lag Figure Descriptions

## Input metric
The analysis uses annual topic shares within topic-positive documents. For each source-year-topic, the base series equals the number of documents containing the topic divided by all topic-positive documents in that source-year.

## Detrending transformations
Two transformed series are analyzed for each source-topic combination. The first is the first difference of the annual share. The second is the residual from a linear detrending regression fitted separately to each source-topic annual share series.

## Lag convention
Positive lag means that the `source_from` series leads the `source_to` series by that many years. A positive dominant lag therefore indicates that changes in the `source_from` series tend to be followed by changes in the `source_to` series later.

## CCF topic figures
Each topic figure contains six panels, one per ordered source pair. The plotted values are lagged correlations across annual detrended topic-share series over lags from -5 to +5 years. Both detrending methods are shown in each panel.

## Heatmap summary
The heatmap summarizes, for each topic and ordered source pair, the strongest lagged correlation observed within ±5 years. Cell color shows the correlation value and the annotation shows the winning lag.
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


def plot_ccf_by_topic(all_corr: pd.DataFrame) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    note = (
        "Base calculation: Annual series are topic shares within topic-positive documents. "
        "Blue = first difference; red = linear detrend residual. "
        "Positive lag means source_from leads source_to by that many years."
    )
    for topic in TOPIC_ORDER:
        topic_df = all_corr.loc[all_corr["assigned_label"] == topic].copy()
        topic_name = topic_df["topic_name"].dropna().iloc[0]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, pair in zip(axes, ORDERED_PAIRS):
            pair_df = topic_df.loc[
                (topic_df["source_from"] == pair[0]) & (topic_df["source_to"] == pair[1])
            ].sort_values("lag")
            for method, method_label in METHODS:
                series = pair_df.loc[pair_df["transform_method"] == method]
                ax.plot(
                    series["lag"],
                    series["correlation"],
                    marker="o",
                    linewidth=2,
                    color=METHOD_COLORS[method],
                    label=method_label,
                )
            ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
            ax.axvline(0, color="#cccccc", linestyle=":", linewidth=1)
            ax.set_title(PAIR_LABELS[pair])
            ax.set_xlabel("Lag (years)")
            ax.set_ylabel("Correlation")
            ax.set_xticks(range(LAG_MIN, LAG_MAX + 1, 1))
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
        fig.suptitle(f"{topic} {topic_name}: detrended lead-lag correlations", fontsize=15, y=1.02)
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
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        outputs[f"ccf_{topic.lower()}"] = save_figure(fig, f"ccf_{topic.lower()}")
    return outputs


def plot_heatmap_summary(heatmap_input: pd.DataFrame) -> list[str]:
    methods = [m[0] for m in METHODS]
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    vmin, vmax = -1.0, 1.0
    pair_order = [PAIR_LABELS[p] for p in ORDERED_PAIRS]
    note = (
        "Base calculation: Each cell reports the strongest lagged correlation within ±5 years between annual detrended "
        "topic shares within topic-positive documents. Cell annotation shows the winning lag. Positive lag means "
        "source_from leads source_to."
    )

    for ax, method in zip(axes, methods):
        subset = heatmap_input.loc[heatmap_input["transform_method"] == method].copy()
        matrix = (
            subset.pivot(index="assigned_label", columns="pair_label", values="correlation")
            .reindex(index=TOPIC_ORDER, columns=pair_order)
        )
        annot = (
            subset.pivot(index="assigned_label", columns="pair_label", values="cell_label")
            .reindex(index=TOPIC_ORDER, columns=pair_order)
        )
        im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels(pair_order, rotation=30, ha="right")
        ax.set_yticks(range(len(TOPIC_ORDER)))
        ax.set_yticklabels(TOPIC_ORDER)
        method_label = dict(METHODS)[method]
        ax.set_title(method_label)
        for i in range(len(TOPIC_ORDER)):
            for j in range(len(pair_order)):
                label = annot.iloc[i, j]
                if pd.notna(label):
                    ax.text(j, i, str(label), ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Correlation")

    fig.suptitle("Strongest lead-lag correlations by topic and ordered source pair", fontsize=15, y=1.02)
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
    return save_figure(fig, "lead_lag_heatmap_summary")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    series = add_transforms(load_series())
    all_corr = build_correlations(series)
    summary = build_summary(all_corr)
    heatmap_input = build_heatmap_input(summary)

    series_path = OUTPUT_DIR / "lead_lag_series_input.csv"
    corr_path = OUTPUT_DIR / "lead_lag_correlations_all.csv"
    summary_path = OUTPUT_DIR / "lead_lag_summary_strongest.csv"
    heatmap_path = OUTPUT_DIR / "lead_lag_heatmap_input.csv"

    series.to_csv(series_path, index=False)
    all_corr.to_csv(corr_path, index=False)
    summary.to_csv(summary_path, index=False)
    heatmap_input.to_csv(heatmap_path, index=False)

    ccf_outputs = plot_ccf_by_topic(all_corr)
    heatmap_outputs = plot_heatmap_summary(heatmap_input)
    descriptions_path = write_descriptions()

    manifest = {
        "input_path": str(INPUT_PATH),
        "series_rows": int(len(series)),
        "correlation_rows": int(len(all_corr)),
        "summary_rows": int(len(summary)),
        "lag_window": [LAG_MIN, LAG_MAX],
        "min_overlap_years": MIN_OVERLAP,
        "lag_sign_convention": "positive lag means source_from leads source_to by that many years",
        "transform_methods": [m[0] for m in METHODS],
        "ordered_pairs": [PAIR_LABELS[p] for p in ORDERED_PAIRS],
        "outputs": {
            "lead_lag_series_input": str(series_path),
            "lead_lag_correlations_all": str(corr_path),
            "lead_lag_summary_strongest": str(summary_path),
            "lead_lag_heatmap_input": str(heatmap_path),
            "ccf_figures": ccf_outputs,
            "heatmap_figure": heatmap_outputs,
            "figure_descriptions": str(descriptions_path),
        },
    }
    (OUTPUT_DIR / "lead_lag_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
