#!/usr/bin/env python3
"""Render PNG figure suggestions for the six-macro-topic results package."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_RESULTS_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_six_macro_results"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
STATUS_ORDER = ["aligned", "external_relevant_unpaired", "excluded"]
STATUS_LABELS = {
    "aligned": "Aligned",
    "external_relevant_unpaired": "Relevant without pair",
    "excluded": "Excluded",
}
STATUS_COLORS = {
    "aligned": "#2A7F62",
    "external_relevant_unpaired": "#E6A532",
    "excluded": "#C95C54",
}
PRECEDENCE_LABELS = {
    "academic_leads_corporate": "Academic leads",
    "media_leads_corporate": "Media leads",
    "corporate_leads_academic": "Corporate leads academic",
    "corporate_leads_media": "Corporate leads media",
    "synchronous_or_unclear": "Synchronous/unclear",
}
PRECEDENCE_ORDER = [
    "academic_leads_corporate",
    "media_leads_corporate",
    "corporate_leads_academic",
    "corporate_leads_media",
    "synchronous_or_unclear",
]
PRECEDENCE_COLORS = {
    "academic_leads_corporate": "#4472C4",
    "media_leads_corporate": "#8E63CE",
    "corporate_leads_academic": "#2CA58D",
    "corporate_leads_media": "#6CBF43",
    "synchronous_or_unclear": "#D4A72C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def nice_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D8D8D8",
            "grid.linestyle": ":",
            "axes.edgecolor": "#C8C8C8",
        }
    )


def wrap_macro_label(value: str) -> str:
    mapping = {
        "T1": "T1  Clean energy\ntransition",
        "T2": "T2  Operational sustainability\nand circular production",
        "T3": "T3  Sustainable products,\nservices and consumption",
        "T4": "T4  Climate strategy,\ncarbon governance and disclosure",
        "T5": "T5  Climate risk,\nadaptation and resilience",
        "T6": "T6  Ecosystems, pollution\nand stewardship",
    }
    return mapping.get(value, value)


def render_coverage(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "figure_01_macro_topic_coverage.csv")
    pivot = (
        df.assign(macro_topic=pd.Categorical(df["macro_topic"], categories=MACRO_TOPIC_ORDER, ordered=True))
        .pivot(index="macro_topic", columns="paper_status", values="topic_count")
        .fillna(0)
        .reindex(MACRO_TOPIC_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    left = pd.Series(0, index=pivot.index, dtype=float)
    for status in STATUS_ORDER:
        widths = pivot.get(status, pd.Series(0, index=pivot.index))
        ax.barh(
            [wrap_macro_label(v) for v in pivot.index],
            widths,
            left=left,
            color=STATUS_COLORS[status],
            edgecolor="white",
            linewidth=1.2,
            label=STATUS_LABELS[status],
        )
        for idx, value in enumerate(widths):
            if value > 0:
                ax.text(left.iloc[idx] + value / 2, idx, f"{int(value)}", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
        left = left + widths

    ax.set_title("Corporate-focus coverage and blind spots by macro topic")
    ax.set_xlabel("Non-corporate topics after post-comment curation")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "figure_01_macro_topic_coverage.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_temporal_summary(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "figure_02_macro_topic_temporal_summary.csv")
    df["combo"] = df["dyad"].map(
        {
            "academic__corporate": "Academic ↔ Corporate",
            "media__corporate": "Media ↔ Corporate",
        }
    ) + "\n" + df["precedence_type_label"].map(PRECEDENCE_LABELS)

    combos = []
    for dyad in ["academic__corporate", "media__corporate"]:
        for precedence in PRECEDENCE_ORDER:
            combo_df = df[(df["dyad"] == dyad) & (df["precedence_type_label"] == precedence)]
            if not combo_df.empty:
                combos.append(combo_df["combo"].iloc[0])
    if not combos:
        combos = sorted(df["combo"].unique())

    plot_df = (
        df.assign(macro_topic=pd.Categorical(df["macro_topic"], categories=MACRO_TOPIC_ORDER, ordered=True))
        .pivot(index="macro_topic", columns="combo", values="aggregate_pair_count")
        .fillna(0)
        .reindex(index=MACRO_TOPIC_ORDER, columns=combos, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    im = ax.imshow(plot_df.values, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels([wrap_macro_label(v) for v in plot_df.index])
    ax.set_title("Macro-topic temporal pattern summary from retained aligned pairs")

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = int(plot_df.iloc[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Aggregate pair count")
    fig.tight_layout()
    output = results_dir / "figure_02_macro_topic_temporal_summary.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_sample_construction(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "table_01_sample_construction.csv")
    df = df.sort_values("step_order")
    labels = [text if len(text) <= 42 else text[:39] + "…" for text in df["sample_step"]]
    values = df["topic_count"]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    colors = ["#2A7F62" if idx in {0, 3, 4, 6} else "#5B7DB1" if idx in {1, 2} else "#C95C54" for idx in range(len(df))]
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.012, i, f"{int(value)}", va="center", ha="left", fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Analytic sample construction after review and comment round")
    ax.set_xlabel("Topic count")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "table_01_sample_construction.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_macro_summary(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "table_02_macro_topic_summary.csv")
    df["macro_topic"] = pd.Categorical(df["macro_topic"], categories=MACRO_TOPIC_ORDER, ordered=True)
    df = df.sort_values("macro_topic")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    scatter = ax.scatter(
        df["corporate_coverage_ratio"],
        range(len(df)),
        s=df["aligned_external_topic_count"] * 22,
        c=df["external_relevant_unpaired_count"],
        cmap="YlOrRd",
        alpha=0.9,
        edgecolor="#4C4C4C",
        linewidth=0.8,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([wrap_macro_label(v) for v in df["macro_topic"].astype(str)])
    ax.set_xlabel("Corporate coverage ratio")
    ax.set_xlim(-0.02, 1.05)
    ax.set_title("Macro-topic positioning: coverage vs. external spillovers")

    for _, row in df.iterrows():
        ax.text(
            row["corporate_coverage_ratio"] + 0.015,
            list(df["macro_topic"]).index(row["macro_topic"]),
            f"A={int(row['aligned_external_topic_count'])} | U={int(row['external_relevant_unpaired_count'])}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.axvline(0.5, color="#8C8C8C", linestyle="--", linewidth=1)
    ax.axvline(0.8, color="#8C8C8C", linestyle=":", linewidth=1)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88)
    cbar.set_label("Relevant external topics without pair")
    fig.text(0.79, 0.08, "Bubble size = aligned external topics", ha="center", fontsize=9, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "table_02_macro_topic_summary.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    nice_theme()
    outputs = [
        render_coverage(args.results_dir),
        render_temporal_summary(args.results_dir),
        render_sample_construction(args.results_dir),
        render_macro_summary(args.results_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
