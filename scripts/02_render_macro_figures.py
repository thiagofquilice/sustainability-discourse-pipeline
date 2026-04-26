#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from common import MACRO_TOPIC_ORDER, STATUS_ORDER, ensure_dir


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
SOURCE_COLORS = {
    "academic_document_count": "#C07A1E",
    "media_document_count": "#1F8A70",
    "corporate_document_count": "#163A70",
}
SOURCE_LABELS = {
    "academic_document_count": "Academic",
    "media_document_count": "Media",
    "corporate_document_count": "Corporate",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render macro-level paper figures.")
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def set_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D8D8D8",
            "grid.linestyle": ":",
            "axes.edgecolor": "#C8C8C8",
        }
    )


def wrap_macro_label(topic: str) -> str:
    labels = {
        "T1": "T1  Clean energy\ntransition",
        "T2": "T2  Operational sustainability\nand circular production",
        "T3": "T3  Sustainable products,\nservices and consumption",
        "T4": "T4  Climate strategy,\ncarbon governance and disclosure",
        "T5": "T5  Climate risk,\nadaptation and resilience",
        "T6": "T6  Ecosystems, pollution\nand stewardship",
    }
    return labels.get(topic, topic)


def render_sample_construction(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "table_01_sample_construction.csv").sort_values("step_order")
    labels = [label if len(label) <= 44 else label[:41] + "..." for label in df["sample_step"]]
    values = df["topic_count"]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    colors = ["#2A7F62", "#5B7DB1", "#5B7DB1", "#2A7F62", "#E6A532", "#C95C54", "#2A7F62"]
    ax.barh(labels, values, color=colors[: len(df)], edgecolor="white", linewidth=1.2)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.012, index, f"{int(value)}", va="center", ha="left")
    ax.invert_yaxis()
    ax.set_title("Analytic sample construction after final review")
    ax.set_xlabel("Topic count")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "table_01_sample_construction.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_macro_document_counts(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "figure_00_macro_document_counts.csv")
    df["macro_topic"] = pd.Categorical(df["macro_topic"], MACRO_TOPIC_ORDER, ordered=True)
    df = df.sort_values("macro_topic")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    left = pd.Series(0, index=df.index, dtype=float)
    labels = [wrap_macro_label(str(topic)) for topic in df["macro_topic"]]
    for column in SOURCE_COLORS:
        values = df[column].fillna(0)
        ax.barh(
            labels,
            values,
            left=left,
            color=SOURCE_COLORS[column],
            edgecolor="white",
            linewidth=1.0,
            label=SOURCE_LABELS[column],
        )
        left = left + values

    for index, total in enumerate(df["total_unique_document_count"]):
        ax.text(total + df["total_unique_document_count"].max() * 0.01, index, f"{int(total):,}", va="center", ha="left")

    ax.invert_yaxis()
    ax.set_title("Final retained documents by source and macro topic")
    ax.set_xlabel("Unique documents")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "figure_00_macro_document_counts.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_coverage(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "figure_01_macro_topic_coverage.csv")
    pivot = (
        df.assign(macro_topic=pd.Categorical(df["macro_topic"], MACRO_TOPIC_ORDER, ordered=True))
        .pivot(index="macro_topic", columns="paper_status", values="topic_count")
        .fillna(0)
        .reindex(MACRO_TOPIC_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    left = pd.Series(0, index=pivot.index, dtype=float)
    for status in STATUS_ORDER:
        widths = pivot.get(status, pd.Series(0, index=pivot.index))
        ax.barh(
            [wrap_macro_label(str(topic)) for topic in pivot.index],
            widths,
            left=left,
            color=STATUS_COLORS[status],
            edgecolor="white",
            linewidth=1.2,
            label=STATUS_LABELS[status],
        )
        for index, value in enumerate(widths):
            if value:
                ax.text(left.iloc[index] + value / 2, index, f"{int(value)}", va="center", ha="center", color="white", fontweight="bold")
        left = left + widths

    ax.set_title("Corporate-focus coverage and external gaps by macro topic")
    ax.set_xlabel("Non-corporate topics after final review")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "figure_01_macro_topic_coverage.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_timing_diagnostics(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "figure_02_source_timing_diagnostics.csv")
    if df.empty:
        return results_dir / "figure_02_source_timing_diagnostics.png"

    df["macro_topic"] = pd.Categorical(df["macro_topic"], MACRO_TOPIC_ORDER, ordered=True)
    df = df.sort_values(["macro_topic", "relation_level"])
    levels = ["aggregate", "individual"]
    colors = {"aggregate": "#5B7DB1", "individual": "#C07A1E"}
    labels = {"aggregate": "Aggregate source relations", "individual": "Individual topic relations"}

    fig, axes = plt.subplots(ncols=2, figsize=(12.2, 6.2), sharey=True)
    y_positions = range(len(MACRO_TOPIC_ORDER))
    for axis_index, (metric, title) in enumerate(
        [
            ("relation_count", "Relations tested"),
            ("spearman_p_lt_10_count", "Same-year Spearman p < .10"),
        ]
    ):
        ax = axes[axis_index]
        left = pd.Series(0, index=MACRO_TOPIC_ORDER, dtype=float)
        for level in levels:
            subset = (
                df[df["relation_level"] == level]
                .set_index("macro_topic")[metric]
                .reindex(MACRO_TOPIC_ORDER)
                .fillna(0)
            )
            ax.barh(
                [wrap_macro_label(topic) for topic in MACRO_TOPIC_ORDER],
                subset,
                left=left,
                color=colors[level],
                edgecolor="white",
                linewidth=1.0,
                label=labels[level],
            )
            for index, value in enumerate(subset):
                if value:
                    ax.text(left.iloc[index] + value / 2, index, f"{int(value)}", ha="center", va="center", color="white", fontweight="bold")
            left = left + subset
        ax.set_title(title)
        ax.set_xlabel("Relation count")
        ax.set_yticks(list(y_positions))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Source-relation timing diagnostics", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.06,
        0.02,
        "Diagnostics are descriptive: same-year Spearman indicates co-movement, while onset and peak gaps are reported in the CSV table.",
        ha="left",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    output = results_dir / "figure_02_source_timing_diagnostics.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_macro_summary(results_dir: Path) -> Path:
    df = pd.read_csv(results_dir / "table_02_macro_topic_summary.csv")
    df["macro_topic"] = pd.Categorical(df["macro_topic"], MACRO_TOPIC_ORDER, ordered=True)
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
    ax.set_yticklabels([wrap_macro_label(str(topic)) for topic in df["macro_topic"]])
    ax.set_xlabel("Corporate coverage ratio")
    ax.set_xlim(-0.02, 1.05)
    ax.set_title("Macro-topic positioning: coverage and external spillovers")

    for index, row in enumerate(df.itertuples(index=False)):
        ax.text(
            row.corporate_coverage_ratio + 0.015,
            index,
            f"A={int(row.aligned_external_topic_count)} | U={int(row.external_relevant_unpaired_count)}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.axvline(0.5, color="#8C8C8C", linestyle="--", linewidth=1)
    ax.axvline(0.8, color="#8C8C8C", linestyle=":", linewidth=1)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.88)
    colorbar.set_label("Relevant external topics without pair")
    fig.text(0.78, 0.08, "Bubble size = aligned external topics", ha="center", fontsize=9, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = results_dir / "table_02_macro_topic_summary.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)
    set_theme()
    outputs = [
        render_sample_construction(args.results_dir),
        render_macro_document_counts(args.results_dir),
        render_coverage(args.results_dir),
        render_timing_diagnostics(args.results_dir),
        render_macro_summary(args.results_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
