#!/usr/bin/env python3
"""Build descriptive analysis tables and figures for the supervised BERTopic phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic

from workflow_common import configure_logging, load_config


SOURCE_ORDER = ["academic", "media", "corporate"]


def plot_heatmap(data: pd.DataFrame, title: str, cbar_label: str, output_base: Path, footnote: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 9))
    matrix = data.fillna(0.0).to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(x) for x in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--document-topics", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def write_descriptions(output_dir: Path) -> None:
    technical = """# BERTopic Supervised Figure Descriptions

## Figure 1: BERTopic topic prevalence by source
This heatmap shows the share of rows assigned to each BERTopic topic within each source. The unit of analysis is the validated text row used in the supervised BERTopic fit.

## Figure 2: BERTopic topic evolution by source over time
This figure shows how the largest BERTopic topics evolve over time within each source. For each source-year-topic cell, the value equals the count of rows assigned to that BERTopic topic.

## Figure 3: BERTopic topic concentration within the six supervised labels
This heatmap shows how BERTopic topics distribute across the original six supervised labels. Each cell reports the share of rows within a label assigned to a BERTopic topic.
"""
    accessible = """# BERTopic Supervised Figure Descriptions in Accessible English

## Figure 1: BERTopic topic prevalence by source
This heatmap shows which BERTopic topics are most common in academic, media, and corporate texts after the supervised model is fitted.

## Figure 2: BERTopic topic evolution by source over time
This figure shows how the biggest BERTopic topics rise or fall over time within each source.

## Figure 3: BERTopic topic concentration within the six supervised labels
This heatmap shows how the discovered BERTopic topics connect back to the original six-topic taxonomy.
"""
    (output_dir / "figure_descriptions.md").write_text(technical, encoding="utf-8")
    (output_dir / "figure_descriptions_accessible_english.md").write_text(accessible, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["paths"]["descriptive_analysis_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    document_topics_path = args.document_topics or Path(config["paths"]["document_topics"])
    frame = pd.read_parquet(document_topics_path).copy()
    frame["source"] = pd.Categorical(frame["source"], categories=SOURCE_ORDER, ordered=True)

    counts_by_topic_source = (
        frame.groupby(["bertopic_topic_id", "source"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    counts_by_topic_label = (
        frame.groupby(["bertopic_topic_id", "assigned_label"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    counts_by_topic_source_label = (
        frame.groupby(["bertopic_topic_id", "source", "assigned_label"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    source_year_topic = (
        frame.groupby(["source", "year", "bertopic_topic_id"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["source", "year", "row_count"], ascending=[True, True, False])
    )
    source_year_totals = source_year_topic.groupby(["source", "year"], dropna=False)["row_count"].transform("sum")
    source_year_topic["share_within_source_year"] = source_year_topic["row_count"] / source_year_totals

    label_concentration = counts_by_topic_label.copy()
    label_totals = label_concentration.groupby("assigned_label")["row_count"].transform("sum")
    label_concentration["share_within_label"] = label_concentration["row_count"] / label_totals

    counts_by_topic_source.to_csv(output_dir / "bertopic_topic_counts_by_source.csv", index=False)
    counts_by_topic_label.to_csv(output_dir / "bertopic_topic_counts_by_assigned_label.csv", index=False)
    counts_by_topic_source_label.to_csv(output_dir / "bertopic_topic_counts_by_source_assigned_label.csv", index=False)
    source_year_topic.to_csv(output_dir / "bertopic_topic_prevalence_by_source_year.csv", index=False)
    label_concentration.to_csv(output_dir / "bertopic_topic_concentration_within_label.csv", index=False)

    top_topics = (
        frame.groupby("bertopic_topic_id", dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
        .head(8)["bertopic_topic_id"]
        .tolist()
    )

    representative = (
        frame.sort_values(["bertopic_topic_id", "bertopic_probability"], ascending=[True, False])
        .groupby("bertopic_topic_id", dropna=False)
        .head(5)[
            ["bertopic_topic_id", "source", "year", "chunk_id", "source_doc_id", "assigned_label", "topic_name", "bertopic_probability", "text"]
        ]
    )
    representative.to_csv(output_dir / "representative_docs_by_bertopic_topic.csv", index=False)

    prevalence = counts_by_topic_source.copy()
    prevalence["share_within_source"] = prevalence["row_count"] / prevalence.groupby("source")["row_count"].transform("sum")
    pivot = prevalence.pivot(index="bertopic_topic_id", columns="source", values="share_within_source").fillna(0.0)
    plot_heatmap(
        pivot,
        "BERTopic topic prevalence by source",
        "Share within source",
        figures_dir / "bertopic_topic_prevalence_by_source",
        "Base calculation: each cell shows the share of validated text rows assigned to a BERTopic topic within a source.",
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=False)
    for ax, source_value in zip(axes, SOURCE_ORDER):
        subset = source_year_topic.loc[(source_year_topic["source"] == source_value) & (source_year_topic["bertopic_topic_id"].isin(top_topics))].copy()
        for topic_id, topic_subset in subset.groupby("bertopic_topic_id", dropna=False):
            ax.plot(topic_subset["year"], topic_subset["row_count"], label=f"Topic {topic_id}", linewidth=1.5)
        ax.set_title(str(source_value).title())
        ax.set_ylabel("Row count")
    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("BERTopic topic evolution by source", y=0.98)
    fig.text(
        0.01,
        0.01,
        "Base calculation: each line shows the annual number of validated text rows assigned to a BERTopic topic within a source.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(figures_dir / "bertopic_topics_over_time_by_source.png", dpi=200)
    fig.savefig(figures_dir / "bertopic_topics_over_time_by_source.pdf")
    plt.close(fig)

    concentration_pivot = label_concentration.pivot(index="bertopic_topic_id", columns="assigned_label", values="share_within_label").fillna(0.0)
    plot_heatmap(
        concentration_pivot,
        "BERTopic topic concentration within supervised labels",
        "Share within label",
        figures_dir / "bertopic_topic_concentration_within_label",
        "Base calculation: each cell shows the share of rows within a six-topic label assigned to a BERTopic topic.",
    )

    try:
        model_dir = args.model_dir or Path(config["paths"]["model_dir"])
        topic_model = BERTopic.load(str(model_dir))
        info = topic_model.get_topic_info()
        info.to_csv(output_dir / "topic_info_from_saved_model.csv", index=False)
    except Exception:
        pass

    write_descriptions(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
