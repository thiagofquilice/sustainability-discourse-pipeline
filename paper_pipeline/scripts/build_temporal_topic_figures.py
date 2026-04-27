#!/usr/bin/env python3
"""Build temporal tables and figures for the adjusted 6-topic full corpus."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


PIPELINE_ROOT = Path("paper_pipeline")
FULL_BEST_ONLY = PIPELINE_ROOT / "outputs" / "full_run" / "best_only" / "validation_output.csv"
REMAP_FILE = (
    PIPELINE_ROOT
    / "outputs"
    / "t2_secondary_recovery_cosine_round"
    / "old_t2_under_secondary_recovery_variant.parquet"
)
CHANGED_GEMMA = (
    PIPELINE_ROOT
    / "outputs"
    / "t2_secondary_recovery_cosine_round"
    / "gemma_local_full"
    / "validation_output.csv"
)
LABELED_CORPUS = PIPELINE_ROOT / "outputs" / "cosine" / "labeled_corpus.parquet"
BEST_ONLY_POSITIVE = PIPELINE_ROOT / "outputs" / "cosine" / "best_only_positive.parquet"
CATALOG = PIPELINE_ROOT / "catalog" / "six_topic_discourse_catalog.csv"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "temporal_prep"
FIGURES_DIR = OUTPUT_DIR / "figures"

SOURCE_ORDER = ["academic", "corporate", "media"]
TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
TOPIC_COLORS = {
    "T1": "#1b9e77",
    "T2": "#d95f02",
    "T3": "#7570b3",
    "T4": "#e7298a",
    "T5": "#66a61e",
    "T6": "#e6ab02",
}
SOURCE_COLORS = {
    "academic": "#1f77b4",
    "corporate": "#d62728",
    "media": "#2ca02c",
}


def read_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG)
    return catalog[["topic_code", "topic_name"]].rename(columns={"topic_code": "assigned_label"})


def build_adjusted_full() -> pd.DataFrame:
    full = pd.read_csv(
        FULL_BEST_ONLY,
        usecols=[
            "chunk_id",
            "source",
            "year",
            "source_doc_id",
            "assigned_label",
            "topic_name",
            "v_gemma",
        ],
    )
    remap = pd.read_parquet(
        REMAP_FILE,
        columns=["chunk_id", "transition"],
    ).drop_duplicates(subset=["chunk_id"])
    changed = pd.read_csv(
        CHANGED_GEMMA,
        usecols=[
            "chunk_id",
            "source",
            "year",
            "source_doc_id",
            "assigned_label",
            "topic_name",
            "v_gemma",
        ],
    )

    full["chunk_id"] = full["chunk_id"].astype(str)
    changed["chunk_id"] = changed["chunk_id"].astype(str)
    remap["chunk_id"] = remap["chunk_id"].astype(str)

    full_t2 = full.loc[full["assigned_label"] == "T2"].merge(remap, on="chunk_id", how="left")
    remain_t2 = full_t2.loc[full_t2["transition"] == "T2_to_T2", full.columns]
    changed = changed.loc[changed["chunk_id"].isin(set(full["chunk_id"]))].copy()
    full_non_t2 = full.loc[full["assigned_label"] != "T2", full.columns]

    adjusted = pd.concat([full_non_t2, remain_t2, changed[full.columns]], ignore_index=True)
    adjusted["year"] = pd.to_numeric(adjusted["year"], errors="coerce").astype("Int64")
    adjusted = adjusted.dropna(subset=["source", "year", "source_doc_id", "assigned_label", "v_gemma"]).copy()
    adjusted["year"] = adjusted["year"].astype(int)
    return adjusted


def build_document_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_docs = pd.read_parquet(
        LABELED_CORPUS,
        columns=["source", "year", "source_doc_id"],
    ).drop_duplicates()
    positive_docs = pd.read_parquet(
        BEST_ONLY_POSITIVE,
        columns=["source", "year", "source_doc_id"],
    ).drop_duplicates()

    all_docs["year"] = pd.to_numeric(all_docs["year"], errors="coerce").astype("Int64")
    positive_docs["year"] = pd.to_numeric(positive_docs["year"], errors="coerce").astype("Int64")
    all_docs = all_docs.dropna(subset=["source", "year", "source_doc_id"]).copy()
    positive_docs = positive_docs.dropna(subset=["source", "year", "source_doc_id"]).copy()
    all_docs["year"] = all_docs["year"].astype(int)
    positive_docs["year"] = positive_docs["year"].astype(int)
    return all_docs, positive_docs


def build_temporal_tables(adjusted: pd.DataFrame, all_docs: pd.DataFrame, positive_docs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    adjusted_yes = adjusted.loc[adjusted["v_gemma"] == "yes"].copy()
    doc_topic = adjusted_yes[["source", "year", "source_doc_id", "assigned_label", "topic_name"]].drop_duplicates()
    yes_docs = adjusted_yes[["source", "year", "source_doc_id"]].drop_duplicates()

    year_grid = (
        all_docs[["source", "year"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"assigned_label": TOPIC_ORDER}), how="cross")
    )

    topic_doc_counts = (
        doc_topic.groupby(["source", "year", "assigned_label", "topic_name"], dropna=False)
        .size()
        .reset_index(name="topic_doc_n")
    )
    total_all_docs = (
        all_docs.groupby(["source", "year"], dropna=False)
        .size()
        .reset_index(name="all_doc_n")
    )
    total_positive_docs = (
        positive_docs.groupby(["source", "year"], dropna=False)
        .size()
        .reset_index(name="positive_doc_n")
    )
    total_yes_docs = (
        yes_docs.groupby(["source", "year"], dropna=False)
        .size()
        .reset_index(name="yes_doc_n")
    )

    catalog = read_catalog()
    topic_doc_counts = topic_doc_counts.drop(columns=["topic_name"]).merge(catalog, on="assigned_label", how="left")
    year_grid = year_grid.merge(catalog, on="assigned_label", how="left")

    presence_all = (
        year_grid.merge(topic_doc_counts, on=["source", "year", "assigned_label", "topic_name"], how="left")
        .merge(total_all_docs, on=["source", "year"], how="left")
        .fillna({"topic_doc_n": 0})
    )
    presence_all["topic_doc_n"] = presence_all["topic_doc_n"].astype(int)
    presence_all["presence_rate"] = presence_all["topic_doc_n"] / presence_all["all_doc_n"]

    presence_positive = (
        year_grid.merge(topic_doc_counts, on=["source", "year", "assigned_label", "topic_name"], how="left")
        .merge(total_positive_docs, on=["source", "year"], how="left")
        .fillna({"topic_doc_n": 0})
    )
    presence_positive["topic_doc_n"] = presence_positive["topic_doc_n"].astype(int)
    presence_positive["presence_rate"] = presence_positive["topic_doc_n"] / presence_positive["positive_doc_n"]

    share_within_yes = (
        year_grid.merge(topic_doc_counts, on=["source", "year", "assigned_label", "topic_name"], how="left")
        .merge(total_yes_docs, on=["source", "year"], how="left")
        .fillna({"topic_doc_n": 0})
    )
    share_within_yes["topic_doc_n"] = share_within_yes["topic_doc_n"].astype(int)
    share_within_yes["share_within_yes"] = share_within_yes["topic_doc_n"] / share_within_yes["yes_doc_n"]

    indexed_rows: list[pd.DataFrame] = []
    indexed_avg_rows: list[pd.DataFrame] = []
    for (source, topic), group in share_within_yes.groupby(["source", "assigned_label"], dropna=False):
        group = group.sort_values("year").copy()
        positive = group.loc[group["share_within_yes"] > 0]
        if positive.empty:
            group["indexed_share"] = pd.NA
            group["indexed_share_avg_base"] = pd.NA
            group["index_avg_base_start_year"] = pd.NA
            group["index_avg_base_end_year"] = pd.NA
            group["index_avg_base_value"] = pd.NA
        else:
            base = float(positive.iloc[0]["share_within_yes"])
            base_year = int(positive.iloc[0]["year"])
            group["indexed_share"] = group["share_within_yes"] / base * 100.0
            group.loc[group["year"] < base_year, "indexed_share"] = pd.NA
            group["index_base_year"] = base_year
            group["index_base_value"] = base
            avg_base_subset = positive.head(3).copy()
            avg_base = float(avg_base_subset["share_within_yes"].mean())
            avg_base_start_year = int(avg_base_subset["year"].min())
            avg_base_end_year = int(avg_base_subset["year"].max())
            group["indexed_share_avg_base"] = group["share_within_yes"] / avg_base * 100.0
            group.loc[group["year"] < avg_base_start_year, "indexed_share_avg_base"] = pd.NA
            group["index_avg_base_start_year"] = avg_base_start_year
            group["index_avg_base_end_year"] = avg_base_end_year
            group["index_avg_base_value"] = avg_base
        indexed_rows.append(group)
        indexed_avg_rows.append(group)
    indexed = pd.concat(indexed_rows, ignore_index=True)
    indexed_avg = pd.concat(indexed_avg_rows, ignore_index=True)

    corp_counts = (
        adjusted_yes.loc[adjusted_yes["source"] == "corporate", ["year", "source_doc_id", "assigned_label", "topic_name", "chunk_id"]]
        .groupby(["year", "source_doc_id", "assigned_label", "topic_name"], dropna=False)
        .size()
        .reset_index(name="topic_chunk_count")
    )
    corporate_intensity = (
        corp_counts.groupby(["year", "assigned_label", "topic_name"], dropna=False)["topic_chunk_count"]
        .mean()
        .reset_index(name="mean_topic_chunks_per_doc")
    )

    return {
        "doc_topic_presence_by_source_year": presence_all,
        "doc_topic_presence_by_source_year_positive_denominator": presence_positive,
        "topic_share_within_yes_by_source_year": indexed,
        "topic_share_within_yes_by_source_year_avg_base": indexed_avg,
        "corporate_topic_mean_mentions_per_doc_by_year": corporate_intensity,
    }


def percent_formatter(x: float, _pos: int) -> str:
    return f"{x * 100:.0f}%"


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def figure1(presence: pd.DataFrame, stem: str, title: str, base_note: str) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, source in zip(axes, SOURCE_ORDER):
        subset = presence.loc[presence["source"] == source].sort_values("year")
        for topic in TOPIC_ORDER:
            series = subset.loc[subset["assigned_label"] == topic]
            ax.plot(
                series["year"],
                series["presence_rate"],
                label=topic,
                color=TOPIC_COLORS[topic],
                linewidth=2,
            )
        ax.set_title(source.capitalize())
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    axes[0].set_ylabel("Documents containing topic (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle(title, fontsize=15, y=1.03)
    fig.text(
        0.01,
        0.01,
        base_note,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.92])
    return save_figure(fig, stem)


def figure2(indexed: pd.DataFrame, stem: str, title: str, base_note: str) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    for ax, topic in zip(axes, TOPIC_ORDER):
        subset = indexed.loc[indexed["assigned_label"] == topic].sort_values("year")
        for source in SOURCE_ORDER:
            series = subset.loc[subset["source"] == source]
            ax.plot(
                series["year"],
                series["indexed_share"],
                label=source,
                color=SOURCE_COLORS[source],
                linewidth=2,
            )
        topic_name = subset["topic_name"].dropna().iloc[0] if not subset["topic_name"].dropna().empty else topic
        ax.set_title(f"{topic} {topic_name}")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
        ax.axhline(100, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Index (base = 100)")
    axes[3].set_ylabel("Index (base = 100)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle(title, fontsize=15, y=1.03)
    fig.text(
        0.01,
        0.01,
        base_note,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    return save_figure(fig, stem)


def figure2_avg_base(indexed: pd.DataFrame, stem: str, title: str, base_note: str) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    for ax, topic in zip(axes, TOPIC_ORDER):
        subset = indexed.loc[indexed["assigned_label"] == topic].sort_values("year")
        for source in SOURCE_ORDER:
            series = subset.loc[subset["source"] == source]
            ax.plot(
                series["year"],
                series["indexed_share_avg_base"],
                label=source,
                color=SOURCE_COLORS[source],
                linewidth=2,
            )
        topic_name = subset["topic_name"].dropna().iloc[0] if not subset["topic_name"].dropna().empty else topic
        ax.set_title(f"{topic} {topic_name}")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
        ax.axhline(100, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Index (base = 100)")
    axes[3].set_ylabel("Index (base = 100)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)
    fig.suptitle(title, fontsize=15, y=1.03)
    fig.text(
        0.01,
        0.01,
        base_note,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    return save_figure(fig, stem)


def figure3(corporate_intensity: pd.DataFrame, stem: str, title: str, base_note: str) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    for ax, topic in zip(axes, TOPIC_ORDER):
        subset = corporate_intensity.loc[corporate_intensity["assigned_label"] == topic].sort_values("year")
        ax.plot(
            subset["year"],
            subset["mean_topic_chunks_per_doc"],
            color=SOURCE_COLORS["corporate"],
            linewidth=2,
        )
        topic_name = subset["topic_name"].dropna().iloc[0] if not subset["topic_name"].dropna().empty else topic
        ax.set_title(f"{topic} {topic_name}")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean topic-positive chunks per document")
    axes[3].set_ylabel("Mean topic-positive chunks per document")
    fig.suptitle(title, fontsize=15, y=1.02)
    fig.text(
        0.01,
        0.01,
        base_note,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    return save_figure(fig, stem)


def write_descriptions() -> Path:
    path = OUTPUT_DIR / "figure_descriptions.md"
    text = """# Figure Descriptions

## Figure 1: Topic presence in the source universe
For each source-year and topic, the plotted value equals the share of documents containing at least one validated mention of that topic. The main version uses all observed documents in the source-year as the denominator, including documents that never entered the topical pipeline. The alternative version uses only topic-positive documents in the source-year as the denominator. The unit of analysis is the document, identified by `source_doc_id`.

## Figure 2: Relative topic attention within each source over time
For each source-year and topic, the underlying ratio equals the number of documents containing the topic divided by all topic-positive documents in that source-year. Each source-topic series is then indexed to 100 in its first non-zero year, so the figure emphasizes relative change over time rather than differences in absolute corpus size across sources.

## Figure 2b: Relative topic attention with averaged base years
For each source-year and topic, the underlying ratio equals the number of documents containing the topic divided by all topic-positive documents in that source-year. Each source-topic series is then indexed to 100 using the average of the first three non-zero years in that series, which reduces sensitivity to a single early-year fluctuation.

## Figure 3: Corporate topic mention intensity
For each corporate-year and topic, the plotted value equals the mean number of validated topic chunks per document among corporate documents that contain at least one chunk for that topic. The unit of analysis is the corporate document, identified by `source_doc_id`, and the figure is intended to show whether firms that discuss a topic appear to discuss it more extensively over time.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    adjusted = build_adjusted_full()
    all_docs, positive_docs = build_document_universe()
    tables = build_temporal_tables(adjusted, all_docs, positive_docs)

    output_paths: dict[str, str] = {}
    for name, table in tables.items():
        path = OUTPUT_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = str(path)

    fig1_note = (
        "Base calculation: For each source-year, the value equals the share of documents containing at least one "
        "validated mention of the topic. Denominator = all documents observed in that source-year."
    )
    fig1b_note = (
        "Base calculation: For each source-year, the value equals the share of documents containing at least one "
        "validated mention of the topic. Denominator = topic-positive documents only."
    )
    fig2_note = (
        "Base calculation: For each source-year, the underlying ratio is documents containing the topic divided by "
        "all topic-positive documents in that source-year. Each source-topic series is then indexed to 100 in its "
        "first non-zero year."
    )
    fig2b_note = (
        "Base calculation: For each source-year, the underlying ratio is documents containing the topic divided by "
        "all topic-positive documents in that source-year. Each source-topic series is then indexed to 100 using "
        "the average of its first three non-zero years."
    )
    fig3_note = (
        "Base calculation: For each corporate-year and topic, the value equals the mean number of validated "
        "topic chunks per document among corporate documents that contain at least one chunk for that topic."
    )

    figure_files = {
        "figure_01_topic_presence_in_source_universe": figure1(
            tables["doc_topic_presence_by_source_year"],
            "figure_01_topic_presence_in_source_universe",
            "Topic Presence in the Source Universe",
            fig1_note,
        ),
        "figure_01b_topic_presence_positive_only_denominator": figure1(
            tables["doc_topic_presence_by_source_year_positive_denominator"],
            "figure_01b_topic_presence_positive_only_denominator",
            "Topic Presence with Topic-Positive Denominator",
            fig1b_note,
        ),
        "figure_02_relative_topic_attention_indexed": figure2(
            tables["topic_share_within_yes_by_source_year"],
            "figure_02_relative_topic_attention_indexed",
            "Relative Topic Attention Within Each Source Over Time",
            fig2_note,
        ),
        "figure_02b_relative_topic_attention_indexed_avg_base": figure2_avg_base(
            tables["topic_share_within_yes_by_source_year_avg_base"],
            "figure_02b_relative_topic_attention_indexed_avg_base",
            "Relative Topic Attention Within Each Source Over Time (Averaged Base Years)",
            fig2b_note,
        ),
        "figure_03_corporate_topic_mention_intensity": figure3(
            tables["corporate_topic_mean_mentions_per_doc_by_year"],
            "figure_03_corporate_topic_mention_intensity",
            "Corporate Topic Mention Intensity Over Time",
            fig3_note,
        ),
    }

    descriptions_path = write_descriptions()
    manifest = {
        "adjusted_full_rows": int(len(adjusted)),
        "adjusted_yes_rows": int((adjusted["v_gemma"] == "yes").sum()),
        "all_document_universe_rows": int(len(all_docs)),
        "pipeline_positive_document_rows": int(len(positive_docs)),
        "table_outputs": output_paths,
        "figure_outputs": figure_files,
        "description_file": str(descriptions_path),
        "assumptions": {
            "document_id": "source_doc_id",
            "topic_presence_unit": "document contains at least one validated chunk for the topic",
            "figure_1_denominator_main": "all documents observed in source-year",
            "figure_1_denominator_appendix": "topic-positive documents observed in source-year",
            "figure_2_index_rule": "first non-zero year per source-topic series",
            "figure_2b_index_rule": "average of first three non-zero years per source-topic series",
            "figure_3_metric": "mean yes chunks per corporate document among documents mentioning the topic",
            "smoothing": "none",
        },
    }
    (OUTPUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
