#!/usr/bin/env python3
"""Build summary tables, methods notes, and figures for the micro BERTopic temporal reading workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from micro_topic_evolution_common import OUTPUT_ROOT, SOURCE_ORDER, configure_logging, ensure_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def plot_heatmap(data: pd.DataFrame, title: str, cbar_label: str, output_base: Path, footnote: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    matrix = data.fillna(0.0).to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(x) for x in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_phase_range(text: str) -> tuple[int | None, int | None]:
    if not isinstance(text, str) or not text.strip():
        return None, None
    match = re.search(r"(\d{4})\D+(\d{4})", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def write_notes(output_root: Path) -> None:
    methods = """# Micro BERTopic Temporal Evolution Methods Note

This workflow interprets how selected BERTopic micro-topics vary over time within each source-topic subgroup. BERTopic topics remain fixed after fitting. We use BERTopic `topics_over_time` outputs to track temporal variation in topic prevalence and year-specific word representation. We interpret this as topic evolution over time, but not as a strict dynamic topic model with topics re-estimated at each period.

The workflow has two LLM stages. First, yearly summaries are generated from the most representative chunks available for each selected micro-topic and year, together with the year-specific BERTopic topic words. Second, these annual summaries are synthesized into phase-based temporal narratives. Both stages use Gemma 4 E4B via local Ollama.
"""
    accessible = """# Micro BERTopic Temporal Evolution Methods Note in Accessible English

We keep the BERTopic micro-topics fixed and then examine how their wording and frequency change over time. The `topics_over_time` outputs help us track those changes year by year. This lets us describe topic evolution over time, but we do not claim that the model itself was dynamically refit at each period.

We then use Gemma 4 E4B in two steps. First, it summarizes what each selected micro-topic looks like in each year using the most representative yearly chunks and the BERTopic words for that year. Second, it reads those year-by-year summaries and writes a short phase-based account of how the micro-topic evolves.
"""
    figures = """# Micro BERTopic Temporal Evolution Figure Descriptions

## Figure 1: Selected micro-topics per subgroup
This heatmap shows how many BERTopic micro-topics were selected in each subgroup after applying the cumulative coverage rule.

## Figure 2: Coverage achieved by selected micro-topics
This heatmap shows the share of non-outlier subgroup rows covered by the selected micro-topics.

## Figure 3: Example micro-topic evolution with phases
This panel figure shows the yearly BERTopic frequency of selected micro-topics together with the final phase segmentation inferred from the annual summaries.
"""
    figures_accessible = """# Micro BERTopic Temporal Evolution Figure Descriptions in Accessible English

## Figure 1: Selected micro-topics per subgroup
This heatmap shows how many internal micro-topics were kept for interpretation in each subgroup.

## Figure 2: Coverage achieved by selected micro-topics
This heatmap shows how much of each subgroup is covered by the selected micro-topics.

## Figure 3: Example micro-topic evolution with phases
This figure shows how some selected micro-topics rise or fall over time and how the final narrative divides that history into phases.
"""
    (output_root / "methods_note.md").write_text(methods, encoding="utf-8")
    (output_root / "methods_note_accessible_english.md").write_text(accessible, encoding="utf-8")
    (output_root / "figure_descriptions.md").write_text(figures, encoding="utf-8")
    (output_root / "figure_descriptions_accessible_english.md").write_text(figures_accessible, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_directory(args.output_root)
    figures_dir = args.output_root / "figures"
    ensure_directory(figures_dir)

    selected = pd.read_csv(args.output_root / "selected_micro_topics.csv")
    subgroups = pd.read_csv(args.output_root / "selected_micro_topics_by_subgroup.csv")
    annual = pd.read_csv(args.output_root / "year_summaries" / "micro_topic_year_summaries.csv")
    narratives = pd.read_csv(args.output_root / "evolution_summaries" / "micro_topic_evolution_narratives.csv")
    yearly_evidence = pd.read_csv(args.output_root / "micro_topic_year_evidence.csv")

    summary = selected.merge(
        narratives,
        on=["subgroup", "source", "assigned_label", "micro_topic_id", "topic_name_original"],
        how="left",
    )
    annual_counts = (
        annual.groupby(["subgroup", "micro_topic_id"], dropna=False)
        .size()
        .reset_index(name="annual_summary_count")
    )
    summary = summary.merge(annual_counts, on=["subgroup", "micro_topic_id"], how="left")
    summary["annual_summary_count"] = summary["annual_summary_count"].fillna(0).astype(int)
    summary = summary.rename(columns={"assigned_label": "macro_topic"})
    summary.to_csv(args.output_root / "micro_topic_evolution_summary.csv", index=False)

    subgroup_summary = subgroups.merge(
        annual.groupby("subgroup", dropna=False).size().reset_index(name="annual_summary_rows"),
        on="subgroup",
        how="left",
    )
    subgroup_summary["annual_summary_rows"] = subgroup_summary["annual_summary_rows"].fillna(0).astype(int)
    subgroup_summary.to_csv(args.output_root / "subgroup_evolution_coverage_summary.csv", index=False)

    subgroup_frame = subgroup_summary.copy()
    subgroup_frame["source"] = pd.Categorical(subgroup_frame["source"], categories=SOURCE_ORDER, ordered=True)
    subgroup_frame = subgroup_frame.sort_values(["source", "assigned_label"]).reset_index(drop=True)

    selected_heatmap = subgroup_frame.pivot(index="source", columns="assigned_label", values="selected_topic_count").reindex(SOURCE_ORDER)
    plot_heatmap(
        selected_heatmap,
        "Selected micro-topics per subgroup",
        "Selected topics",
        figures_dir / "selected_micro_topics_per_subgroup",
        "Base calculation: each cell shows how many non-outlier micro-topics were selected after applying the 75% cumulative coverage rule with a cap of 10.",
        "Blues",
    )

    coverage_heatmap = subgroup_frame.pivot(index="source", columns="assigned_label", values="selected_topic_coverage").reindex(SOURCE_ORDER)
    plot_heatmap(
        coverage_heatmap,
        "Coverage achieved by selected micro-topics",
        "Coverage share",
        figures_dir / "selected_micro_topic_coverage",
        "Base calculation: each cell shows the share of non-outlier subgroup rows covered by the selected micro-topics.",
        "YlOrBr",
    )

    examples = summary.sort_values(["topic_size", "subgroup"], ascending=[False, True]).head(6)[["subgroup", "micro_topic_id"]].drop_duplicates()
    if not examples.empty:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False, sharey=False)
        for ax, (_, ex) in zip(axes.flatten(), examples.iterrows()):
            subgroup = ex["subgroup"]
            micro_topic_id = int(ex["micro_topic_id"])
            freq = yearly_evidence.loc[
                (yearly_evidence["subgroup"] == subgroup) & (yearly_evidence["micro_topic_id"].astype(int) == micro_topic_id)
            ].sort_values("year")
            ax.plot(freq["year"], freq["year_frequency"], color="#1f77b4", linewidth=1.8)
            narrative_row = narratives.loc[
                (narratives["subgroup"] == subgroup) & (narratives["micro_topic_id"].astype(int) == micro_topic_id)
            ]
            if not narrative_row.empty:
                nr = narrative_row.iloc[0]
                colors = ["#dceaf7", "#f8e4c6", "#dff0d8", "#eadcf8"]
                for idx, phase_field in enumerate(["phase_1_years", "phase_2_years", "phase_3_years", "phase_4_years"]):
                    start, end = parse_phase_range(str(nr.get(phase_field, "")))
                    if start is None or end is None:
                        continue
                    ax.axvspan(start, end, color=colors[idx], alpha=0.35)
            ax.set_title(f"{subgroup} | topic {micro_topic_id}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Frequency")
        fig.suptitle("Example micro-topic evolution with inferred phases", y=0.98)
        fig.text(
            0.01,
            0.01,
            "Base calculation: the line shows yearly BERTopic frequency for one selected micro-topic, and shaded spans mark the final phase ranges inferred from the annual summaries.",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.savefig(figures_dir / "example_micro_topic_evolution_phases.png", dpi=200)
        fig.savefig(figures_dir / "example_micro_topic_evolution_phases.pdf")
        plt.close(fig)

    write_notes(args.output_root)
    write_json(
        args.output_root / "summary_manifest.json",
        {
            "selected_micro_topic_count": int(summary.shape[0]),
            "subgroup_count": int(subgroup_summary.shape[0]),
            "annual_summary_row_count": int(annual.shape[0]),
            "narrative_row_count": int(narratives.shape[0]),
            "outputs": {
                "micro_topic_evolution_summary": str(args.output_root / "micro_topic_evolution_summary.csv"),
                "subgroup_evolution_coverage_summary": str(args.output_root / "subgroup_evolution_coverage_summary.csv"),
                "methods_note": str(args.output_root / "methods_note.md"),
                "methods_note_accessible_english": str(args.output_root / "methods_note_accessible_english.md"),
            },
        },
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
