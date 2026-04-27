#!/usr/bin/env python3
"""Build grouped figures for aggregated corporate-focus temporal precedence results."""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-nonstable", action="store_true")
    return parser.parse_args()


def clean_topic_label(raw: str) -> str:
    text = " ".join(str(raw or "").split())
    if "_" in text and text.split("_", 1)[0].isdigit():
        text = text.split("_", 1)[1]
    return text.replace("_", " ").strip()


def wrap_text(text: str, width: int) -> str:
    return textwrap.fill(" ".join(str(text).split()), width=width)


def precedence_note(row: pd.Series) -> str:
    label = str(row.get("precedence_type_label", "synchronous_or_unclear"))
    lag = row.get("best_lag_docs", None)
    corr = row.get("best_correlation_docs", None)
    if label == "synchronous_or_unclear":
        return "Precedence: unclear"
    lag_abs = abs(int(lag)) if pd.notna(lag) else "?"
    corr_text = f"{float(corr):.3f}" if pd.notna(corr) else "NA"
    return f"Precedence: {label.replace('_', ' ')} | lag={lag_abs}y | r={corr_text}"


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return text or "figure"


def load_inputs(input_root: Path, include_nonstable: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification = pd.read_csv(input_root / "aggregate_pair_precedence_classification.csv")
    series = pd.read_csv(input_root / "aggregate_pair_annual_series.csv")
    if not include_nonstable:
        classification = classification.loc[classification["is_stable_leading_pair"].fillna(False)].copy()
    classification = classification.sort_values(["macro_topic", "dyad", "matched_corporate_group_id"]).reset_index(drop=True)
    return classification, series


def composition_lines(row: pd.Series) -> list[str]:
    corporate_label = clean_topic_label(str(row.get("corporate_topic_name_original", "")))
    corp_line = (
        f"Corporate: {row['matched_corporate_group_id']} | "
        f"topic {int(row['corporate_micro_topic_id'])} | {corporate_label}"
    )
    names = json.loads(str(row.get("constituent_noncorporate_topic_names_json", "[]")))
    ids = json.loads(str(row.get("constituent_noncorporate_micro_topic_ids_json", "[]")))
    source_label = str(row.get("source_a", "")).capitalize()
    noncorp_header = (
        f"{source_label} aggregate: {int(row.get('constituent_noncorporate_count', 0))} microtopics "
        f"from {row.get('subgroup_a', '')}"
    )
    topic_lines = []
    for topic_id, name in zip(ids, names, strict=False):
        topic_lines.append(f"- {int(topic_id)}: {clean_topic_label(str(name))}")
    return [corp_line, noncorp_header, *topic_lines]


def plot_macro_topic(group: pd.DataFrame, series: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    macro_topic = str(group["macro_topic"].iloc[0])
    macro_topic_name = str(group["macro_topic_name"].iloc[0])
    nrows = len(group)
    max_constituents = int(group["constituent_noncorporate_count"].max()) if "constituent_noncorporate_count" in group.columns and not group.empty else 1
    row_height = min(8.0, max(4.8, 2.5 + 0.20 * max_constituents))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(18, row_height * nrows + 1.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.4]},
    )

    if nrows == 1:
        axes = [axes]

    for ax_plot, ax_text in axes:
        ax_text.axis("off")

    for (ax_plot, ax_text), row in zip(axes, group.itertuples(index=False), strict=False):
        pair_series = series.loc[series["aggregate_pair_id"] == row.aggregate_pair_id].copy().sort_values("year")
        years = pair_series["year"].astype(int)
        share_a = pair_series["source_a_share_docs"].fillna(0.0) * 100.0
        share_b = pair_series["source_b_share_docs"].fillna(0.0) * 100.0

        ax_plot.plot(years, share_a, color="#1f77b4", linewidth=2.3, marker="o", markersize=3, label=row.source_a)
        ax_plot.plot(years, share_b, color="#d62728", linewidth=2.3, marker="o", markersize=3, label=row.source_b)
        ax_plot.set_ylabel("Doc share (%)")
        ax_plot.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax_plot.tick_params(axis="x", labelrotation=45)
        ymax = max(float(share_a.max()), float(share_b.max()), 0.5)
        ax_plot.set_ylim(0, ymax * 1.18)

        title = (
            f"{row.dyad.replace('__', ' ↔ ')} | corporate group {row.matched_corporate_group_id}\n"
            f"direct pairs={int(row.direct_pair_count)} | aggregated microtopics={int(row.constituent_noncorporate_count)}"
        )
        ax_plot.set_title(title, fontsize=11)
        ax_plot.text(
            0.01,
            0.98,
            precedence_note(pd.Series(row._asdict())),
            transform=ax_plot.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
        )
        ax_plot.legend(loc="upper right", frameon=False, fontsize=9)

        comp_text = "\n".join(wrap_text(line, 56) for line in composition_lines(pd.Series(row._asdict())))
        ax_text.text(
            0.0,
            1.0,
            comp_text,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            linespacing=1.22,
        )

    fig.suptitle(
        f"Corporate-Focused Aggregated Temporal Relations | {macro_topic} | {macro_topic_name}",
        fontsize=15,
        y=1.01,
    )

    base_name = f"{macro_topic}_{slugify(macro_topic_name)}_aggregated_temporal_relations"
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_root / "figures_by_macro_topic")
    output_dir.mkdir(parents=True, exist_ok=True)

    classification, series = load_inputs(args.input_root, include_nonstable=args.include_nonstable)
    if classification.empty:
        raise SystemExit(f"No rows available to plot in {args.input_root}")

    manifest_rows = []
    for _, group in classification.groupby(["macro_topic", "macro_topic_name"], dropna=False):
        png_path, pdf_path = plot_macro_topic(group.reset_index(drop=True), series, output_dir)
        manifest_rows.append(
            {
                "macro_topic": str(group["macro_topic"].iloc[0]),
                "macro_topic_name": str(group["macro_topic_name"].iloc[0]),
                "pair_count": int(len(group)),
                "png_path": str(png_path),
                "pdf_path": str(pdf_path),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("macro_topic").reset_index(drop=True)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    (output_dir / "README.txt").write_text(
        "Figures are grouped by corporate macro-topic. Each row shows one aggregated pair, "
        "with the temporal series on the left and the full list of composing microtopics on the right.\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
