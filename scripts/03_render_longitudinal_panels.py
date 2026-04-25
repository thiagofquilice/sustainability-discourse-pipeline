#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd

from common import MACRO_TOPIC_ORDER, ensure_dir


MACRO_TOPIC_LABELS = {
    "T1": "T1  Clean energy transition",
    "T2": "T2  Operational sustainability and circular production",
    "T3": "T3  Sustainable products, services and consumption",
    "T4": "T4  Climate strategy, carbon governance and disclosure",
    "T5": "T5  Climate risk, adaptation and resilience",
    "T6": "T6  Ecosystems, pollution and stewardship",
}
SERIES_ORDER = ["corporate", "academic_aggregate", "media_aggregate"]
SERIES_STYLES = {
    "corporate": {"label": "Corporate", "color": "#163A70", "linestyle": "-", "linewidth": 2.4},
    "academic_aggregate": {"label": "Academic aggregate", "color": "#C07A1E", "linestyle": "--", "linewidth": 2.0},
    "media_aggregate": {"label": "Media aggregate", "color": "#1F8A70", "linestyle": ":", "linewidth": 2.4},
}
YEAR_TICKS = [2000, 2005, 2010, 2015, 2020, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render longitudinal relative-salience panels.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "#FAFAF8",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#D8D8D8",
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#C8C8C8",
        }
    )


def percent_formatter(value: float, _position: float) -> str:
    return f"{value * 100:.0f}%"


def wrapped(value: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def global_y_max(df: pd.DataFrame) -> float:
    max_value = float(df["relative_share_of_macro_topic_source_docs"].max())
    step = 0.05
    return max(step, math.ceil(max_value / step) * step)


def topic_order(df_macro: pd.DataFrame) -> list[str]:
    meta = (
        df_macro.groupby("corporate_final_merge_group_id", as_index=False)
        .agg(
            corporate_topic_label=("corporate_topic_label", "first"),
            corporate_topic_size_count=("corporate_topic_size_count", "max"),
        )
        .sort_values(
            ["corporate_topic_size_count", "corporate_topic_label", "corporate_final_merge_group_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    )
    return meta["corporate_final_merge_group_id"].astype(str).tolist()


def has_signal(df_series: pd.DataFrame, role: str) -> bool:
    if role == "corporate":
        return True
    return (
        df_series["year_frequency"].gt(0).any()
        or df_series["n_contributing_external_topics"].gt(0).any()
        or df_series["contributing_external_group_count"].gt(0).any()
    )


def legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for role in SERIES_ORDER:
        style = SERIES_STYLES[role]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=style["label"],
            )
        )
    return handles


def render_macro(df_macro: pd.DataFrame, macro_topic: str, output_dir: Path, y_max: float) -> list[Path]:
    order = topic_order(df_macro)
    n_topics = len(order)
    ncols = 1 if n_topics == 1 else 2
    nrows = math.ceil(n_topics / ncols)
    fig_width = 8.8 if ncols == 1 else 15.2
    fig_height = 4.2 if n_topics == 1 else 3.9 * nrows + 1.3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes_list = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for index, corporate_group_id in enumerate(order):
        ax = axes_list[index]
        panel = df_macro[df_macro["corporate_final_merge_group_id"].astype(str) == corporate_group_id].copy()
        label = str(panel["corporate_topic_label"].iloc[0])

        for role in SERIES_ORDER:
            series = panel[panel["series_role"] == role].sort_values("year")
            if series.empty or not has_signal(series, role):
                continue
            style = SERIES_STYLES[role]
            ax.plot(
                series["year"],
                series["relative_share_of_macro_topic_source_docs"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                solid_capstyle="round",
            )

        academic_topics = int(panel.loc[panel["series_role"] == "academic_aggregate", "n_contributing_external_topics"].max() or 0)
        media_topics = int(panel.loc[panel["series_role"] == "media_aggregate", "n_contributing_external_topics"].max() or 0)
        corporate_docs = int(panel.loc[panel["series_role"] == "corporate", "denominator_source_docs_in_macro_topic"].max() or 0)
        ax.set_title(wrapped(label), loc="left", fontweight="bold", pad=14)
        ax.text(
            0.0,
            0.97,
            f"Academic topics = {academic_topics} | Media topics = {media_topics}\nCorporate denominator = {corporate_docs:,} documents",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
            color="#5C5C5C",
        )
        ax.set_xlim(2000, 2025)
        ax.set_ylim(0, y_max)
        ax.set_xticks(YEAR_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_list[n_topics:]:
        fig.delaxes(ax)

    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle(MACRO_TOPIC_LABELS.get(macro_topic, macro_topic), x=0.06, y=0.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.03, "Year", ha="center", fontsize=11)
    fig.text(0.012, 0.5, "Share of source documents in macro topic", va="center", rotation=90, fontsize=11)
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.93])

    png_path = output_dir / f"{macro_topic}_relative_longitudinal.png"
    pdf_path = output_dir / f"{macro_topic}_relative_longitudinal.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    args = parse_args()
    figure_dir = ensure_dir(args.output_dir / "figures")
    set_theme()

    df = pd.read_csv(args.input_dir / "corporate_topic_external_relative_longitudinal.csv")
    df["corporate_final_merge_group_id"] = df["corporate_final_merge_group_id"].astype(str)
    y_max = global_y_max(df)

    outputs: list[Path] = []
    for macro_topic in MACRO_TOPIC_ORDER:
        subset = df[df["macro_topic"] == macro_topic]
        if not subset.empty:
            outputs.extend(render_macro(subset, macro_topic, figure_dir, y_max))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
