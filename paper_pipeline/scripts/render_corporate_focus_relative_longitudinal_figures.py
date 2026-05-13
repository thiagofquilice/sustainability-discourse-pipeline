#!/usr/bin/env python3
"""Render longitudinal relative-share figures for corporate-focus macro topics."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_RESULTS_DIR = PIPELINE_ROOT / "outputs" / "paper_tables" / "corporate_focus_relative_longitudinal_series"
DEFAULT_INPUT_CSV = DEFAULT_RESULTS_DIR / "corporate_topic_external_relative_longitudinal.csv"
DEFAULT_UNPAIRED_INPUT_CSV = DEFAULT_RESULTS_DIR / "unpaired_external_relative_longitudinal.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "figures"

MACRO_TOPIC_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
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
    "corporate": {
        "label": "Corporate",
        "color": "#163A70",
        "linestyle": "-",
        "linewidth": 2.4,
    },
    "academic_aggregate": {
        "label": "Academic aggregate",
        "color": "#C07A1E",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "media_aggregate": {
        "label": "Media aggregate",
        "color": "#1F8A70",
        "linestyle": ":",
        "linewidth": 2.4,
    },
}
UNPAIRED_SOURCE_STYLES = {
    "academic": {
        "label": "Academic unpaired topic",
        "color": "#C07A1E",
        "linestyle": "--",
        "linewidth": 2.2,
    },
    "media": {
        "label": "Media unpaired topic",
        "color": "#1F8A70",
        "linestyle": ":",
        "linewidth": 2.6,
    },
}
YEAR_TICKS = [2000, 2005, 2010, 2015, 2020, 2025]
LEGACY_RELATIVE_COLUMN = "relative_share_of_macro_topic_source_docs"
ANNUAL_PREVALENCE_COLUMN = "annual_document_prevalence"
TITLE_Y = 0.995
LEGEND_Y = 0.94
LAYOUT_TOP = 0.84
PANEL_LETTER_Y = 1.12
PANEL_NOTE_FONT_SIZE = 10.2
PANEL_NOTE_BBOX = {
    "facecolor": "#FFFFFF",
    "alpha": 1.0,
    "edgecolor": "#CFCFCF",
    "linewidth": 0.6,
    "boxstyle": "round,pad=0.32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--unpaired-input-csv", type=Path, default=DEFAULT_UNPAIRED_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--render-legacy",
        action="store_true",
        help="Also refresh the original period-total relative-share figures.",
    )
    return parser.parse_args()


def nice_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def wrapped_label(value: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def panel_letter(index: int) -> str:
    letters = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def add_panel_letter(ax: plt.Axes, index: int) -> None:
    ax.text(
        -0.075,
        PANEL_LETTER_Y,
        panel_letter(index),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def make_panel_grid(
    nrows: int,
    ncols: int,
    fig_width: float,
    fig_height: float,
) -> tuple[plt.Figure, list[plt.Axes], list[plt.Axes]]:
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        nrows=2 * nrows,
        ncols=ncols,
        height_ratios=[1.55, 3.65] * nrows,
        left=0.06,
        right=0.99,
        bottom=0.07,
        top=LAYOUT_TOP,
        hspace=0.08,
        wspace=0.15,
    )
    header_axes: list[plt.Axes] = []
    plot_axes: list[plt.Axes] = []
    first_plot_ax: plt.Axes | None = None
    for row in range(nrows):
        for col in range(ncols):
            header_ax = fig.add_subplot(grid[2 * row, col])
            header_ax.axis("off")
            if first_plot_ax is None:
                plot_ax = fig.add_subplot(grid[2 * row + 1, col])
                first_plot_ax = plot_ax
            else:
                plot_ax = fig.add_subplot(grid[2 * row + 1, col], sharex=first_plot_ax, sharey=first_plot_ax)
            header_axes.append(header_ax)
            plot_axes.append(plot_ax)
    return fig, header_axes, plot_axes


def add_panel_header(header_ax: plt.Axes, index: int, label: str, note: str) -> None:
    header_ax.text(
        -0.075,
        0.68,
        panel_letter(index),
        transform=header_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )
    header_ax.text(
        0.0,
        0.68,
        wrapped_label(label),
        transform=header_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#222222",
    )
    header_ax.text(
        0.012,
        0.03,
        note,
        transform=header_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_NOTE_FONT_SIZE,
        linespacing=1.22,
        color="#363636",
        bbox=PANEL_NOTE_BBOX,
    )


def safe_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def max_int(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    values = frame[column].dropna()
    if values.empty:
        return 0
    return int(values.max())


def compute_global_y_max(frames: list[pd.DataFrame], value_column: str) -> float:
    values: list[float] = []
    for frame in frames:
        if not frame.empty and value_column in frame.columns:
            values.extend(frame[value_column].dropna().astype(float).tolist())
    max_rel = max(values) if values else 0.0
    step = 0.05
    if max_rel <= 0:
        return step
    return max(step, math.ceil(max_rel / step) * step)


def get_macro_topics(df: pd.DataFrame) -> list[str]:
    available = df["macro_topic"].dropna().unique().tolist()
    return [topic for topic in MACRO_TOPIC_ORDER if topic in available]


def get_aligned_topic_order(df_macro: pd.DataFrame) -> list[str]:
    topic_meta = (
        df_macro.groupby("corporate_final_merge_group_id", as_index=False)
        .agg(
            corporate_topic_label=("corporate_topic_label", "first"),
            corporate_topic_size_count=("corporate_topic_size_count", "max"),
            corporate_unique_document_count=("corporate_unique_document_count", "max"),
        )
        .sort_values(
            by=["corporate_topic_size_count", "corporate_unique_document_count", "corporate_topic_label", "corporate_final_merge_group_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    )
    return topic_meta["corporate_final_merge_group_id"].astype(str).tolist()


def get_unpaired_topic_order(df_macro: pd.DataFrame) -> list[str]:
    if df_macro.empty:
        return []
    topic_meta = (
        df_macro.groupby("final_merge_group_id", as_index=False)
        .agg(
            source=("source", "first"),
            topic_label=("topic_label", "first"),
            external_unique_document_count=("external_unique_document_count", "max"),
            external_topic_size_count=("external_topic_size_count", "max"),
        )
        .sort_values(
            by=["external_unique_document_count", "external_topic_size_count", "source", "topic_label", "final_merge_group_id"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
    )
    return topic_meta["final_merge_group_id"].astype(str).tolist()


def series_has_signal(df_series: pd.DataFrame, role: str) -> bool:
    if role == "corporate":
        return True
    if df_series.empty:
        return False
    if "year_document_count" in df_series.columns and df_series["year_document_count"].gt(0).any():
        return True
    if df_series["year_frequency"].gt(0).any():
        return True
    if df_series["n_contributing_external_topics"].gt(0).any():
        return True
    if df_series["contributing_external_group_count"].gt(0).any():
        return True
    return False


def build_legend_handles() -> list[Line2D]:
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


def build_unpaired_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for source in ["academic", "media"]:
        style = UNPAIRED_SOURCE_STYLES[source]
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


def panel_value(panel_df: pd.DataFrame, role: str, column: str) -> int:
    role_df = panel_df[panel_df["series_role"] == role]
    return max_int(role_df, column)


def render_macro_topic_figure(
    df_macro: pd.DataFrame,
    macro_topic: str,
    output_dir: Path,
    y_max: float,
    value_column: str,
    filename_suffix: str,
    y_axis_label: str,
    figure_type: str,
    annotation_note: str,
) -> dict[str, object]:
    topic_order = get_aligned_topic_order(df_macro)
    n_topics = len(topic_order)
    ncols = 1 if n_topics == 1 else 2
    nrows = math.ceil(n_topics / ncols)
    fig_width = 8.8 if ncols == 1 else 15.2
    fig_height = 5.9 if n_topics == 1 else 5.2 * nrows + 2.0
    fig, header_axes, axes_list = make_panel_grid(nrows, ncols, fig_width, fig_height)

    plotted_series_count: dict[str, int] = {role: 0 for role in SERIES_ORDER}

    for ax_idx, corporate_group_id in enumerate(topic_order):
        ax = axes_list[ax_idx]
        header_ax = header_axes[ax_idx]
        panel_df = df_macro[df_macro["corporate_final_merge_group_id"].astype(str) == corporate_group_id].copy()
        label = str(panel_df["corporate_topic_label"].iloc[0])
        academic_topics = panel_value(panel_df, "academic_aggregate", "n_contributing_external_topics")
        media_topics = panel_value(panel_df, "media_aggregate", "n_contributing_external_topics")
        academic_docs = panel_value(panel_df, "academic_aggregate", "series_unique_document_count")
        media_docs = panel_value(panel_df, "media_aggregate", "series_unique_document_count")
        corporate_docs = panel_value(panel_df, "corporate", "series_unique_document_count")

        for role in SERIES_ORDER:
            df_series = panel_df[panel_df["series_role"] == role].sort_values("year")
            if not series_has_signal(df_series, role):
                continue
            style = SERIES_STYLES[role]
            ax.plot(
                df_series["year"],
                df_series[value_column],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                solid_capstyle="round",
                zorder=2,
            )
            plotted_series_count[role] += 1

        add_panel_header(
            header_ax,
            ax_idx,
            label,
            f"Academic topics = {academic_topics} | Media topics = {media_topics}\n"
            f"Academic topic documents = {academic_docs:,} | Media topic documents = {media_docs:,}\n"
            f"Corporate topic documents = {corporate_docs:,}\n"
            f"{annotation_note}",
        )
        ax.set_xlim(2000, 2025)
        ax.set_ylim(0, y_max)
        ax.set_xticks(YEAR_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for header_ax, ax in zip(header_axes[n_topics:], axes_list[n_topics:]):
        fig.delaxes(header_ax)
        fig.delaxes(ax)

    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(MACRO_TOPIC_LABELS.get(macro_topic, macro_topic), x=0.06, y=TITLE_Y, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.03, "Year", ha="center", fontsize=11)
    fig.text(0.012, 0.5, y_axis_label, va="center", rotation=90, fontsize=11)
    png_path = output_dir / f"{macro_topic}_{filename_suffix}.png"
    pdf_path = output_dir / f"{macro_topic}_{filename_suffix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "macro_topic": macro_topic,
        "figure_type": figure_type,
        "panel_count": n_topics,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "plotted_series_count": plotted_series_count,
    }


def render_empty_unpaired_figure(
    macro_topic: str,
    output_dir: Path,
    y_max: float,
    filename_suffix: str,
    y_axis_label: str,
    figure_type: str,
) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    add_panel_letter(ax, 0)
    ax.set_xlim(2000, 2025)
    ax.set_ylim(0, y_max)
    ax.set_xticks(YEAR_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.text(
        0.5,
        0.5,
        "No relevant external topics without a corporate counterpart\nafter final review.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )
    ax.set_title("Unpaired external topics", loc="left", fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(MACRO_TOPIC_LABELS.get(macro_topic, macro_topic), x=0.06, y=TITLE_Y, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.03, "Year", ha="center", fontsize=11)
    fig.text(0.012, 0.5, y_axis_label, va="center", rotation=90, fontsize=11)
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.9])

    png_path = output_dir / f"{macro_topic}_{filename_suffix}.png"
    pdf_path = output_dir / f"{macro_topic}_{filename_suffix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "macro_topic": macro_topic,
        "figure_type": figure_type,
        "panel_count": 0,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }


def render_unpaired_macro_topic_figure(
    df_macro: pd.DataFrame,
    macro_topic: str,
    output_dir: Path,
    y_max: float,
    value_column: str,
    filename_suffix: str,
    y_axis_label: str,
    figure_type: str,
    annotation_note: str,
) -> dict[str, object]:
    if df_macro.empty:
        return render_empty_unpaired_figure(macro_topic, output_dir, y_max, filename_suffix, y_axis_label, figure_type)

    topic_order = get_unpaired_topic_order(df_macro)
    n_topics = len(topic_order)
    ncols = 1 if n_topics == 1 else 2
    nrows = math.ceil(n_topics / ncols)
    fig_width = 8.8 if ncols == 1 else 15.2
    fig_height = 5.9 if n_topics == 1 else 5.2 * nrows + 2.0
    fig, header_axes, axes_list = make_panel_grid(nrows, ncols, fig_width, fig_height)
    plotted_by_source = {"academic": 0, "media": 0}

    for ax_idx, group_id in enumerate(topic_order):
        ax = axes_list[ax_idx]
        header_ax = header_axes[ax_idx]
        panel_df = df_macro[df_macro["final_merge_group_id"].astype(str) == group_id].copy()
        label = str(panel_df["topic_label"].iloc[0]).strip() or str(panel_df["topic_name_original"].iloc[0])
        source = str(panel_df["source"].iloc[0])
        style = UNPAIRED_SOURCE_STYLES.get(source, UNPAIRED_SOURCE_STYLES["academic"])
        docs = max_int(panel_df, "external_unique_document_count")

        panel_df = panel_df.sort_values("year")
        ax.plot(
            panel_df["year"],
            panel_df[value_column],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            zorder=2,
        )
        plotted_by_source[source] = plotted_by_source.get(source, 0) + 1

        add_panel_header(
            header_ax,
            ax_idx,
            label,
            f"Source = {source.title()}\n"
            f"Topic documents = {docs:,}\n"
            f"{annotation_note}",
        )
        ax.set_xlim(2000, 2025)
        ax.set_ylim(0, y_max)
        ax.set_xticks(YEAR_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for header_ax, ax in zip(header_axes[n_topics:], axes_list[n_topics:]):
        fig.delaxes(header_ax)
        fig.delaxes(ax)

    fig.legend(
        handles=build_unpaired_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"{MACRO_TOPIC_LABELS.get(macro_topic, macro_topic)}: unpaired external topics",
        x=0.06,
        y=TITLE_Y,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(0.5, 0.03, "Year", ha="center", fontsize=11)
    fig.text(0.012, 0.5, y_axis_label, va="center", rotation=90, fontsize=11)
    png_path = output_dir / f"{macro_topic}_{filename_suffix}.png"
    pdf_path = output_dir / f"{macro_topic}_{filename_suffix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "macro_topic": macro_topic,
        "figure_type": figure_type,
        "panel_count": n_topics,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "plotted_series_count": plotted_by_source,
    }


def build_manifest(
    aligned_records: list[dict[str, object]],
    unpaired_records: list[dict[str, object]],
    input_csv: Path,
    unpaired_input_csv: Path,
    output_dir: Path,
    y_max: float,
    value_column: str,
    y_axis_unit: str,
    y_axis_label: str,
) -> dict[str, object]:
    return {
        "input_csv": str(input_csv),
        "unpaired_input_csv": str(unpaired_input_csv),
        "output_dir": str(output_dir),
        "global_y_axis_limit": y_max,
        "value_column": value_column,
        "y_axis_unit": y_axis_unit,
        "y_axis_label": y_axis_label,
        "y_axis_display": "percent",
        "year_range": [2000, 2025],
        "aligned_macro_topic_records": aligned_records,
        "unpaired_macro_topic_records": unpaired_records,
        "expected_macro_topics": MACRO_TOPIC_ORDER,
        "file_count": (len(aligned_records) + len(unpaired_records)) * 2,
    }


def main() -> None:
    args = parse_args()
    nice_theme()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned_df = pd.read_csv(args.input_csv)
    unpaired_df = pd.read_csv(args.unpaired_input_csv) if args.unpaired_input_csv.exists() else pd.DataFrame()
    if ANNUAL_PREVALENCE_COLUMN not in aligned_df.columns:
        raise KeyError(
            f"{ANNUAL_PREVALENCE_COLUMN} is missing from {args.input_csv}. "
            "Run build_corporate_focus_relative_longitudinal_series.py first."
        )

    y_axis_label = "Share of source-domain documents in year"
    annual_annotation = "Line denominator = source-domain documents in each year"
    annual_y_max = compute_global_y_max([aligned_df, unpaired_df], ANNUAL_PREVALENCE_COLUMN)

    aligned_records = []
    for macro_topic in get_macro_topics(aligned_df):
        df_macro = aligned_df[aligned_df["macro_topic"] == macro_topic].copy()
        aligned_records.append(
            render_macro_topic_figure(
                df_macro,
                macro_topic,
                args.output_dir,
                annual_y_max,
                ANNUAL_PREVALENCE_COLUMN,
                "annual_document_prevalence",
                y_axis_label,
                "aligned_corporate_anchor_annual_document_prevalence",
                annual_annotation,
            )
        )

    unpaired_records = []
    for macro_topic in MACRO_TOPIC_ORDER:
        if unpaired_df.empty:
            df_macro = pd.DataFrame()
        else:
            df_macro = unpaired_df[unpaired_df["macro_topic"] == macro_topic].copy()
        unpaired_records.append(
            render_unpaired_macro_topic_figure(
                df_macro,
                macro_topic,
                args.output_dir,
                annual_y_max,
                ANNUAL_PREVALENCE_COLUMN,
                "unpaired_annual_document_prevalence",
                y_axis_label,
                "unpaired_external_annual_document_prevalence",
                annual_annotation,
            )
        )

    manifest = build_manifest(
        aligned_records,
        unpaired_records,
        args.input_csv,
        args.unpaired_input_csv,
        args.output_dir,
        annual_y_max,
        ANNUAL_PREVALENCE_COLUMN,
        "annual_document_prevalence",
        y_axis_label,
    )
    manifest_path = args.output_dir / "annual_document_prevalence_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.render_legacy:
        legacy_y_max = compute_global_y_max([aligned_df, unpaired_df], LEGACY_RELATIVE_COLUMN)
        legacy_records = []
        for macro_topic in get_macro_topics(aligned_df):
            df_macro = aligned_df[aligned_df["macro_topic"] == macro_topic].copy()
            legacy_records.append(
                render_macro_topic_figure(
                    df_macro,
                    macro_topic,
                    args.output_dir,
                    legacy_y_max,
                    LEGACY_RELATIVE_COLUMN,
                    "relative_longitudinal",
                    "Share of source documents in macro topic",
                    "aligned_corporate_anchor",
                    "Line denominator = full-period source-domain documents",
                )
            )
        legacy_unpaired_records = []
        for macro_topic in MACRO_TOPIC_ORDER:
            df_macro = pd.DataFrame() if unpaired_df.empty else unpaired_df[unpaired_df["macro_topic"] == macro_topic].copy()
            legacy_unpaired_records.append(
                render_unpaired_macro_topic_figure(
                    df_macro,
                    macro_topic,
                    args.output_dir,
                    legacy_y_max,
                    LEGACY_RELATIVE_COLUMN,
                    "unpaired_relative_longitudinal",
                    "Share of source documents in macro topic",
                    "unpaired_external",
                    "Line denominator = full-period source-domain documents",
                )
            )
        legacy_manifest = build_manifest(
            legacy_records,
            legacy_unpaired_records,
            args.input_csv,
            args.unpaired_input_csv,
            args.output_dir,
            legacy_y_max,
            LEGACY_RELATIVE_COLUMN,
            "share_of_full_period_macro_topic_source_documents",
            "Share of source documents in macro topic",
        )
        legacy_manifest_path = args.output_dir / "relative_longitudinal_figures_manifest.json"
        legacy_manifest_path.write_text(json.dumps(legacy_manifest, indent=2), encoding="utf-8")

    for record in aligned_records + unpaired_records:
        print(record["png_path"])
        print(record["pdf_path"])
    print(manifest_path)


if __name__ == "__main__":
    main()
