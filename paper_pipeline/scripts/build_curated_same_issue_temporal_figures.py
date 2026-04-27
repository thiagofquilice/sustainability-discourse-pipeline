#!/usr/bin/env python3
"""Build grouped temporal figures for a curated set of same-issue discourse pairs."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DISCOURSE_ROOT = Path(
    "paper_pipeline/outputs/"
    "microtopic_same_issue_discourse_function_top2_threshold055"
)
DEFAULT_PAIR_ROOT = Path(
    "paper_pipeline/outputs/"
    "microtopic_cross_source_pairs_threshold_065"
)

CURATED_PAIR_ORDER = [
    "media__corporate__T4__media_1__corporate_0",
    "academic__corporate__T3__academic_7__corporate_4",
    "media__corporate__T1__media_0__corporate_3",
    "academic__corporate__T2__academic_0__corporate_4",
    "media__corporate__T4__media_0__corporate_0",
    "academic__corporate__T1__academic_3__corporate_3",
    "academic__corporate__T4__academic_2__corporate_0",
    "media__corporate__T1__media_2__corporate_4",
    "academic__corporate__T3__academic_3__corporate_4",
    "media__corporate__T5__media_0__corporate_1",
    "academic__corporate__T6__academic_9__corporate_8",
    "academic__corporate__T1__academic_6__corporate_2",
]


def pick_column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discourse-root", type=Path, default=DEFAULT_DISCOURSE_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pair-ids-csv", type=Path, default=None)
    return parser.parse_args()


def short_label(value: str, width: int = 36) -> str:
    text = " ".join(str(value).replace("_", " ").split())
    return textwrap.fill(text, width=width)


def format_function_label(value: str) -> str:
    return " ".join(str(value).split("_"))


def load_curated_pair_ids(path: Path | None) -> list[str]:
    if path is None:
        return CURATED_PAIR_ORDER
    pair_frame = pd.read_csv(path)
    if "pair_id" not in pair_frame.columns:
        raise ValueError(f"Expected a pair_id column in {path}")
    pair_ids = [str(value).strip() for value in pair_frame["pair_id"].tolist() if str(value).strip()]
    if not pair_ids:
        raise ValueError(f"No pair_id values found in {path}")
    return pair_ids


def build_pair_series(
    pair_row: pd.Series,
    annual_counts: pd.DataFrame,
    denominators: pd.DataFrame,
) -> pd.DataFrame:
    source_a = str(pair_row["source_a"])
    source_b = str(pair_row["source_b"])
    assigned_label = str(pair_row["macro_topic"])
    micro_a = int(pair_row["microtopic_id_a"])
    micro_b = int(pair_row["microtopic_id_b"])

    den_a = denominators[
        (denominators["source"] == source_a) & (denominators["assigned_label"] == assigned_label)
    ].copy()
    den_b = denominators[
        (denominators["source"] == source_b) & (denominators["assigned_label"] == assigned_label)
    ].copy()
    years = sorted(set(den_a["year"].tolist()) | set(den_b["year"].tolist()))
    if not years:
        raise ValueError(f"No denominator years found for {pair_row['pair_id']}")

    base = pd.DataFrame({"year": years})
    base["pair_id"] = pair_row["pair_id"]
    base["macro_topic"] = assigned_label
    base["dyad"] = pair_row["dyad"]
    base["source_a"] = source_a
    base["source_b"] = source_b

    annual_a = annual_counts[
        (annual_counts["source"] == source_a)
        & (annual_counts["assigned_label"] == assigned_label)
        & (annual_counts["micro_topic_id"] == micro_a)
    ][["year", "microtopic_doc_n", "microtopic_chunk_n"]].rename(
        columns={
            "microtopic_doc_n": "source_a_doc_numerator",
            "microtopic_chunk_n": "source_a_chunk_numerator",
        }
    )
    annual_b = annual_counts[
        (annual_counts["source"] == source_b)
        & (annual_counts["assigned_label"] == assigned_label)
        & (annual_counts["micro_topic_id"] == micro_b)
    ][["year", "microtopic_doc_n", "microtopic_chunk_n"]].rename(
        columns={
            "microtopic_doc_n": "source_b_doc_numerator",
            "microtopic_chunk_n": "source_b_chunk_numerator",
        }
    )

    den_a = den_a[["year", "yes_doc_n", "yes_chunk_n"]].rename(
        columns={"yes_doc_n": "source_a_doc_denominator", "yes_chunk_n": "source_a_chunk_denominator"}
    )
    den_b = den_b[["year", "yes_doc_n", "yes_chunk_n"]].rename(
        columns={"yes_doc_n": "source_b_doc_denominator", "yes_chunk_n": "source_b_chunk_denominator"}
    )

    base = base.merge(annual_a, on="year", how="left")
    base = base.merge(annual_b, on="year", how="left")
    base = base.merge(den_a, on="year", how="left")
    base = base.merge(den_b, on="year", how="left")

    for column in [
        "source_a_doc_numerator",
        "source_b_doc_numerator",
        "source_a_chunk_numerator",
        "source_b_chunk_numerator",
        "source_a_doc_denominator",
        "source_b_doc_denominator",
        "source_a_chunk_denominator",
        "source_b_chunk_denominator",
    ]:
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)

    base["source_a_share_docs"] = base["source_a_doc_numerator"] / base["source_a_doc_denominator"].where(
        base["source_a_doc_denominator"] > 0
    )
    base["source_b_share_docs"] = base["source_b_doc_numerator"] / base["source_b_doc_denominator"].where(
        base["source_b_doc_denominator"] > 0
    )
    base["source_a_share_chunks"] = base["source_a_chunk_numerator"] / base["source_a_chunk_denominator"].where(
        base["source_a_chunk_denominator"] > 0
    )
    base["source_b_share_chunks"] = base["source_b_chunk_numerator"] / base["source_b_chunk_denominator"].where(
        base["source_b_chunk_denominator"] > 0
    )
    return base


def plot_group_figure(group_df: pd.DataFrame, series_df: pd.DataFrame, output_path: Path) -> None:
    pair_count = len(group_df)
    ncols = 2
    nrows = math.ceil(pair_count / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.3 * nrows), constrained_layout=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (_, row) in zip(axes_list, group_df.iterrows()):
        pair_series = series_df[series_df["pair_id"] == row["pair_id"]].copy()
        years = pair_series["year"].astype(int)
        share_a = pair_series["source_a_share_docs"].fillna(0.0) * 100.0
        share_b = pair_series["source_b_share_docs"].fillna(0.0) * 100.0

        ax.plot(years, share_a, color="#1f77b4", linewidth=2.2, marker="o", markersize=3, label=row["source_a"])
        ax.plot(years, share_b, color="#d62728", linewidth=2.2, marker="o", markersize=3, label=row["source_b"])

        ymax = max(float(share_a.max()), float(share_b.max()), 1.0)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_xlim(years.min(), years.max())
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel("Doc share (%)")

        title = f"{row['macro_topic']} | sim {row['cosine_similarity']:.3f}"
        subtitle = f"{short_label(row['topic_label_refined_a'], 22)}\nvs\n{short_label(row['topic_label_refined_b'], 22)}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)

        note = (
            f"{row['source_a']}: {format_function_label(row['function_label_a'])}\n"
            f"{row['source_b']}: {format_function_label(row['function_label_b'])}"
        )
        ax.text(
            0.01,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
        )

    for ax in axes_list[pair_count:]:
        ax.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"Temporal Evolution of Curated Same-Issue / Different-Function Pairs: {group_df['dyad_label'].iloc[0]}",
        fontsize=14,
        y=1.04,
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_single_pair_figure(row: pd.Series, pair_series: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    years = pair_series["year"].astype(int)
    share_a = pair_series["source_a_share_docs"].fillna(0.0) * 100.0
    share_b = pair_series["source_b_share_docs"].fillna(0.0) * 100.0

    ax.plot(years, share_a, color="#1f77b4", linewidth=2.4, marker="o", markersize=4, label=row["source_a"])
    ax.plot(years, share_b, color="#d62728", linewidth=2.4, marker="o", markersize=4, label=row["source_b"])

    ymax = max(float(share_a.max()), float(share_b.max()), 1.0)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlim(years.min(), years.max())
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Doc share (%)")
    ax.set_xlabel("Year")

    title = f"{row['macro_topic']} | {row['dyad_label']} | sim {row['cosine_similarity']:.3f}"
    subtitle = (
        f"{short_label(row['topic_label_refined_a'], 34)}\n"
        f"vs\n"
        f"{short_label(row['topic_label_refined_b'], 34)}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)

    note = (
        f"{row['source_a']}: {format_function_label(row['function_label_a'])}\n"
        f"{row['source_b']}: {format_function_label(row['function_label_b'])}"
    )
    ax.text(
        0.01,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.28"},
    )
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.discourse_root / "curated_temporal_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    curated_pair_order = load_curated_pair_ids(args.pair_ids_csv)

    analyst_table = pd.read_csv(args.discourse_root / "same_issue_discourse_analyst_table.csv")
    source_a_col = pick_column(analyst_table, "source_a", "source_a_x", "source_a_y")
    source_b_col = pick_column(analyst_table, "source_b", "source_b_x", "source_b_y")
    annual_counts = pd.read_csv(args.pair_root / "microtopic_annual_counts.csv")
    denominators = pd.read_csv(args.pair_root / "source_topic_year_denominators.csv")

    curated = analyst_table[analyst_table["pair_id"].isin(curated_pair_order)].copy()
    curated = curated.rename(columns={source_a_col: "source_a", source_b_col: "source_b"})
    curated["selection_order"] = curated["pair_id"].map({pair_id: idx for idx, pair_id in enumerate(curated_pair_order)})
    curated = curated.sort_values("selection_order").reset_index(drop=True)
    if len(curated) != len(curated_pair_order):
        missing = sorted(set(curated_pair_order) - set(curated["pair_id"]))
        raise ValueError(f"Missing curated pairs in analyst table: {missing}")

    series_frames = [build_pair_series(row, annual_counts, denominators) for _, row in curated.iterrows()]
    curated_series = pd.concat(series_frames, ignore_index=True)

    curated_table = curated[
        [
            "selection_order",
            "pair_id",
            "macro_topic",
            "macro_topic_name",
            "dyad",
            "dyad_label",
            "cosine_similarity",
            "source_a",
            "source_b",
            "topic_label_refined_a",
            "topic_label_refined_b",
            "function_label_a",
            "function_label_b",
            "difference_summary",
        ]
    ].copy()
    curated_table["figure_group"] = curated_table["dyad"]

    curated_table.to_csv(output_dir / "curated_12_pairs_for_paper.csv", index=False)
    curated_series.to_csv(output_dir / "curated_12_pairs_temporal_series.csv", index=False)

    with pd.ExcelWriter(output_dir / "curated_12_pairs_for_paper.xlsx", engine="openpyxl") as writer:
        curated_table.to_excel(writer, sheet_name="selected_pairs", index=False)
        curated_series.to_excel(writer, sheet_name="temporal_series", index=False)

    individual_dir = output_dir / "individual_pair_figures"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for dyad, group_df in curated.sort_values("selection_order").groupby("dyad", sort=False):
        safe_name = dyad.replace("__", "_")
        plot_group_figure(group_df, curated_series, output_dir / f"{safe_name}_curated_temporal_evolution")

    for _, row in curated.sort_values("selection_order").iterrows():
        pair_series = curated_series[curated_series["pair_id"] == row["pair_id"]].copy()
        pair_slug = row["pair_id"].replace("__", "_")
        prefix = f"{int(row['selection_order']) + 1:02d}"
        plot_single_pair_figure(row, pair_series, individual_dir / f"{prefix}_{pair_slug}")

    readme = output_dir / "README_curated_temporal_figures.md"
    readme.write_text(
        "\n".join(
            [
                "# Curated Temporal Figures",
                "",
                "This directory contains grouped temporal figures for the 12 curated same-issue / different-function pairs.",
                "",
                "Method:",
                "- Pair selection uses the curated list discussed for paper development.",
                *([f"- Pair IDs were loaded from `{args.pair_ids_csv}`."] if args.pair_ids_csv else []),
                "- Temporal values are document-share trajectories within `source × macro-topic × year`.",
                "- Numerators come from `microtopic_annual_counts.csv`.",
                "- Denominators come from `source_topic_year_denominators.csv`.",
                "",
                "Outputs:",
                "- `academic_corporate_curated_temporal_evolution.png/.pdf`",
                "- `media_corporate_curated_temporal_evolution.png/.pdf`",
                "- `individual_pair_figures/*.png/.pdf`",
                "- `curated_12_pairs_for_paper.csv/.xlsx`",
                "- `curated_12_pairs_temporal_series.csv`",
            ]
        ),
        encoding="utf-8",
    )

    print(output_dir)


if __name__ == "__main__":
    main()
