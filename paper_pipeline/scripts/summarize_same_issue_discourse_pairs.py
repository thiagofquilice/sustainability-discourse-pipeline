#!/usr/bin/env python3
"""Summarize same-issue/different-discourse-function pair comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT, load_stage2_narratives, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--compare-csv", type=Path, default=None)
    parser.add_argument("--narratives-csv", type=Path, default=None)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_markdown(output_root: Path, analyst: pd.DataFrame) -> None:
    lines = ["# Same Issue / Different Discourse Function", ""]
    lines.append(f"- candidate pairs: `{int(analyst['pair_id'].nunique()) if not analyst.empty else 0}`")
    if not analyst.empty:
        same_issue = int((analyst["same_issue_assessment"] == "same_issue").sum())
        diff_func = int(
            ((analyst["same_issue_assessment"] == "same_issue") & (analyst["discourse_function_relation"] == "different_function")).sum()
        )
        kept = int(analyst["keep_for_discourse_analysis"].fillna(False).sum())
        lines.extend(
            [
                f"- same_issue pairs: `{same_issue}`",
                f"- same_issue + different_function pairs: `{diff_func}`",
                f"- keep_for_discourse_analysis = true: `{kept}`",
                "",
                "## Strongest retained discourse pairs",
            ]
        )
        top = analyst.sort_values(["keep_for_discourse_analysis", "cosine_similarity"], ascending=[False, False]).head(10)
        for row in top.itertuples(index=False):
            lines.append(
                f"- `{row.pair_id}` | `{row.same_issue_assessment}` | `{row.discourse_function_relation}` | "
                f"`{row.function_label_a}` vs `{row.function_label_b}` | sim `{row.cosine_similarity:.3f}`"
            )
    (output_root / "README_same_issue_discourse_function.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidate_csv = args.candidate_csv or (args.output_root / "candidate_pairs_topk.csv")
    compare_csv = args.compare_csv or (args.output_root / "same_issue_discourse_compare.csv")
    narratives = load_stage2_narratives(args.narratives_csv) if args.narratives_csv else load_stage2_narratives()

    candidates = safe_read_csv(candidate_csv)
    compare = safe_read_csv(compare_csv)
    analyst_csv = args.output_root / "same_issue_discourse_analyst_table.csv"
    analyst_xlsx = args.output_root / "same_issue_discourse_analyst_table.xlsx"
    summary_by_dyad_csv = args.output_root / "same_issue_discourse_summary_by_dyad.csv"
    summary_by_macro_csv = args.output_root / "same_issue_discourse_summary_by_macro_topic.csv"
    summary_by_relation_csv = args.output_root / "same_issue_discourse_summary_by_relation.csv"
    summary_by_function_pair_csv = args.output_root / "same_issue_discourse_summary_by_function_pair.csv"

    if candidates.empty or compare.empty or narratives.empty:
        for path in [
            analyst_csv,
            summary_by_dyad_csv,
            summary_by_macro_csv,
            summary_by_relation_csv,
            summary_by_function_pair_csv,
        ]:
            pd.DataFrame().to_csv(path, index=False)
        pd.DataFrame().to_excel(analyst_xlsx, index=False)
        build_markdown(args.output_root, pd.DataFrame())
        write_json(
            args.output_root / "same_issue_discourse_summary_manifest.json",
            {
                "candidate_csv": str(candidate_csv),
                "compare_csv": str(compare_csv),
                "row_counts": {
                    "candidates": int(candidates.shape[0]),
                    "compare_rows": int(compare.shape[0]),
                    "analyst_rows": 0,
                },
                "outputs": {
                    "analyst_table_csv": str(analyst_csv),
                    "analyst_table_xlsx": str(analyst_xlsx),
                    "summary_by_dyad": str(summary_by_dyad_csv),
                    "summary_by_macro_topic": str(summary_by_macro_csv),
                    "summary_by_relation": str(summary_by_relation_csv),
                    "summary_by_function_pair": str(summary_by_function_pair_csv),
                    "readme": str(args.output_root / "README_same_issue_discourse_function.md"),
                },
            },
        )
        print(args.output_root)
        return

    narratives = narratives.loc[narratives["is_narrative_ready"]].copy()
    narr_a = narratives.add_suffix("_a").rename(columns={"subgroup_a": "subgroup_a", "micro_topic_id_a": "microtopic_id_a"})
    narr_b = narratives.add_suffix("_b").rename(columns={"subgroup_b": "subgroup_b", "micro_topic_id_b": "microtopic_id_b"})

    analyst = candidates.merge(compare, on=[
        "pair_id",
        "macro_topic",
        "dyad",
        "source_a",
        "subgroup_a",
        "microtopic_id_a",
        "source_b",
        "subgroup_b",
        "microtopic_id_b",
    ], how="inner")
    analyst = analyst.merge(narr_a, on=["subgroup_a", "microtopic_id_a"], how="left")
    analyst = analyst.merge(narr_b, on=["subgroup_b", "microtopic_id_b"], how="left")
    analyst["is_same_issue_and_different_function"] = (
        (analyst["same_issue_assessment"] == "same_issue")
        & (analyst["discourse_function_relation"] == "different_function")
    )

    summary_by_dyad = (
        analyst.groupby(["dyad", "dyad_label"], dropna=False)
        .agg(
            pair_count=("pair_id", "nunique"),
            same_issue_count=("same_issue_assessment", lambda s: int((pd.Series(s) == "same_issue").sum())),
            different_function_count=(
                "is_same_issue_and_different_function",
                lambda s: int(pd.Series(s).fillna(False).sum()),
            ),
            kept_count=("keep_for_discourse_analysis", lambda s: int(pd.Series(s).fillna(False).sum())),
            mean_similarity=("cosine_similarity", "mean"),
        )
        .reset_index()
    )
    summary_by_macro = (
        analyst.groupby(["macro_topic", "macro_topic_name"], dropna=False)
        .agg(
            pair_count=("pair_id", "nunique"),
            same_issue_count=("same_issue_assessment", lambda s: int((pd.Series(s) == "same_issue").sum())),
            different_function_count=(
                "is_same_issue_and_different_function",
                lambda s: int(pd.Series(s).fillna(False).sum()),
            ),
            kept_count=("keep_for_discourse_analysis", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    summary_by_relation = (
        analyst.groupby(["same_issue_assessment", "discourse_function_relation"], dropna=False)
        .agg(pair_count=("pair_id", "nunique"))
        .reset_index()
        .sort_values("pair_count", ascending=False)
    )
    summary_by_function_pair = (
        analyst.groupby(["function_label_a", "function_label_b"], dropna=False)
        .agg(pair_count=("pair_id", "nunique"))
        .reset_index()
        .sort_values("pair_count", ascending=False)
    )

    analyst.to_csv(analyst_csv, index=False)
    with pd.ExcelWriter(analyst_xlsx) as writer:
        analyst.to_excel(writer, sheet_name="analyst_table", index=False)
        summary_by_dyad.to_excel(writer, sheet_name="by_dyad", index=False)
        summary_by_macro.to_excel(writer, sheet_name="by_macro", index=False)
        summary_by_relation.to_excel(writer, sheet_name="by_relation", index=False)
    summary_by_dyad.to_csv(summary_by_dyad_csv, index=False)
    summary_by_macro.to_csv(summary_by_macro_csv, index=False)
    summary_by_relation.to_csv(summary_by_relation_csv, index=False)
    summary_by_function_pair.to_csv(summary_by_function_pair_csv, index=False)

    build_markdown(args.output_root, analyst)
    write_json(
        args.output_root / "same_issue_discourse_summary_manifest.json",
        {
            "candidate_csv": str(candidate_csv),
            "compare_csv": str(compare_csv),
            "row_counts": {
                "candidates": int(candidates.shape[0]),
                "compare_rows": int(compare.shape[0]),
                "analyst_rows": int(analyst.shape[0]),
            },
            "outputs": {
                "analyst_table_csv": str(analyst_csv),
                "analyst_table_xlsx": str(analyst_xlsx),
                "summary_by_dyad": str(summary_by_dyad_csv),
                "summary_by_macro_topic": str(summary_by_macro_csv),
                "summary_by_relation": str(summary_by_relation_csv),
                "summary_by_function_pair": str(summary_by_function_pair_csv),
                "readme": str(args.output_root / "README_same_issue_discourse_function.md"),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
