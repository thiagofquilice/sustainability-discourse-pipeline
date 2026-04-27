#!/usr/bin/env python3
"""Join cross-source precedence pairs with completed LLM evolution narratives."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import (
    EVOLUTION_NARRATIVES_PATH,
    OUTPUT_ROOT,
    configure_logging,
    load_stage2_narratives,
    write_json,
)


PHASE_FIELDS = [
    ("phase_1_years", "phase_1_summary"),
    ("phase_2_years", "phase_2_summary"),
    ("phase_3_years", "phase_3_summary"),
    ("phase_4_years", "phase_4_summary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--join-ready-csv", type=Path, default=None)
    parser.add_argument("--narratives-csv", type=Path, default=EVOLUTION_NARRATIVES_PATH)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_phase_range(text: object) -> tuple[int | None, int | None]:
    if not isinstance(text, str) or not text.strip() or text.strip().lower() == "none":
        return None, None
    match = re.search(r"(\d{4}).*?(\d{4})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(\d{4})", text)
    if match:
        year = int(match.group(1))
        return year, year
    return None, None


def narrative_span(row: pd.Series, suffix: str) -> tuple[int | None, int | None]:
    starts: list[int] = []
    ends: list[int] = []
    for phase_field, _ in PHASE_FIELDS:
        start, end = parse_phase_range(row.get(f"{phase_field}_{suffix}"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def alignment_note(row: pd.Series) -> str:
    if not row.get("both_have_narratives", False):
        return "missing narrative on at least one side"
    if row.get("same_evolution_pattern", False):
        return "both sides share the same labeled evolution pattern"
    sparse_a = "sparse" in str(row.get("evidence_note_a", "")).lower()
    sparse_b = "sparse" in str(row.get("evidence_note_b", "")).lower()
    if sparse_a or sparse_b:
        return "at least one side reports sparse evidence"
    return "patterns differ across the two discourse narratives"


def build_markdown(output_root: Path, join_ready: pd.DataFrame, enriched: pd.DataFrame) -> None:
    deterministic_lines = [
        "## Deterministic findings",
        f"- pair count in join-ready table: `{int(join_ready.shape[0])}`",
        f"- pairs with balanced viability: `{int(join_ready['balanced_is_viable'].fillna(False).sum()) if 'balanced_is_viable' in join_ready.columns else 0}`",
        f"- pairs with narrative rows on both sides: `{int(join_ready['both_have_narrative_rows'].fillna(False).sum()) if 'both_have_narrative_rows' in join_ready.columns else 0}`",
        f"- pairs with both narratives: `{int(join_ready['both_have_narratives'].fillna(False).sum()) if 'both_have_narratives' in join_ready.columns else 0}`",
    ]
    if not join_ready.empty and "precedence_type_label" in join_ready.columns:
        counts = (
            join_ready["precedence_type_label"]
            .fillna("missing")
            .value_counts()
            .rename_axis("precedence_type_label")
            .reset_index(name="pair_count")
        )
        deterministic_lines.extend(
            [f"- `{row.precedence_type_label}`: `{int(row.pair_count)}`" for row in counts.itertuples(index=False)]
        )

    enriched_lines = [
        "## LLM-enriched interpretation",
        f"- enriched pair rows with two narratives: `{int(enriched.shape[0])}`",
    ]
    if not enriched.empty:
        top_rows = enriched.sort_values(["cosine_similarity", "best_correlation_docs"], ascending=[False, False]).head(8)
        enriched_lines.append("- strongest enriched pairs:")
        enriched_lines.extend(
            [
                f"  - `{row.pair_id}` | `{row.precedence_type_label}` | similarity `{row.cosine_similarity:.3f}` | lag `{int(row.best_lag_docs):+d}`"
                for row in top_rows.itertuples(index=False)
            ]
        )
    else:
        enriched_lines.append("- no fully enriched pairs yet; rerun this step after Stage 2 completes.")

    text = "\n".join(
        [
            "# Cross-Source Pairs with LLM Evolution Enrichment",
            "",
            *deterministic_lines,
            "",
            *enriched_lines,
            "",
        ]
    )
    (output_root / "README_microtopic_cross_source_pairs_threshold_065.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    join_ready_csv = args.join_ready_csv or (args.output_root / "pair_join_ready_table.csv")
    join_ready = safe_read_csv(join_ready_csv)
    narratives = load_stage2_narratives(args.narratives_csv)

    enriched_csv = args.output_root / "llm_enriched_pair_table.csv"
    enriched_xlsx = args.output_root / "llm_enriched_pair_table.xlsx"

    if join_ready.empty or narratives.empty:
        pd.DataFrame().to_csv(enriched_csv, index=False)
        pd.DataFrame().to_excel(enriched_xlsx, index=False)
        build_markdown(args.output_root, join_ready, pd.DataFrame())
        write_json(
            args.output_root / "llm_enrichment_manifest.json",
            {
                "join_ready_csv": str(join_ready_csv),
                "narratives_csv": str(args.narratives_csv),
                "row_counts": {
                    "join_ready_rows": int(join_ready.shape[0]),
                    "narrative_rows": int(narratives.shape[0]),
                    "enriched_rows": 0,
                },
                "outputs": {
                    "llm_enriched_pair_table": str(enriched_csv),
                    "llm_enriched_pair_table_xlsx": str(enriched_xlsx),
                    "readme": str(args.output_root / "README_microtopic_cross_source_pairs_threshold_065.md"),
                },
            },
        )
        print(args.output_root)
        return

    narratives = narratives.copy()

    narr_a = narratives.add_suffix("_a").rename(
        columns={"subgroup_a": "subgroup_a", "micro_topic_id_a": "microtopic_id_a"}
    )
    narr_b = narratives.add_suffix("_b").rename(
        columns={"subgroup_b": "subgroup_b", "micro_topic_id_b": "microtopic_id_b"}
    )

    enriched = join_ready.merge(narr_a, on=["subgroup_a", "microtopic_id_a"], how="left")
    enriched = enriched.merge(narr_b, on=["subgroup_b", "microtopic_id_b"], how="left")
    enriched["has_narrative_row_a"] = enriched["subgroup_a"].notna() & enriched["microtopic_id_a"].notna() & enriched["topic_name_original_a"].notna()
    enriched["has_narrative_row_b"] = enriched["subgroup_b"].notna() & enriched["microtopic_id_b"].notna() & enriched["topic_name_original_b"].notna()
    enriched["both_have_narrative_rows"] = enriched["has_narrative_row_a"] & enriched["has_narrative_row_b"]
    enriched["has_narrative_ready_a"] = enriched["is_narrative_ready_a"].fillna(False).astype(bool)
    enriched["has_narrative_ready_b"] = enriched["is_narrative_ready_b"].fillna(False).astype(bool)
    enriched["has_narrative_a"] = enriched["has_narrative_ready_a"]
    enriched["has_narrative_b"] = enriched["has_narrative_ready_b"]
    enriched["both_have_narratives"] = enriched["has_narrative_ready_a"] & enriched["has_narrative_ready_b"]
    enriched = enriched.loc[enriched["both_have_narratives"]].copy()

    if enriched.empty:
        pd.DataFrame().to_csv(enriched_csv, index=False)
        pd.DataFrame().to_excel(enriched_xlsx, index=False)
        build_markdown(args.output_root, join_ready, enriched)
        write_json(
            args.output_root / "llm_enrichment_manifest.json",
            {
                "join_ready_csv": str(join_ready_csv),
                "narratives_csv": str(args.narratives_csv),
                "row_counts": {
                    "join_ready_rows": int(join_ready.shape[0]),
                    "narrative_rows": int(narratives.shape[0]),
                    "enriched_rows": 0,
                },
                "outputs": {
                    "llm_enriched_pair_table": str(enriched_csv),
                    "llm_enriched_pair_table_xlsx": str(enriched_xlsx),
                    "readme": str(args.output_root / "README_microtopic_cross_source_pairs_threshold_065.md"),
                },
            },
        )
        print(args.output_root)
        return

    enriched["same_evolution_pattern"] = (
        enriched["evolution_pattern_a"].fillna("").astype(str).str.strip()
        == enriched["evolution_pattern_b"].fillna("").astype(str).str.strip()
    )
    spans_a = enriched.apply(lambda row: narrative_span(row, suffix="a"), axis=1)
    enriched["source_a_first_phase_start"] = [item[0] for item in spans_a]
    enriched["source_a_last_phase_end"] = [item[1] for item in spans_a]
    spans_b = enriched.apply(lambda row: narrative_span(row, suffix="b"), axis=1)
    enriched["source_b_first_phase_start"] = [item[0] for item in spans_b]
    enriched["source_b_last_phase_end"] = [item[1] for item in spans_b]
    enriched["narrative_time_span_overlap"] = enriched.apply(
        lambda row: (
            max(
                0,
                min(
                    row["source_a_last_phase_end"] if pd.notna(row["source_a_last_phase_end"]) else -1,
                    row["source_b_last_phase_end"] if pd.notna(row["source_b_last_phase_end"]) else -1,
                )
                - max(
                    row["source_a_first_phase_start"] if pd.notna(row["source_a_first_phase_start"]) else 9999,
                    row["source_b_first_phase_start"] if pd.notna(row["source_b_first_phase_start"]) else 9999,
                )
                + 1,
            )
        ),
        axis=1,
    )
    enriched["narrative_alignment_note"] = enriched.apply(alignment_note, axis=1)

    enriched.to_csv(enriched_csv, index=False)
    with pd.ExcelWriter(enriched_xlsx) as writer:
        enriched.to_excel(writer, sheet_name="llm_enriched_pairs", index=False)

    build_markdown(args.output_root, join_ready, enriched)
    write_json(
        args.output_root / "llm_enrichment_manifest.json",
        {
            "join_ready_csv": str(join_ready_csv),
            "narratives_csv": str(args.narratives_csv),
            "row_counts": {
                "join_ready_rows": int(join_ready.shape[0]),
                "narrative_rows": int(narratives.shape[0]),
                "enriched_rows": int(enriched.shape[0]),
            },
            "outputs": {
                "llm_enriched_pair_table": str(enriched_csv),
                "llm_enriched_pair_table_xlsx": str(enriched_xlsx),
                "readme": str(args.output_root / "README_microtopic_cross_source_pairs_threshold_065.md"),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
