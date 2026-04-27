#!/usr/bin/env python3
"""Build a hybrid corporate-focus review workbook for human comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from build_corporate_focus_review import build_pair_evidence, load_group_catalog
from cross_source_microtopic_common import MACRO_TOPIC_ORDER, SOURCE_ORDER, configure_logging
from microtopic_posthoc_merge_common import (
    MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED,
    ensure_directory,
    sanitize_frame_for_excel,
    workbook_autofit,
    write_json,
)


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_BUNDLE_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_stage12_colab_drive_with_overrides"
DEFAULT_REVIEW_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_review_with_overrides"
DEFAULT_STAGE12_INPUT_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_stage12_input_with_overrides"
DEFAULT_PAIR_ROOT = PIPELINE_ROOT / "outputs" / "microtopic_cross_source_pairs_corporate_focus"
DEFAULT_GLOBAL_LLM_PAIR_ROOT = PIPELINE_ROOT / "outputs" / "microtopic_cross_source_pairs_threshold_065"


PHASE_NUMBERS = [1, 2, 3, 4]
NARRATIVE_CORE_FIELDS = [
    "topic_name_original",
    "topic_label_refined",
    "overall_summary",
    "evolution_pattern",
    "evidence_note",
]
PHASE_YEAR_FIELDS = [f"phase_{idx}_years" for idx in PHASE_NUMBERS]
PHASE_SUMMARY_FIELDS = [f"phase_{idx}_summary" for idx in PHASE_NUMBERS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--stage12-input-root", type=Path, default=DEFAULT_STAGE12_INPUT_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--global-llm-pair-root", type=Path, default=DEFAULT_GLOBAL_LLM_PAIR_ROOT)
    parser.add_argument("--merged-micro-root", type=Path, default=MERGED_MICRO_ROOT_MULTIASPECT_REVIEWED)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def choose_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(path) for path in paths))


def parse_phase_range(text: object) -> tuple[int | None, int | None]:
    if not isinstance(text, str):
        return None, None
    value = text.strip()
    if not value:
        return None, None
    if "-" in value:
        left, right = value.split("-", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            return int(left.strip()), int(right.strip())
    if value.isdigit():
        year = int(value)
        return year, year
    return None, None


def narrative_span(row: pd.Series, prefix: str) -> tuple[int | None, int | None]:
    starts: list[int] = []
    ends: list[int] = []
    for field in PHASE_YEAR_FIELDS:
        start, end = parse_phase_range(row.get(f"{field}_{prefix}"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def narrative_overlap_years(row: pd.Series) -> int | None:
    start_a, end_a = narrative_span(row, "noncorporate")
    start_b, end_b = narrative_span(row, "corporate")
    if None in {start_a, end_a, start_b, end_b}:
        return None
    overlap = min(end_a, end_b) - max(start_a, start_b) + 1
    return max(0, overlap)


def build_alignment_note(row: pd.Series) -> str:
    if row.get("inclusion_reason") == "manual_override_from_excluded_review":
        if not row.get("best_dyad"):
            return "Manual override: no direct dyad above threshold; compare summaries and phase ranges manually."
        return "Manual override retained with a corporate anchor; compare summaries and phase ranges manually."

    overlap = narrative_overlap_years(row)
    precedence = str(row.get("precedence_type_label") or "").strip()
    if overlap is None:
        return "Narratives available on both sides; compare summaries manually."
    if precedence:
        return f"Narrative spans overlap for {overlap} year(s); precedence signal: {precedence}."
    return f"Narrative spans overlap for {overlap} year(s); no precedence label available."


def narrative_digest(row: pd.Series) -> str:
    lines = []
    overall = str(row.get("overall_summary") or "").strip()
    if overall:
        lines.append(overall)
    for idx in PHASE_NUMBERS:
        years = str(row.get(f"phase_{idx}_years") or "").strip()
        summary = str(row.get(f"phase_{idx}_summary") or "").strip()
        if years or summary:
            lines.append(f"Phase {idx} ({years}): {summary}".strip())
    return "\n".join(lines)


def load_selected_with_narratives(bundle_root: Path) -> pd.DataFrame:
    selected_path = bundle_root / "data" / "selected_micro_topics.csv"
    narratives_path = (
        bundle_root
        / "colab_outputs"
        / "phase_02_evolution_summaries"
        / "micro_topic_evolution_narratives.csv"
    )
    selected = read_csv_required(selected_path)
    narratives = read_csv_required(narratives_path)
    merged = selected.merge(
        narratives,
        on=["subgroup", "source", "assigned_label", "micro_topic_id", "topic_name_original"],
        how="left",
        validate="one_to_one",
    )
    missing = merged["topic_label_refined"].isna().sum()
    if missing:
        raise SystemExit(f"Missing Stage 2 narratives for {missing} selected group(s) in {narratives_path}")
    merged["narrative_digest"] = merged.apply(narrative_digest, axis=1)
    return merged


def load_review_inputs(review_root: Path, stage12_input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included_corporate = read_csv_required(
        choose_existing_path(
            review_root / "included_corporate_groups.csv",
            stage12_input_root / "included_corporate_groups.csv",
        )
    )
    included_noncorporate = read_csv_required(
        choose_existing_path(
            review_root / "included_noncorporate_groups.csv",
            stage12_input_root / "included_noncorporate_groups.csv",
        )
    )
    manual_override = pd.read_csv(
        choose_existing_path(
            review_root / "manual_override_noncorporate_groups.csv",
            stage12_input_root / "manual_override_noncorporate_groups.csv",
        )
    ) if (
        (review_root / "manual_override_noncorporate_groups.csv").exists()
        or (stage12_input_root / "manual_override_noncorporate_groups.csv").exists()
    ) else pd.DataFrame(columns=included_noncorporate.columns)
    return included_corporate, included_noncorporate, manual_override


def build_pair_metadata(pair_root: Path, global_llm_pair_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    precedence_path = pair_root / "pair_precedence_classification.csv"
    join_ready_path = pair_root / "pair_join_ready_table.csv"
    global_llm_path = global_llm_pair_root / "llm_enriched_pair_table.csv"

    precedence = read_csv_required(precedence_path) if precedence_path.exists() else pd.DataFrame()
    join_ready = read_csv_required(join_ready_path) if join_ready_path.exists() else pd.DataFrame()
    global_llm = read_csv_required(global_llm_path) if global_llm_path.exists() else pd.DataFrame()

    if not precedence.empty and not join_ready.empty:
        precedence = precedence.merge(
            join_ready[
                [
                    "pair_id",
                    "lenient_is_viable",
                    "balanced_is_viable",
                    "both_have_narrative_rows",
                    "both_have_narratives",
                ]
            ],
            on="pair_id",
            how="left",
        )

    if not global_llm.empty:
        precedence = precedence.merge(
            global_llm[["pair_id", "narrative_alignment_note"]].rename(
                columns={"narrative_alignment_note": "global_narrative_alignment_note"}
            ),
            on="pair_id",
            how="left",
        )
    return precedence, join_ready


def prepare_selected_lookup(selected_with_narratives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_groups = selected_with_narratives.copy()
    corporate = all_groups.loc[all_groups["source"] == "corporate"].copy()
    corporate_lookup = corporate.rename(
        columns={
            "subgroup": "subgroup_corporate",
            "source": "source_corporate",
            "assigned_label": "assigned_label_corporate",
            "macro_topic_name": "macro_topic_name_corporate",
            "micro_topic_id": "micro_topic_id_corporate",
            "topic_name_original": "topic_name_original_corporate",
            "topic_label_refined": "topic_label_refined_corporate",
            "overall_summary": "overall_summary_corporate",
            "evolution_pattern": "evolution_pattern_corporate",
            "evidence_note": "evidence_note_corporate",
            "final_merge_group_id": "final_merge_group_id_corporate",
            "group_size": "group_size_corporate",
            "topic_size": "topic_size_corporate",
            "narrative_digest": "narrative_digest_corporate",
            "active_year_count": "active_year_count_corporate",
            "active_year_min": "active_year_min_corporate",
            "active_year_max": "active_year_max_corporate",
            "overall_keywords": "overall_keywords_corporate",
            "share_of_non_outlier": "share_of_non_outlier_corporate",
            "cumulative_share": "cumulative_share_corporate",
        }
    )
    for field in PHASE_YEAR_FIELDS + PHASE_SUMMARY_FIELDS:
        corporate_lookup = corporate_lookup.rename(columns={field: f"{field}_corporate"})
    return all_groups, corporate_lookup


def build_master_review(
    included_noncorporate: pd.DataFrame,
    selected_with_narratives: pd.DataFrame,
    corporate_lookup: pd.DataFrame,
    precedence: pd.DataFrame,
) -> pd.DataFrame:
    noncorp_lookup = selected_with_narratives.rename(
        columns={
            "subgroup": "subgroup_noncorporate",
            "source": "source_noncorporate",
            "assigned_label": "assigned_label_noncorporate",
            "macro_topic_name": "macro_topic_name_noncorporate",
            "micro_topic_id": "micro_topic_id_noncorporate",
            "topic_name_original": "topic_name_original_noncorporate",
            "topic_label_refined": "topic_label_refined_noncorporate",
            "overall_summary": "overall_summary_noncorporate",
            "evolution_pattern": "evolution_pattern_noncorporate",
            "evidence_note": "evidence_note_noncorporate",
            "final_merge_group_id": "final_merge_group_id_noncorporate",
            "group_size": "group_size_noncorporate",
            "topic_size": "topic_size_noncorporate",
            "narrative_digest": "narrative_digest_noncorporate",
            "active_year_count": "active_year_count_noncorporate",
            "active_year_min": "active_year_min_noncorporate",
            "active_year_max": "active_year_max_noncorporate",
            "overall_keywords": "overall_keywords_noncorporate",
            "share_of_non_outlier": "share_of_non_outlier_noncorporate",
            "cumulative_share": "cumulative_share_noncorporate",
        }
    )
    for field in PHASE_YEAR_FIELDS + PHASE_SUMMARY_FIELDS:
        noncorp_lookup = noncorp_lookup.rename(columns={field: f"{field}_noncorporate"})

    master = included_noncorporate.rename(
        columns={
            "subgroup": "subgroup_noncorporate",
            "source": "source_noncorporate",
            "assigned_label": "assigned_label_noncorporate",
            "micro_topic_id": "micro_topic_id_noncorporate",
            "topic_name_original": "topic_name_original_noncorporate_raw",
            "final_merge_group_id": "final_merge_group_id_noncorporate",
            "topic_size": "topic_size_noncorporate_raw",
            "group_size": "group_size_noncorporate_raw",
        }
    ).copy()

    master = master.merge(
        noncorp_lookup,
        on=[
            "subgroup_noncorporate",
            "source_noncorporate",
            "assigned_label_noncorporate",
            "micro_topic_id_noncorporate",
            "final_merge_group_id_noncorporate",
        ],
        how="left",
        validate="one_to_one",
    )
    master = master.merge(
        corporate_lookup,
        left_on="best_matched_corporate_group_id",
        right_on="final_merge_group_id_corporate",
        how="left",
        validate="many_to_one",
    )
    master = master.merge(
        precedence,
        left_on="best_pair_id",
        right_on="pair_id",
        how="left",
        suffixes=("", "_pair"),
    )

    master["macro_topic"] = master["assigned_label_noncorporate"]
    master["macro_topic_name"] = master["macro_topic_name_noncorporate"]
    master["dyad"] = master["best_dyad"].fillna("").replace("", "manual_override_no_direct_dyad")
    master["narrative_alignment_note"] = master.apply(build_alignment_note, axis=1)
    if "global_narrative_alignment_note" in master.columns:
        global_note = master["global_narrative_alignment_note"].fillna("").astype(str).str.strip()
        master.loc[global_note != "", "narrative_alignment_note"] = global_note[global_note != ""]
    master["review_decision"] = ""
    master["review_notes"] = ""

    desired_columns = [
        "macro_topic",
        "macro_topic_name",
        "dyad",
        "inclusion_reason",
        "subgroup_noncorporate",
        "source_noncorporate",
        "final_merge_group_id_noncorporate",
        "micro_topic_id_noncorporate",
        "topic_name_original_noncorporate",
        "topic_label_refined_noncorporate",
        "overall_summary_noncorporate",
        *[f"phase_{idx}_years_noncorporate" for idx in PHASE_NUMBERS],
        "subgroup_corporate",
        "source_corporate",
        "final_merge_group_id_corporate",
        "micro_topic_id_corporate",
        "topic_name_original_corporate",
        "topic_label_refined_corporate",
        "overall_summary_corporate",
        *[f"phase_{idx}_years_corporate" for idx in PHASE_NUMBERS],
        "best_cosine_similarity",
        "precedence_class",
        "precedence_type_label",
        "narrative_alignment_note",
        "direct_pair_count",
        "best_pair_id",
        "manual_review_notes",
        "review_decision",
        "review_notes",
    ]
    master = master[desired_columns].copy()
    master["macro_topic"] = pd.Categorical(master["macro_topic"], categories=MACRO_TOPIC_ORDER, ordered=True)
    master["source_noncorporate"] = pd.Categorical(
        master["source_noncorporate"], categories=SOURCE_ORDER, ordered=True
    )
    master = master.sort_values(
        ["macro_topic", "source_noncorporate", "best_cosine_similarity", "subgroup_noncorporate", "micro_topic_id_noncorporate"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    return master


def build_backlog_noncorporate(
    merged_micro_root: Path,
    pair_root: Path,
    included_noncorporate: pd.DataFrame,
) -> pd.DataFrame:
    catalog = load_group_catalog(merged_micro_root)
    direct_pairs, _ = build_pair_evidence(pair_root=pair_root, catalog=catalog, threshold=0.0)
    included_keys = included_noncorporate[["subgroup", "micro_topic_id"]].assign(_included=True)
    backlog = catalog.loc[catalog["source"].isin(["academic", "media"])].copy()
    backlog = backlog.merge(included_keys, on=["subgroup", "micro_topic_id"], how="left")
    backlog = backlog.loc[backlog["_included"] != True].drop(columns=["_included"]).copy()  # noqa: E712

    if not direct_pairs.empty:
        best = (
            direct_pairs.sort_values(
                ["cosine_similarity", "matched_corporate_group_id"],
                ascending=[False, True],
            )
            .groupby(["noncorporate_subgroup", "noncorporate_micro_topic_id"], dropna=False)
            .head(1)
            .rename(
                columns={
                    "noncorporate_subgroup": "subgroup",
                    "noncorporate_micro_topic_id": "micro_topic_id",
                    "matched_corporate_subgroup": "best_matched_corporate_subgroup",
                    "matched_corporate_micro_topic_id": "best_matched_corporate_micro_topic_id",
                    "matched_corporate_topic_name_original": "best_matched_corporate_topic_name_original",
                    "matched_corporate_group_id": "best_matched_corporate_group_id",
                }
            )[
                [
                    "subgroup",
                    "micro_topic_id",
                    "pair_id",
                    "dyad",
                    "cosine_similarity",
                    "best_matched_corporate_subgroup",
                    "best_matched_corporate_micro_topic_id",
                    "best_matched_corporate_topic_name_original",
                    "best_matched_corporate_group_id",
                ]
            ]
            .rename(
                columns={
                    "pair_id": "best_pair_id",
                    "dyad": "best_dyad",
                    "cosine_similarity": "best_corporate_similarity",
                }
            )
        )
        backlog = backlog.merge(best, on=["subgroup", "micro_topic_id"], how="left")
    else:
        backlog["best_pair_id"] = ""
        backlog["best_dyad"] = ""
        backlog["best_corporate_similarity"] = pd.NA
        backlog["best_matched_corporate_subgroup"] = ""
        backlog["best_matched_corporate_micro_topic_id"] = pd.NA
        backlog["best_matched_corporate_topic_name_original"] = ""
        backlog["best_matched_corporate_group_id"] = ""

    backlog["backlog_reason"] = backlog["best_pair_id"].apply(
        lambda value: "no_direct_pair_to_corporate" if not str(value).strip() else "not_selected_after_override_review"
    )
    backlog["review_decision"] = ""
    backlog["review_notes"] = ""
    backlog["source"] = pd.Categorical(backlog["source"], categories=SOURCE_ORDER, ordered=True)
    backlog["assigned_label"] = pd.Categorical(backlog["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
    backlog = backlog.sort_values(
        ["assigned_label", "source", "best_corporate_similarity", "subgroup", "micro_topic_id"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    return backlog[
        [
            "assigned_label",
            "macro_topic_name",
            "source",
            "subgroup",
            "final_merge_group_id",
            "micro_topic_id",
            "topic_name_original",
            "topic_size",
            "group_size",
            "best_dyad",
            "best_corporate_similarity",
            "best_pair_id",
            "best_matched_corporate_group_id",
            "best_matched_corporate_subgroup",
            "best_matched_corporate_micro_topic_id",
            "best_matched_corporate_topic_name_original",
            "backlog_reason",
            "review_decision",
            "review_notes",
        ]
    ].rename(columns={"assigned_label": "macro_topic"})


def build_all_groups_optional(
    selected_with_narratives: pd.DataFrame,
    included_noncorporate: pd.DataFrame,
) -> pd.DataFrame:
    all_groups = selected_with_narratives.copy()
    included_map = included_noncorporate[
        ["subgroup", "micro_topic_id", "inclusion_reason", "best_matched_corporate_group_id", "best_dyad"]
    ].rename(
        columns={
            "inclusion_reason": "noncorporate_inclusion_reason",
            "best_matched_corporate_group_id": "best_matched_corporate_group_id",
            "best_dyad": "best_dyad",
        }
    )
    all_groups = all_groups.merge(included_map, on=["subgroup", "micro_topic_id"], how="left")
    all_groups["selection_bucket"] = all_groups["source"].map(
        lambda source: "included_corporate" if source == "corporate" else "included_noncorporate"
    )
    all_groups.loc[all_groups["source"] == "corporate", "noncorporate_inclusion_reason"] = "all_corporate_groups_included"
    ordered = [
        "assigned_label",
        "macro_topic_name",
        "source",
        "subgroup",
        "final_merge_group_id",
        "micro_topic_id",
        "topic_name_original",
        "topic_label_refined",
        "overall_summary",
        *PHASE_YEAR_FIELDS,
        *PHASE_SUMMARY_FIELDS,
        "evolution_pattern",
        "evidence_note",
        "selection_bucket",
        "noncorporate_inclusion_reason",
        "best_matched_corporate_group_id",
        "best_dyad",
        "topic_size",
        "group_size",
        "active_year_count",
        "active_year_min",
        "active_year_max",
        "share_of_non_outlier",
        "cumulative_share",
        "overall_keywords",
    ]
    all_groups["source"] = pd.Categorical(all_groups["source"], categories=SOURCE_ORDER, ordered=True)
    all_groups["assigned_label"] = pd.Categorical(all_groups["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True)
    all_groups = all_groups.sort_values(["assigned_label", "source", "subgroup", "micro_topic_id"]).reset_index(drop=True)
    return all_groups[ordered].rename(columns={"assigned_label": "macro_topic"})


def build_summary_sheet(
    selected_with_narratives: pd.DataFrame,
    included_corporate: pd.DataFrame,
    included_noncorporate: pd.DataFrame,
    linked: pd.DataFrame,
    unlinked: pd.DataFrame,
    backlog: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"section": "overall", "macro_topic": "ALL", "metric": "selected_groups_total", "value": int(len(selected_with_narratives))},
        {"section": "overall", "macro_topic": "ALL", "metric": "included_corporate_groups", "value": int(len(included_corporate))},
        {"section": "overall", "macro_topic": "ALL", "metric": "included_noncorporate_groups", "value": int(len(included_noncorporate))},
        {"section": "overall", "macro_topic": "ALL", "metric": "included_linked_rows", "value": int(len(linked))},
        {"section": "overall", "macro_topic": "ALL", "metric": "included_unlinked_rows", "value": int(len(unlinked))},
        {"section": "overall", "macro_topic": "ALL", "metric": "backlog_noncorporate_rows", "value": int(len(backlog))},
    ]
    for macro_topic in MACRO_TOPIC_ORDER:
        rows.extend(
            [
                {
                    "section": "macro_topic",
                    "macro_topic": macro_topic,
                    "metric": "master_review_rows",
                    "value": int((included_noncorporate["assigned_label"] == macro_topic).sum()),
                },
                {
                    "section": "macro_topic",
                    "macro_topic": macro_topic,
                    "metric": "linked_rows",
                    "value": int((linked["macro_topic"] == macro_topic).sum()),
                },
                {
                    "section": "macro_topic",
                    "macro_topic": macro_topic,
                    "metric": "unlinked_rows",
                    "value": int((unlinked["macro_topic"] == macro_topic).sum()),
                },
                {
                    "section": "macro_topic",
                    "macro_topic": macro_topic,
                    "metric": "backlog_rows",
                    "value": int((backlog["macro_topic"] == macro_topic).sum()),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_readme_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": "Purpose",
                "detail": "Workbook híbrido para revisão humana de relações corporate–academic/media e dos incluídos posteriores.",
            },
            {
                "section": "master_review",
                "detail": "Aba principal: uma linha por grupo não-corporate incluído, comparado ao seu melhor corporate match.",
            },
            {
                "section": "included_linked",
                "detail": "Subconjunto com ligação direta explícita a corporate (best_dyad presente / par direto).",
            },
            {
                "section": "included_unlinked",
                "detail": "Incluídos posteriores por override/manual review sem dyad direto acima do limiar.",
            },
            {
                "section": "backlog_noncorporate",
                "detail": "Grupos academic/media fora do subset corporate-focus atual, com melhor match corporate quando existir.",
            },
            {
                "section": "T1_review-T6_review",
                "detail": "Recortes da master_review por macrotema para leitura mais rápida.",
            },
            {
                "section": "all_groups_optional",
                "detail": "Visão enciclopédica dos 241 grupos selecionados com narrativas Stage 2 completas.",
            },
            {
                "section": "Manual columns",
                "detail": "Use review_decision e review_notes para registrar julgamento qualitativo sem alterar as colunas derivadas.",
            },
        ]
    )


def write_outputs(
    output_root: Path,
    readme: pd.DataFrame,
    summary: pd.DataFrame,
    master_review: pd.DataFrame,
    included_linked: pd.DataFrame,
    included_unlinked: pd.DataFrame,
    backlog_noncorporate: pd.DataFrame,
    all_groups_optional: pd.DataFrame,
) -> dict[str, Path]:
    ensure_directory(output_root)
    paths = {
        "workbook": output_root / "corporate_focus_hybrid_review.xlsx",
        "master_review_csv": output_root / "corporate_focus_master_review.csv",
        "included_linked_csv": output_root / "corporate_focus_included_linked.csv",
        "included_unlinked_csv": output_root / "corporate_focus_included_unlinked.csv",
        "backlog_noncorporate_csv": output_root / "corporate_focus_backlog_noncorporate.csv",
        "all_groups_optional_csv": output_root / "corporate_focus_all_groups_optional.csv",
        "manifest": output_root / "corporate_focus_hybrid_review_manifest.json",
    }

    master_review.to_csv(paths["master_review_csv"], index=False)
    included_linked.to_csv(paths["included_linked_csv"], index=False)
    included_unlinked.to_csv(paths["included_unlinked_csv"], index=False)
    backlog_noncorporate.to_csv(paths["backlog_noncorporate_csv"], index=False)
    all_groups_optional.to_csv(paths["all_groups_optional_csv"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        sheets: list[tuple[str, pd.DataFrame]] = [
            ("README", readme),
            ("summary", summary),
            ("master_review", master_review),
            ("included_linked", included_linked),
            ("included_unlinked", included_unlinked),
            ("backlog_noncorporate", backlog_noncorporate),
        ]
        for macro_topic in MACRO_TOPIC_ORDER:
            sheets.append((f"{macro_topic}_review", master_review.loc[master_review["macro_topic"] == macro_topic].copy()))
        sheets.append(("all_groups_optional", all_groups_optional))

        for sheet_name, frame in sheets:
            cleaned = sanitize_frame_for_excel(frame)
            cleaned.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook_autofit(writer, sheet_name, cleaned)
    return paths


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    selected_with_narratives = load_selected_with_narratives(args.bundle_root)
    included_corporate, included_noncorporate, manual_override = load_review_inputs(
        review_root=args.review_root,
        stage12_input_root=args.stage12_input_root,
    )
    precedence, _ = build_pair_metadata(
        pair_root=args.pair_root,
        global_llm_pair_root=args.global_llm_pair_root,
    )
    _, corporate_lookup = prepare_selected_lookup(selected_with_narratives)

    master_review = build_master_review(
        included_noncorporate=included_noncorporate,
        selected_with_narratives=selected_with_narratives,
        corporate_lookup=corporate_lookup,
        precedence=precedence,
    )
    included_linked = master_review.loc[master_review["dyad"] != "manual_override_no_direct_dyad"].copy()
    included_unlinked = master_review.loc[master_review["dyad"] == "manual_override_no_direct_dyad"].copy()
    backlog_noncorporate = build_backlog_noncorporate(
        merged_micro_root=args.merged_micro_root,
        pair_root=args.pair_root,
        included_noncorporate=included_noncorporate,
    )
    all_groups_optional = build_all_groups_optional(
        selected_with_narratives=selected_with_narratives,
        included_noncorporate=included_noncorporate,
    )
    readme = build_readme_sheet()
    summary = build_summary_sheet(
        selected_with_narratives=selected_with_narratives,
        included_corporate=included_corporate,
        included_noncorporate=included_noncorporate,
        linked=included_linked,
        unlinked=included_unlinked,
        backlog=backlog_noncorporate,
    )

    output_paths = write_outputs(
        output_root=args.review_root,
        readme=readme,
        summary=summary,
        master_review=master_review,
        included_linked=included_linked,
        included_unlinked=included_unlinked,
        backlog_noncorporate=backlog_noncorporate,
        all_groups_optional=all_groups_optional,
    )

    write_json(
        output_paths["manifest"],
        {
            "bundle_root": str(args.bundle_root),
            "review_root": str(args.review_root),
            "stage12_input_root": str(args.stage12_input_root),
            "pair_root": str(args.pair_root),
            "row_counts": {
                "selected_with_narratives": int(len(selected_with_narratives)),
                "included_corporate": int(len(included_corporate)),
                "included_noncorporate": int(len(included_noncorporate)),
                "manual_override_noncorporate": int(len(manual_override)),
                "master_review": int(len(master_review)),
                "included_linked": int(len(included_linked)),
                "included_unlinked": int(len(included_unlinked)),
                "backlog_noncorporate": int(len(backlog_noncorporate)),
                "all_groups_optional": int(len(all_groups_optional)),
            },
            "outputs": {name: str(path) for name, path in output_paths.items()},
        },
    )

    print(args.review_root)


if __name__ == "__main__":
    main()
