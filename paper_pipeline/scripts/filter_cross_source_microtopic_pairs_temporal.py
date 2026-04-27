#!/usr/bin/env python3
"""Build temporal diagnostics and viability views for matched microtopic pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cross_source_microtopic_common import (
    OUTPUT_ROOT,
    TEMPORAL_RULES,
    configure_logging,
    threshold_suffix,
    variance_flag,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--semantic-pairs-csv", type=Path, default=None)
    parser.add_argument("--profiles-csv", type=Path, default=None)
    parser.add_argument("--main-threshold", type=float, default=0.75)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def parse_years(raw: str) -> list[int]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [int(x) for x in json.loads(raw)]


def parse_count_map(raw: str) -> dict[int, int]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    payload = json.loads(raw)
    return {int(k): int(v) for k, v in payload.items()}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    semantic_pairs_csv = args.semantic_pairs_csv or (
        args.output_root / f"retained_pairs_threshold_{threshold_suffix(args.main_threshold)}.csv"
    )
    profiles_csv = args.profiles_csv or (args.output_root / "microtopic_profiles.csv")

    semantic_pairs = pd.read_csv(semantic_pairs_csv)
    profiles = pd.read_csv(profiles_csv)
    if semantic_pairs.empty:
        diagnostics_path = args.output_root / "semantic_pairs_with_temporal_diagnostics.csv"
        excluded_path = args.output_root / "temporally_excluded_pairs.csv"
        pd.DataFrame().to_csv(diagnostics_path, index=False)
        pd.DataFrame().to_csv(excluded_path, index=False)
        output_paths: dict[str, str] = {
            "semantic_pairs_with_temporal_diagnostics": str(diagnostics_path),
            "temporally_excluded_pairs": str(excluded_path),
        }
        for rule_name in TEMPORAL_RULES:
            path = args.output_root / f"temporally_viable_pairs_{rule_name}.csv"
            pd.DataFrame().to_csv(path, index=False)
            output_paths[f"temporally_viable_pairs_{rule_name}"] = str(path)
        write_json(
            args.output_root / "temporal_filter_manifest.json",
            {
                "semantic_pairs_csv": str(semantic_pairs_csv),
                "main_threshold": args.main_threshold,
                "profiles_csv": str(profiles_csv),
                "rule_definitions": TEMPORAL_RULES,
                "diagnostic_row_count": 0,
                "excluded_row_count": 0,
                "outputs": output_paths,
            },
        )
        print(args.output_root)
        return

    profiles["microtopic_id"] = pd.to_numeric(profiles["microtopic_id"], errors="coerce").astype(int)
    profile_map = {
        (str(row.subgroup), int(row.microtopic_id)): row._asdict()
        for row in profiles.itertuples(index=False)
    }

    diagnostic_rows: list[dict] = []
    excluded_rows: list[dict] = []
    viable_tables: dict[str, list[dict]] = {rule: [] for rule in TEMPORAL_RULES}

    for row in semantic_pairs.itertuples(index=False):
        key_a = (str(row.subgroup_a), int(row.microtopic_id_a))
        key_b = (str(row.subgroup_b), int(row.microtopic_id_b))
        profile_a = profile_map[key_a]
        profile_b = profile_map[key_b]

        years_a = parse_years(profile_a["years_present"])
        years_b = parse_years(profile_b["years_present"])
        overlap_years = sorted(set(years_a).intersection(years_b))
        doc_counts_a = parse_count_map(profile_a["year_doc_counts"])
        doc_counts_b = parse_count_map(profile_b["year_doc_counts"])
        chunk_counts_a = parse_count_map(profile_a["year_chunk_counts"])
        chunk_counts_b = parse_count_map(profile_b["year_chunk_counts"])

        overlap_doc_counts_a = [doc_counts_a.get(year, 0) for year in overlap_years]
        overlap_doc_counts_b = [doc_counts_b.get(year, 0) for year in overlap_years]
        overlap_chunk_counts_a = [chunk_counts_a.get(year, 0) for year in overlap_years]
        overlap_chunk_counts_b = [chunk_counts_b.get(year, 0) for year in overlap_years]

        low_variance_flag = (
            variance_flag(overlap_doc_counts_a)
            or variance_flag(overlap_doc_counts_b)
            or variance_flag(overlap_chunk_counts_a)
            or variance_flag(overlap_chunk_counts_b)
        )

        base_record = {
            **row._asdict(),
            "source_a_active_year_count": int(profile_a["active_year_count"]),
            "source_b_active_year_count": int(profile_b["active_year_count"]),
            "source_a_first_year": int(profile_a["first_year"]),
            "source_b_first_year": int(profile_b["first_year"]),
            "source_a_last_year": int(profile_a["last_year"]),
            "source_b_last_year": int(profile_b["last_year"]),
            "source_a_document_count": int(profile_a["document_count"]),
            "source_b_document_count": int(profile_b["document_count"]),
            "source_a_chunk_count": int(profile_a["chunk_count"]),
            "source_b_chunk_count": int(profile_b["chunk_count"]),
            "overlap_year_count": int(len(overlap_years)),
            "overlap_years": json.dumps(overlap_years),
            "low_variance_flag": bool(low_variance_flag),
            "too_sparse_flag": bool(
                int(profile_a["document_count"]) < TEMPORAL_RULES["lenient"]["min_docs_per_side"]
                or int(profile_b["document_count"]) < TEMPORAL_RULES["lenient"]["min_docs_per_side"]
                or int(profile_a["chunk_count"]) < TEMPORAL_RULES["lenient"]["min_chunks_per_side"]
                or int(profile_b["chunk_count"]) < TEMPORAL_RULES["lenient"]["min_chunks_per_side"]
            ),
            "temporal_notes": "low_variance_or_flat_overlap" if low_variance_flag else "",
        }

        for rule_name, rule in TEMPORAL_RULES.items():
            reasons: list[str] = []
            if base_record["source_a_active_year_count"] < rule["min_active_years_per_side"]:
                reasons.append("source_a_active_years_below_threshold")
            if base_record["source_b_active_year_count"] < rule["min_active_years_per_side"]:
                reasons.append("source_b_active_years_below_threshold")
            if base_record["overlap_year_count"] < rule["min_overlap_years"]:
                reasons.append("overlap_years_below_threshold")
            if base_record["source_a_document_count"] < rule["min_docs_per_side"]:
                reasons.append("source_a_docs_below_threshold")
            if base_record["source_b_document_count"] < rule["min_docs_per_side"]:
                reasons.append("source_b_docs_below_threshold")
            if base_record["source_a_chunk_count"] < rule["min_chunks_per_side"]:
                reasons.append("source_a_chunks_below_threshold")
            if base_record["source_b_chunk_count"] < rule["min_chunks_per_side"]:
                reasons.append("source_b_chunks_below_threshold")

            base_record[f"{rule_name}_is_viable"] = len(reasons) == 0
            base_record[f"{rule_name}_exclusion_reasons"] = ";".join(reasons)
            if reasons:
                excluded_rows.append(
                    {
                        "pair_id": base_record["pair_id"],
                        "macro_topic": base_record["macro_topic"],
                        "dyad": base_record["dyad"],
                        "rule_name": rule_name,
                        "exclusion_reasons": ";".join(reasons),
                    }
                )
            else:
                viable_tables[rule_name].append(base_record.copy())

        diagnostic_rows.append(base_record)

    diagnostics = pd.DataFrame(diagnostic_rows)
    excluded = pd.DataFrame(excluded_rows)

    diagnostics_path = args.output_root / "semantic_pairs_with_temporal_diagnostics.csv"
    excluded_path = args.output_root / "temporally_excluded_pairs.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    output_paths: dict[str, str] = {
        "semantic_pairs_with_temporal_diagnostics": str(diagnostics_path),
        "temporally_excluded_pairs": str(excluded_path),
    }
    for rule_name, records in viable_tables.items():
        frame = pd.DataFrame(records)
        path = args.output_root / f"temporally_viable_pairs_{rule_name}.csv"
        frame.to_csv(path, index=False)
        output_paths[f"temporally_viable_pairs_{rule_name}"] = str(path)

    write_json(
        args.output_root / "temporal_filter_manifest.json",
        {
            "semantic_pairs_csv": str(semantic_pairs_csv),
            "main_threshold": args.main_threshold,
            "profiles_csv": str(profiles_csv),
            "rule_definitions": TEMPORAL_RULES,
            "diagnostic_row_count": int(diagnostics.shape[0]),
            "excluded_row_count": int(excluded.shape[0]),
            "outputs": output_paths,
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
