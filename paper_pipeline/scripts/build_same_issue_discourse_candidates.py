#!/usr/bin/env python3
"""Build top-k same-issue discourse candidates from cross-source semantic similarities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import (
    DISCOURSE_DYADS,
    DISCOURSE_DYAD_LABELS,
    DYADS,
    DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT,
    DEFAULT_THRESHOLD_065_OUTPUT_ROOT,
    build_pair_id,
    configure_logging,
    dyad_code,
    load_stage2_narratives,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_THRESHOLD_065_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DISCOURSE_FUNCTION_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--dyads", type=str, nargs="*", default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--embedding-npy", type=Path, default=None)
    parser.add_argument("--narratives-csv", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_requested_dyads(raw_dyads: list[str] | None) -> list[str]:
    if not raw_dyads:
        return [dyad_code(source_a, source_b) for source_a, source_b in DYADS]
    requested: list[str] = []
    for raw_value in raw_dyads:
        text = str(raw_value).strip()
        if not text:
            continue
        if "__" in text:
            source_a, source_b = text.split("__", 1)
        elif ":" in text:
            source_a, source_b = text.split(":", 1)
        elif "-" in text:
            source_a, source_b = text.split("-", 1)
        else:
            raise ValueError(f"Unsupported dyad format: {raw_value!r}")
        requested.append(dyad_code(source_a.strip(), source_b.strip()))
    return sorted(dict.fromkeys(requested))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def compute_missing_dyad_pairwise(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    requested_dyads: list[str],
) -> pd.DataFrame:
    if metadata.empty or embeddings.size == 0:
        return pd.DataFrame()

    normalized_embeddings = normalize_embeddings(embeddings)
    records: list[dict[str, object]] = []

    for dyad in requested_dyads:
        source_a, source_b = dyad.split("__", 1)
        dyad_label = DISCOURSE_DYAD_LABELS.get((source_a, source_b), f"{source_a} ↔ {source_b}")
        for macro_topic, macro_frame in metadata.groupby("assigned_label", sort=False):
            left = macro_frame.loc[macro_frame["source"] == source_a].reset_index(drop=True)
            right = macro_frame.loc[macro_frame["source"] == source_b].reset_index(drop=True)
            if left.empty or right.empty:
                continue

            left_indices = left["embedding_row_index"].astype(int).to_numpy()
            right_indices = right["embedding_row_index"].astype(int).to_numpy()
            similarities = normalized_embeddings[left_indices] @ normalized_embeddings[right_indices].T

            rank_from_a = np.empty_like(similarities, dtype=int)
            for left_idx in range(similarities.shape[0]):
                order = np.argsort(-similarities[left_idx], kind="mergesort")
                rank_from_a[left_idx, order] = np.arange(1, similarities.shape[1] + 1)

            rank_from_b = np.empty_like(similarities, dtype=int)
            for right_idx in range(similarities.shape[1]):
                order = np.argsort(-similarities[:, right_idx], kind="mergesort")
                rank_from_b[order, right_idx] = np.arange(1, similarities.shape[0] + 1)

            macro_topic_name = ""
            if "macro_topic_name" in macro_frame.columns:
                non_empty = macro_frame["macro_topic_name"].dropna().astype(str)
                if not non_empty.empty:
                    macro_topic_name = non_empty.iloc[0]

            left_rows = list(left.itertuples(index=False))
            right_rows = list(right.itertuples(index=False))
            for left_pos, left_row in enumerate(left_rows):
                for right_pos, right_row in enumerate(right_rows):
                    records.append(
                        {
                            "macro_topic": str(macro_topic),
                            "macro_topic_name": macro_topic_name,
                            "dyad": dyad,
                            "dyad_label": dyad_label,
                            "source_a": source_a,
                            "subgroup_a": str(left_row.subgroup),
                            "microtopic_id_a": int(left_row.microtopic_id),
                            "microtopic_label_a": str(left_row.microtopic_label),
                            "document_count_a": int(left_row.document_count),
                            "chunk_count_a": int(left_row.chunk_count),
                            "source_b": source_b,
                            "subgroup_b": str(right_row.subgroup),
                            "microtopic_id_b": int(right_row.microtopic_id),
                            "microtopic_label_b": str(right_row.microtopic_label),
                            "document_count_b": int(right_row.document_count),
                            "chunk_count_b": int(right_row.chunk_count),
                            "cosine_similarity": float(similarities[left_pos, right_pos]),
                            "rank_from_a": int(rank_from_a[left_pos, right_pos]),
                            "rank_from_b": int(rank_from_b[left_pos, right_pos]),
                            "is_mutual_nearest_neighbor": bool(
                                rank_from_a[left_pos, right_pos] == 1 and rank_from_b[left_pos, right_pos] == 1
                            ),
                        }
                    )

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_pairwise_csv = args.pair_root / "all_pairwise_similarities.csv"
    join_ready_csv = args.pair_root / "pair_join_ready_table.csv"
    metadata_csv = args.metadata_csv or (args.pair_root / "microtopic_profile_embedding_metadata.csv")
    embedding_npy = args.embedding_npy or (args.pair_root / "microtopic_profile_embeddings.npy")
    requested_dyads = parse_requested_dyads(args.dyads)
    narratives = load_stage2_narratives(args.narratives_csv) if args.narratives_csv else load_stage2_narratives()
    all_pairwise = safe_read_csv(all_pairwise_csv)
    join_ready = safe_read_csv(join_ready_csv)

    candidate_csv = args.output_root / "candidate_pairs_topk.csv"
    summary_csv = args.output_root / "candidate_summary.csv"
    derived_dyads: list[str] = []
    if not narratives.empty:
        existing_dyads = set()
        if not all_pairwise.empty and "dyad" in all_pairwise.columns:
            existing_dyads = {str(value) for value in all_pairwise["dyad"].dropna().astype(str).unique()}
            all_pairwise = all_pairwise.loc[all_pairwise["dyad"].isin(requested_dyads)].copy()
        missing_dyads = [dyad for dyad in requested_dyads if dyad not in existing_dyads]
        if missing_dyads:
            metadata = safe_read_csv(metadata_csv)
            if metadata.empty or not embedding_npy.exists():
                raise FileNotFoundError(
                    "Cannot derive missing dyads without microtopic embedding metadata and embedding matrix."
                )
            embeddings = np.load(embedding_npy)
            derived_pairwise = compute_missing_dyad_pairwise(metadata=metadata, embeddings=embeddings, requested_dyads=missing_dyads)
            if not derived_pairwise.empty:
                all_pairwise = pd.concat([all_pairwise, derived_pairwise], ignore_index=True, sort=False)
                derived_dyads = sorted({str(value) for value in derived_pairwise["dyad"].dropna().astype(str).unique()})

    if all_pairwise.empty or narratives.empty:
        pd.DataFrame().to_csv(candidate_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        write_json(
            args.output_root / "candidate_manifest.json",
            {
                "pair_root": str(args.pair_root),
                "requested_dyads": requested_dyads,
                "derived_dyads": derived_dyads,
                "top_k": args.top_k,
                "min_similarity": args.min_similarity,
                "candidate_count": 0,
                "outputs": {
                    "candidate_pairs_topk": str(candidate_csv),
                    "candidate_summary": str(summary_csv),
                },
            },
        )
        print(args.output_root)
        return

    ready_narratives = narratives.loc[narratives["is_narrative_ready"]].copy()
    ready_keys = {(str(row.subgroup), int(row.micro_topic_id)) for row in ready_narratives.itertuples(index=False)}

    filtered = all_pairwise.loc[all_pairwise["cosine_similarity"] >= args.min_similarity].copy()
    filtered = filtered.loc[
        filtered.apply(
            lambda row: (
                (str(row["subgroup_a"]), int(row["microtopic_id_a"])) in ready_keys
                and (str(row["subgroup_b"]), int(row["microtopic_id_b"])) in ready_keys
            ),
            axis=1,
        )
    ].copy()
    if filtered.empty:
        pd.DataFrame().to_csv(candidate_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        write_json(
            args.output_root / "candidate_manifest.json",
            {
                "pair_root": str(args.pair_root),
                "requested_dyads": requested_dyads,
                "derived_dyads": derived_dyads,
                "top_k": args.top_k,
                "min_similarity": args.min_similarity,
                "candidate_count": 0,
                "ready_narrative_microtopics": int(ready_narratives.shape[0]),
                "outputs": {
                    "candidate_pairs_topk": str(candidate_csv),
                    "candidate_summary": str(summary_csv),
                },
            },
        )
        print(args.output_root)
        return

    all_pairwise = all_pairwise.loc[all_pairwise["dyad"].isin(requested_dyads)].copy()
    filtered["pair_id"] = filtered.apply(
        lambda row: build_pair_id(
            dyad=str(row["dyad"]),
            macro_topic=str(row["macro_topic"]),
            source_a=str(row["source_a"]),
            microtopic_id_a=int(row["microtopic_id_a"]),
            source_b=str(row["source_b"]),
            microtopic_id_b=int(row["microtopic_id_b"]),
        ),
        axis=1,
    )
    filtered = filtered.sort_values(
        ["dyad", "macro_topic", "source_a", "microtopic_id_a", "cosine_similarity"],
        ascending=[True, True, True, True, False],
    )
    candidates = (
        filtered.groupby(["dyad", "macro_topic", "source_a", "microtopic_id_a"], group_keys=False)
        .head(args.top_k)
        .copy()
    )
    candidates["candidate_rank_from_source_a"] = (
        candidates.groupby(["dyad", "macro_topic", "source_a", "microtopic_id_a"]).cumcount() + 1
    )
    candidates = candidates.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)

    if not join_ready.empty:
        temporal_cols = [
            "pair_id",
            "precedence_type_label",
            "best_lag_docs",
            "best_correlation_docs",
            "balanced_is_viable",
            "both_have_narratives",
        ]
        temporal_cols = [col for col in temporal_cols if col in join_ready.columns]
        if temporal_cols:
            candidates = candidates.merge(join_ready[temporal_cols], on="pair_id", how="left")
            candidates["exists_in_strict_pair_branch"] = candidates["precedence_type_label"].notna()
        else:
            candidates["exists_in_strict_pair_branch"] = False
    else:
        candidates["exists_in_strict_pair_branch"] = False

    summary = (
        candidates.groupby(["dyad", "macro_topic", "macro_topic_name"], dropna=False)
        .agg(
            candidate_pair_count=("pair_id", "nunique"),
            mean_similarity=("cosine_similarity", "mean"),
            max_similarity=("cosine_similarity", "max"),
            pairs_also_in_strict_branch=("exists_in_strict_pair_branch", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
        .sort_values(["candidate_pair_count", "mean_similarity"], ascending=[False, False])
    )

    candidates.to_csv(candidate_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_json(
        args.output_root / "candidate_manifest.json",
        {
            "pair_root": str(args.pair_root),
            "all_pairwise_csv": str(all_pairwise_csv),
            "join_ready_csv": str(join_ready_csv),
            "metadata_csv": str(metadata_csv),
            "embedding_npy": str(embedding_npy),
            "requested_dyads": requested_dyads,
            "derived_dyads": derived_dyads,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "ready_narrative_microtopics": int(ready_narratives.shape[0]),
            "candidate_count": int(candidates.shape[0]),
            "outputs": {
                "candidate_pairs_topk": str(candidate_csv),
                "candidate_summary": str(summary_csv),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
