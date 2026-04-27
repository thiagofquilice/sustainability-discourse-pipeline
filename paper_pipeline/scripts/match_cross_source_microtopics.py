#!/usr/bin/env python3
"""Match semantically similar microtopics across sources within macro-topic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import (
    DYAD_LABELS,
    DYADS,
    MACRO_TOPIC_ORDER,
    OUTPUT_ROOT,
    SIMILARITY_THRESHOLDS,
    build_pair_id,
    configure_logging,
    dyad_code,
    threshold_suffix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--embeddings-npy", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_pairwise_rows(
    profiles_a: pd.DataFrame,
    emb_a: np.ndarray,
    profiles_b: pd.DataFrame,
    emb_b: np.ndarray,
    macro_topic: str,
    dyad: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_a, source_b = dyad
    dyad_name = dyad_code(source_a, source_b)
    dyad_label = DYAD_LABELS[dyad]

    similarity = np.matmul(emb_a, emb_b.T)
    rank_a = np.argsort(np.argsort(-similarity, axis=1), axis=1) + 1
    rank_b = np.argsort(np.argsort(-similarity, axis=0), axis=0) + 1

    rows: list[dict] = []
    for idx_a, row_a in profiles_a.reset_index(drop=True).iterrows():
        for idx_b, row_b in profiles_b.reset_index(drop=True).iterrows():
            score = float(similarity[idx_a, idx_b])
            rows.append(
                {
                    "macro_topic": macro_topic,
                    "macro_topic_name": str(row_a["macro_topic_name"]),
                    "dyad": dyad_name,
                    "dyad_label": dyad_label,
                    "source_a": source_a,
                    "subgroup_a": str(row_a["subgroup"]),
                    "microtopic_id_a": int(row_a["microtopic_id"]),
                    "microtopic_label_a": str(row_a["microtopic_label"]),
                    "document_count_a": int(row_a["document_count"]),
                    "chunk_count_a": int(row_a["chunk_count"]),
                    "source_b": source_b,
                    "subgroup_b": str(row_b["subgroup"]),
                    "microtopic_id_b": int(row_b["microtopic_id"]),
                    "microtopic_label_b": str(row_b["microtopic_label"]),
                    "document_count_b": int(row_b["document_count"]),
                    "chunk_count_b": int(row_b["chunk_count"]),
                    "cosine_similarity": score,
                    "rank_from_a": int(rank_a[idx_a, idx_b]),
                    "rank_from_b": int(rank_b[idx_a, idx_b]),
                    "is_mutual_nearest_neighbor": bool(
                        int(rank_a[idx_a, idx_b]) == 1 and int(rank_b[idx_a, idx_b]) == 1
                    ),
                }
            )

    all_pairs = pd.DataFrame(rows)
    if all_pairs.empty:
        return all_pairs, pd.DataFrame(), pd.DataFrame()

    nearest = pd.concat(
        [
            all_pairs.loc[all_pairs["rank_from_a"] == 1]
            .assign(direction="a_to_b")
            .rename(
                columns={
                    "source_a": "query_source",
                    "subgroup_a": "query_subgroup",
                    "microtopic_id_a": "query_microtopic_id",
                    "microtopic_label_a": "query_microtopic_label",
                    "source_b": "match_source",
                    "subgroup_b": "match_subgroup",
                    "microtopic_id_b": "match_microtopic_id",
                    "microtopic_label_b": "match_microtopic_label",
                }
            ),
            all_pairs.loc[all_pairs["rank_from_b"] == 1]
            .assign(direction="b_to_a")
            .rename(
                columns={
                    "source_b": "query_source",
                    "subgroup_b": "query_subgroup",
                    "microtopic_id_b": "query_microtopic_id",
                    "microtopic_label_b": "query_microtopic_label",
                    "source_a": "match_source",
                    "subgroup_a": "match_subgroup",
                    "microtopic_id_a": "match_microtopic_id",
                    "microtopic_label_a": "match_microtopic_label",
                }
            ),
        ],
        ignore_index=True,
    )

    mnn = all_pairs.loc[all_pairs["is_mutual_nearest_neighbor"]].copy()
    if not mnn.empty:
        mnn["pair_id"] = mnn.apply(
            lambda row: build_pair_id(
                dyad=row["dyad"],
                macro_topic=row["macro_topic"],
                source_a=row["source_a"],
                microtopic_id_a=int(row["microtopic_id_a"]),
                source_b=row["source_b"],
                microtopic_id_b=int(row["microtopic_id_b"]),
            ),
            axis=1,
        )
        for threshold in SIMILARITY_THRESHOLDS:
            mnn[f"passes_threshold_{int(threshold * 100):03d}"] = mnn["cosine_similarity"] >= threshold
    return all_pairs, nearest, mnn


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    metadata_csv = args.metadata_csv or (args.output_root / "microtopic_profile_embedding_metadata.csv")
    embeddings_npy = args.embeddings_npy or (args.output_root / "microtopic_profile_embeddings.npy")

    metadata = pd.read_csv(metadata_csv)
    embeddings = np.load(embeddings_npy)
    if len(metadata) != len(embeddings):
        raise SystemExit("Embedding row count does not match metadata row count.")

    metadata["assigned_label"] = pd.Categorical(
        metadata["assigned_label"], categories=MACRO_TOPIC_ORDER, ordered=True
    )

    all_pairwise_parts: list[pd.DataFrame] = []
    nearest_parts: list[pd.DataFrame] = []
    mnn_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    threshold_columns = [f"retained_count_threshold_{threshold_suffix(threshold)}" for threshold in SIMILARITY_THRESHOLDS]

    for macro_topic in MACRO_TOPIC_ORDER:
        topic_meta = metadata.loc[metadata["assigned_label"] == macro_topic].copy()
        for dyad in DYADS:
            source_a, source_b = dyad
            dyad_name = dyad_code(source_a, source_b)
            meta_a = topic_meta.loc[topic_meta["source"] == source_a].copy()
            meta_b = topic_meta.loc[topic_meta["source"] == source_b].copy()
            emb_a = embeddings[meta_a["embedding_row_index"].astype(int).to_numpy()]
            emb_b = embeddings[meta_b["embedding_row_index"].astype(int).to_numpy()]
            if meta_a.empty or meta_b.empty:
                summary_rows.append(
                    {
                        "macro_topic": macro_topic,
                        "dyad": dyad_name,
                        "dyad_label": DYAD_LABELS[dyad],
                        "source_a_microtopic_count": int(meta_a.shape[0]),
                        "source_b_microtopic_count": int(meta_b.shape[0]),
                        "candidate_pair_count": 0,
                        "mutual_nearest_neighbor_count": 0,
                        **{column: 0 for column in threshold_columns},
                    }
                )
                continue

            all_pairs, nearest, mnn = build_pairwise_rows(
                profiles_a=meta_a,
                emb_a=emb_a,
                profiles_b=meta_b,
                emb_b=emb_b,
                macro_topic=macro_topic,
                dyad=dyad,
            )
            all_pairwise_parts.append(all_pairs)
            nearest_parts.append(nearest)
            mnn_parts.append(mnn)
            summary_row = {
                "macro_topic": macro_topic,
                "dyad": dyad_name,
                "dyad_label": DYAD_LABELS[dyad],
                "source_a_microtopic_count": int(meta_a.shape[0]),
                "source_b_microtopic_count": int(meta_b.shape[0]),
                "candidate_pair_count": int(all_pairs.shape[0]),
                "mutual_nearest_neighbor_count": int(mnn.shape[0]),
            }
            for threshold in SIMILARITY_THRESHOLDS:
                summary_row[f"retained_count_threshold_{threshold_suffix(threshold)}"] = (
                    int((mnn["cosine_similarity"] >= threshold).sum()) if not mnn.empty else 0
                )
            summary_rows.append(summary_row)

    all_pairwise = pd.concat(all_pairwise_parts, ignore_index=True) if all_pairwise_parts else pd.DataFrame()
    nearest_neighbors = pd.concat(nearest_parts, ignore_index=True) if nearest_parts else pd.DataFrame()
    mnn_pairs = pd.concat(mnn_parts, ignore_index=True) if mnn_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    all_pairwise_path = args.output_root / "all_pairwise_similarities.csv"
    nearest_path = args.output_root / "nearest_neighbors.csv"
    mnn_path = args.output_root / "mutual_nearest_neighbor_pairs.csv"
    summary_path = args.output_root / "matching_summary_by_macro_topic_dyad.csv"

    all_pairwise.to_csv(all_pairwise_path, index=False)
    nearest_neighbors.to_csv(nearest_path, index=False)
    mnn_pairs.to_csv(mnn_path, index=False)
    summary.to_csv(summary_path, index=False)

    threshold_paths: dict[str, str] = {}
    for threshold in SIMILARITY_THRESHOLDS:
        suffix = threshold_suffix(threshold)
        path = args.output_root / f"retained_pairs_threshold_{suffix}.csv"
        filtered = mnn_pairs.loc[mnn_pairs["cosine_similarity"] >= threshold].copy()
        filtered.to_csv(path, index=False)
        threshold_paths[f"retained_pairs_threshold_{suffix}"] = str(path)

    write_json(
        args.output_root / "matching_manifest.json",
        {
            "metadata_csv": str(metadata_csv),
            "embeddings_npy": str(embeddings_npy),
            "dyads": [DYAD_LABELS[dyad] for dyad in DYADS],
            "similarity_thresholds": SIMILARITY_THRESHOLDS,
            "all_pairwise_count": int(all_pairwise.shape[0]),
            "nearest_neighbor_count": int(nearest_neighbors.shape[0]),
            "mutual_nearest_neighbor_count": int(mnn_pairs.shape[0]),
            "outputs": {
                "all_pairwise_similarities": str(all_pairwise_path),
                "nearest_neighbors": str(nearest_path),
                "mutual_nearest_neighbor_pairs": str(mnn_path),
                "matching_summary_by_macro_topic_dyad": str(summary_path),
                **threshold_paths,
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
