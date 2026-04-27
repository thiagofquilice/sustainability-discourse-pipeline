#!/usr/bin/env python3
"""Create a source-specific corpus subset and aligned embedding subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "full_corpus_chunked.parquet"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "results" / "bge_large_en_v1_5_embeddings.f32"
DEFAULT_EMBEDDINGS_META = PROJECT_ROOT / "data" / "results" / "bge_large_en_v1_5_embeddings.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDINGS_META)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output-corpus", type=Path, required=True)
    parser.add_argument("--output-embeddings", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def load_embedding_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_embedding_meta(path: Path, rows: int, dims: int, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows,
        "dims": dims,
        "model_name": model_name,
        "dtype": "float32",
        "created_from_subset": True,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_corpus.parent.mkdir(parents=True, exist_ok=True)
    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    source_series = pq.read_table(args.input, columns=["source"]).to_pandas()["source"].astype(str)
    source_mask = source_series.eq(args.source).to_numpy()
    selected_rows = int(source_mask.sum())
    total_rows = int(source_mask.shape[0])
    if selected_rows == 0:
        raise SystemExit(f"No rows found for source={args.source!r}")

    subset_table = pq.read_table(args.input, filters=[("source", "==", args.source)])
    subset_df = subset_table.to_pandas(types_mapper=pd.ArrowDtype).convert_dtypes(dtype_backend="numpy_nullable")
    subset_df.to_parquet(args.output_corpus, index=False)

    meta = load_embedding_meta(args.embedding_meta)
    dims = int(meta["dims"])
    rows = int(meta["rows"])
    if rows != total_rows:
        raise SystemExit(
            f"Embedding row mismatch: embeddings have {rows} rows but corpus has {total_rows}"
        )

    full_embeddings = np.memmap(args.embedding_file, dtype="float32", mode="r", shape=(rows, dims))
    if args.output_embeddings.exists():
        args.output_embeddings.unlink()
    subset_embeddings = np.memmap(
        args.output_embeddings,
        dtype="float32",
        mode="w+",
        shape=(selected_rows, dims),
    )
    subset_embeddings[:] = full_embeddings[source_mask]
    subset_embeddings.flush()
    write_embedding_meta(args.output_meta, selected_rows, dims, str(meta.get("model_name", "")))

    summary = {
        "source": args.source,
        "total_rows_in_full_corpus": total_rows,
        "subset_rows": selected_rows,
        "subset_original_docs": int(subset_df["source_doc_id"].nunique()),
        "subset_was_chunked_original_docs": int(
            subset_df.groupby("source_doc_id", dropna=False)["was_chunked"].max().sum()
        ),
        "mean_chunks_per_original_doc": round(
            float(subset_df.groupby("source_doc_id", dropna=False)["chunk_id"].nunique().mean()),
            6,
        ),
        "median_chunks_per_original_doc": float(
            subset_df.groupby("source_doc_id", dropna=False)["chunk_id"].nunique().median()
        ),
        "max_chunks_per_original_doc": int(
            subset_df.groupby("source_doc_id", dropna=False)["chunk_id"].nunique().max()
        ),
        "year_min": int(subset_df["year"].min()),
        "year_max": int(subset_df["year"].max()),
    }
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote subset corpus to {args.output_corpus}")
    print(f"Wrote subset embeddings to {args.output_embeddings}")


if __name__ == "__main__":
    main()
