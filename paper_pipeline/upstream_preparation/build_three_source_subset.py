#!/usr/bin/env python3
"""Build a three-source subset with full media and academic plus selected corporate industries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "full_corpus_chunked.parquet"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "results" / "bge_large_en_v1_5_embeddings.f32"
DEFAULT_EMBEDDINGS_META = PROJECT_ROOT / "data" / "results" / "bge_large_en_v1_5_embeddings.meta.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "three_source_oil_gas_metal_mining"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDINGS_META)
    parser.add_argument(
        "--all-sources",
        nargs="+",
        default=["media", "academic"],
        help="Sources to include in full without industry filtering.",
    )
    parser.add_argument(
        "--corporate-industries",
        nargs="+",
        default=["oil_gas", "metal_mining"],
        help="Corporate industries to retain.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_meta(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(path: Path, rows: int, dims: int, model_name: str) -> None:
    payload = {
        "rows": rows,
        "dims": dims,
        "model_name": model_name,
        "dtype": "float32",
        "created_from_subset": True,
        "subset_type": "three_source_with_selected_corporate_industries",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_mask(frame: pd.DataFrame, all_sources: list[str], corporate_industries: list[str]) -> np.ndarray:
    source = frame["source"].fillna("").astype(str)
    industry = frame["industry"].fillna("").astype(str)
    keep_full_sources = source.isin(all_sources)
    keep_corporate = source.eq("corporate") & industry.isin(corporate_industries)
    return (keep_full_sources | keep_corporate).to_numpy()


def load_subset_frame(input_path: Path, all_sources: list[str], corporate_industries: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(input_path), format="parquet")
    table = dataset.to_table(
        filter=(
            ds.field("source").isin(all_sources)
            | ((ds.field("source") == "corporate") & ds.field("industry").isin(corporate_industries))
        )
    )
    return table.to_pandas(types_mapper=pd.ArrowDtype).convert_dtypes(dtype_backend="numpy_nullable")


def build_summary(subset_df: pd.DataFrame, total_rows: int, all_sources: list[str], corporate_industries: list[str]) -> dict[str, object]:
    source_counts = subset_df["source"].fillna("missing").astype(str).value_counts().to_dict()
    corporate_breakdown = (
        subset_df.loc[subset_df["source"].eq("corporate")]
        .groupby("industry", dropna=False)
        .agg(
            rows=("chunk_id", "size"),
            companies=("company", lambda s: s.dropna().astype(str).nunique()),
            unique_documents=("source_doc_id", lambda s: s.dropna().astype(str).nunique()),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .to_dict(orient="records")
    )
    return {
        "subset_type": "three_source_with_selected_corporate_industries",
        "all_sources_kept_in_full": all_sources,
        "corporate_industries_kept": corporate_industries,
        "total_rows_in_full_corpus": total_rows,
        "subset_rows": int(len(subset_df)),
        "subset_source_distribution": source_counts,
        "subset_corporate_industry_breakdown": corporate_breakdown,
        "year_min": int(pd.to_numeric(subset_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(subset_df["year"], errors="coerce").max()),
        "source_doc_count": int(subset_df["source_doc_id"].nunique()),
    }


def write_readme(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Three-source subset",
        "",
        "This subset keeps all rows from the selected non-corporate sources and only the selected",
        "industries within the corporate source.",
        "",
        f"- All-source include list: `{', '.join(summary['all_sources_kept_in_full'])}`",
        f"- Corporate industries kept: `{', '.join(summary['corporate_industries_kept'])}`",
        f"- Subset rows: `{summary['subset_rows']}`",
        "",
        "## Source distribution",
        "",
    ]
    for source_name, count in summary["subset_source_distribution"].items():
        lines.append(f"- `{source_name}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_corpus = args.output_dir / "full_corpus_chunked.parquet"
    output_embeddings = args.output_dir / "bge_large_en_v1_5_embeddings.f32"
    output_meta = args.output_dir / "bge_large_en_v1_5_embeddings.meta.json"
    output_summary = args.output_dir / "subset_summary.json"
    output_readme = args.output_dir / "README.md"

    meta_frame = pq.read_table(args.input, columns=["source", "industry"]).to_pandas()
    mask = build_mask(meta_frame, args.all_sources, args.corporate_industries)
    selected_rows = int(mask.sum())
    total_rows = int(mask.shape[0])
    if selected_rows == 0:
        raise SystemExit("The three-source subset mask returned zero rows.")

    subset_df = load_subset_frame(args.input, args.all_sources, args.corporate_industries)
    subset_df.to_parquet(output_corpus, index=False)

    meta = load_meta(args.embedding_meta)
    dims = int(meta["dims"])
    rows = int(meta["rows"])
    if rows != total_rows:
        raise SystemExit(
            f"Embedding row mismatch: embeddings report {rows} rows but the full corpus has {total_rows} rows."
        )

    full_embeddings = np.memmap(args.embedding_file, dtype="float32", mode="r", shape=(rows, dims))
    if output_embeddings.exists():
        output_embeddings.unlink()
    subset_embeddings = np.memmap(output_embeddings, dtype="float32", mode="w+", shape=(selected_rows, dims))
    subset_embeddings[:] = full_embeddings[mask]
    subset_embeddings.flush()
    write_meta(output_meta, selected_rows, dims, str(meta.get("model_name", "")))

    summary = build_summary(subset_df, total_rows, args.all_sources, args.corporate_industries)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(output_readme, summary)

    print(json.dumps(summary, indent=2))
    print(f"Wrote subset corpus to {output_corpus}")
    print(f"Wrote subset embeddings to {output_embeddings}")


if __name__ == "__main__":
    main()
