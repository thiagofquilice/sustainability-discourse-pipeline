#!/usr/bin/env python3
"""Create an industry-filtered corpus subset and aligned embedding subset."""

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "results" / "subsets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-meta", type=Path, default=DEFAULT_EMBEDDINGS_META)
    parser.add_argument("--source", type=str, default="corporate")
    parser.add_argument("--industries", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def slugify_industries(values: list[str]) -> str:
    cleaned = ["_".join(part for part in value.strip().lower().split()) for value in values]
    return "_".join(cleaned)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    slug = slugify_industries(args.industries)
    return DEFAULT_OUTPUT_ROOT / f"{args.source}_{slug}"


def load_embedding_meta(path: Path) -> dict[str, object]:
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


def build_mask(input_path: Path, source: str, industries: list[str]) -> np.ndarray:
    meta_frame = pq.read_table(input_path, columns=["source", "industry"]).to_pandas()
    source_mask = meta_frame["source"].fillna("").astype(str).eq(source)
    industry_mask = meta_frame["industry"].fillna("").astype(str).isin(industries)
    return (source_mask & industry_mask).to_numpy()


def load_subset_frame(input_path: Path, source: str, industries: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(input_path), format="parquet")
    table = dataset.to_table(
        filter=(ds.field("source") == source) & ds.field("industry").isin(industries)
    )
    subset_df = table.to_pandas(types_mapper=pd.ArrowDtype).convert_dtypes(dtype_backend="numpy_nullable")
    return subset_df


def build_summary(
    source: str,
    industries: list[str],
    total_rows: int,
    subset_df: pd.DataFrame,
) -> dict[str, object]:
    per_industry = (
        subset_df.groupby("industry", dropna=False)
        .agg(
            companies=("company", lambda s: s.dropna().astype(str).nunique()),
            unique_documents=("source_doc_id", lambda s: s.dropna().astype(str).nunique()),
            chunks=("source_doc_id", "size"),
        )
        .reset_index()
        .sort_values("chunks", ascending=False)
    )
    per_industry_records = per_industry.to_dict(orient="records")
    docs_per_original = subset_df.groupby("source_doc_id", dropna=False)["chunk_id"].nunique()
    return {
        "source": source,
        "industries": industries,
        "total_rows_in_full_corpus": total_rows,
        "subset_rows": int(len(subset_df)),
        "subset_original_docs": int(subset_df["source_doc_id"].nunique()),
        "subset_companies": int(subset_df["company"].nunique()),
        "year_min": int(pd.to_numeric(subset_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(subset_df["year"], errors="coerce").max()),
        "mean_chunks_per_original_doc": round(float(docs_per_original.mean()), 6),
        "median_chunks_per_original_doc": float(docs_per_original.median()),
        "max_chunks_per_original_doc": int(docs_per_original.max()),
        "per_industry": per_industry_records,
    }


def write_readme(
    path: Path,
    *,
    source: str,
    industries: list[str],
    corpus_path: Path,
    embeddings_path: Path,
    meta_path: Path,
    summary_path: Path,
) -> None:
    lines = [
        "# Industry subset",
        "",
        f"Source: `{source}`",
        f"Industries: `{', '.join(industries)}`",
        "",
        "This subset is compatible with the standard BERTopic pipeline used in the project.",
        "",
        "## Artifacts",
        "",
        f"- Corpus: `{corpus_path}`",
        f"- Embeddings: `{embeddings_path}`",
        f"- Embedding meta: `{meta_path}`",
        f"- Summary: `{summary_path}`",
        "",
        "## Example usage",
        "",
        "```bash",
        ".venv/bin/python scripts/run_topic_model.py \\",
        f"  --input {corpus_path} \\",
        "  --reuse-embeddings \\",
        f"  --embedding-file {embeddings_path} \\",
        f"  --embedding-meta {meta_path} \\",
        "  --results-dir data/results/experiments/oil_gas_metal_mining_example \\",
        "  --corpus-output data/results/experiments/oil_gas_metal_mining_example/corpus_with_topics.parquet \\",
        "  --topic-info-output data/results/experiments/oil_gas_metal_mining_example/topic_info.parquet \\",
        "  --topic-docs-output data/results/experiments/oil_gas_metal_mining_example/topic_top_documents.parquet \\",
        "  --representative-docs-output data/results/experiments/oil_gas_metal_mining_example/topic_representative_docs.parquet \\",
        "  --model-dir data/results/experiments/oil_gas_metal_mining_example/bertopic_model",
        "```",
        "",
        "Any downstream script that accepts explicit corpus/results paths can be pointed at the outputs from this subset run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    industries = [industry.strip() for industry in args.industries if industry.strip()]
    if not industries:
        raise SystemExit("At least one non-empty industry must be provided.")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_corpus = output_dir / "full_corpus_chunked.parquet"
    output_embeddings = output_dir / "bge_large_en_v1_5_embeddings.f32"
    output_meta = output_dir / "bge_large_en_v1_5_embeddings.meta.json"
    output_summary = output_dir / "subset_summary.json"
    output_readme = output_dir / "README.md"

    mask = build_mask(args.input, args.source, industries)
    selected_rows = int(mask.sum())
    total_rows = int(mask.shape[0])
    if selected_rows == 0:
        raise SystemExit(f"No rows found for source={args.source!r} and industries={industries!r}")

    subset_df = load_subset_frame(args.input, args.source, industries)
    subset_df.to_parquet(output_corpus, index=False)

    meta = load_embedding_meta(args.embedding_meta)
    dims = int(meta["dims"])
    rows = int(meta["rows"])
    if rows != total_rows:
        raise SystemExit(
            f"Embedding row mismatch: embeddings have {rows} rows but corpus has {total_rows}"
        )

    full_embeddings = np.memmap(args.embedding_file, dtype="float32", mode="r", shape=(rows, dims))
    if output_embeddings.exists():
        output_embeddings.unlink()
    subset_embeddings = np.memmap(
        output_embeddings,
        dtype="float32",
        mode="w+",
        shape=(selected_rows, dims),
    )
    subset_embeddings[:] = full_embeddings[mask]
    subset_embeddings.flush()
    write_embedding_meta(output_meta, selected_rows, dims, str(meta.get("model_name", "")))

    summary = build_summary(args.source, industries, total_rows, subset_df)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(
        output_readme,
        source=args.source,
        industries=industries,
        corpus_path=output_corpus,
        embeddings_path=output_embeddings,
        meta_path=output_meta,
        summary_path=output_summary,
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote subset corpus to {output_corpus}")
    print(f"Wrote subset embeddings to {output_embeddings}")
    print(f"Wrote subset README to {output_readme}")


if __name__ == "__main__":
    main()
