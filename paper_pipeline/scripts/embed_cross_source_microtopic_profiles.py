#!/usr/bin/env python3
"""Embed canonical microtopic profile texts for cross-source matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_source_microtopic_common import OUTPUT_ROOT, configure_logging, write_json
from workflow_common import detect_embedding_model_name, load_config, load_sentence_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--profiles-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    profiles_csv = args.profiles_csv or (args.output_root / "microtopic_profiles.csv")
    profiles = pd.read_csv(profiles_csv)
    if profiles.empty:
        raise SystemExit("No microtopic profiles found to embed.")

    config = load_config(Path("paper_pipeline/config/paper_6topic_pipeline_config.json"))
    model_name = args.model_name or detect_embedding_model_name(config)
    model = load_sentence_transformer(model_name)

    texts = profiles["profile_text"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    embeddings_path = args.output_root / "microtopic_profile_embeddings.npy"
    metadata_path = args.output_root / "microtopic_profile_embedding_metadata.csv"
    manifest_path = args.output_root / "embedding_manifest.json"

    np.save(embeddings_path, embeddings)
    metadata = profiles.copy()
    metadata.insert(0, "embedding_row_index", np.arange(len(metadata), dtype=int))
    metadata.to_csv(metadata_path, index=False)

    write_json(
        manifest_path,
        {
            "profiles_csv": str(profiles_csv),
            "row_count": int(embeddings.shape[0]),
            "dims": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "batch_size": args.batch_size,
            "embedding_model": model_name,
            "normalize_embeddings": True,
            "outputs": {
                "embeddings_npy": str(embeddings_path),
                "embedding_metadata_csv": str(metadata_path),
            },
        },
    )

    print(args.output_root)


if __name__ == "__main__":
    main()
