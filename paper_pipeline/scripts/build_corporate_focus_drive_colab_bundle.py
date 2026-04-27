#!/usr/bin/env python3
"""Build a Drive-first Colab bundle for the corporate-focused Stage 1/2 workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import time
import zipfile
from pathlib import Path

import pandas as pd


BUNDLE_NAME = "corporate_focus_stage12_colab_drive_with_overrides"
BUNDLE_VERSION = "1.0.0"
EXPECTED_SELECTED_TOPIC_COUNT = 241
EXPECTED_ANNUAL_EVIDENCE_ROW_COUNT = 4344
DEFAULT_STAGE1_BATCH_SIZE = 6
DEFAULT_STAGE1_MAX_NEW_TOKENS = 320
DEFAULT_STAGE1_SAVE_EVERY = 20
DEFAULT_STAGE2_MAX_NEW_TOKENS = 400
DEFAULT_STAGE2_MAX_ATTEMPTS = 4

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_INPUT_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_stage12_input_with_overrides"
DEFAULT_BUNDLE_ROOT = PIPELINE_ROOT / "outputs" / BUNDLE_NAME
DEFAULT_ZIP_PATH = PIPELINE_ROOT / "outputs" / f"{BUNDLE_NAME}.zip"
STAGE1_RUNNER_SOURCE = PIPELINE_ROOT / "colab" / "run_hf_gemma_micro_topic_year_summaries_colab.py"
STAGE2_RUNNER_SOURCE = (
    PIPELINE_ROOT / "colab" / "micro_topic_evolution_full" / "run_hf_gemma_micro_topic_evolution_synthesis_colab.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input-root", type=Path, default=DEFAULT_SOURCE_INPUT_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def to_source_lines(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n")
    if not normalized:
        return []
    return [line + "\n" for line in normalized.splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            f"""
            # Colab Drive-First Run: Corporate-Focused Stage 1 and Stage 2

            Este notebook usa o bundle `{BUNDLE_NAME}` salvo no Google Drive.

            O fluxo foi desenhado para:
            - ler os insumos já preparados localmente
            - executar Stage 1 e Stage 2 usando os runners `.py` do pacote
            - salvar o progresso incrementalmente no próprio Drive
            - retomar a execução a partir dos CSVs já existentes

            Defaults:
            - Stage 1: `batch_size={DEFAULT_STAGE1_BATCH_SIZE}`, `max_new_tokens={DEFAULT_STAGE1_MAX_NEW_TOKENS}`, `save_every={DEFAULT_STAGE1_SAVE_EVERY}`
            - Stage 2: `max_new_tokens={DEFAULT_STAGE2_MAX_NEW_TOKENS}`, `max_attempts={DEFAULT_STAGE2_MAX_ATTEMPTS}`
            """
        ),
        code_cell(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            """
        ),
        code_cell(
            """
            DRIVE_PACKAGE_DIR = ""

            # Se quiser forçar o caminho exato do bundle no Drive, preencha aqui.
            # Exemplo:
            # DRIVE_PACKAGE_DIR = "/content/drive/MyDrive/Colab Notebooks/corporate_focus_stage12_colab_drive_with_overrides"
            """
        ),
        code_cell(
            f"""
            import json
            from pathlib import Path

            BUNDLE_NAME = "{BUNDLE_NAME}"
            MYDRIVE_ROOT = Path("/content/drive/MyDrive")

            def is_valid_bundle_root(path: Path) -> bool:
                manifest_path = path / "colab_bundle_manifest.json"
                if not manifest_path.exists():
                    return False
                try:
                    manifest = json.loads(manifest_path.read_text())
                except Exception:
                    return False
                required_dirs = ["data", "runners", "notebooks", "docs", "colab_outputs"]
                return manifest.get("bundle_name") == BUNDLE_NAME and all((path / name).exists() for name in required_dirs)

            def autodetect_bundle_root(mydrive_root: Path) -> Path:
                candidates = []
                for manifest_path in mydrive_root.rglob("colab_bundle_manifest.json"):
                    candidate_root = manifest_path.parent
                    if is_valid_bundle_root(candidate_root):
                        candidates.append(candidate_root)
                if not candidates:
                    raise FileNotFoundError(
                        f"Could not find a valid {{BUNDLE_NAME}} folder under {{mydrive_root}}."
                    )
                candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
                return candidates[0]

            if DRIVE_PACKAGE_DIR:
                PACKAGE_ROOT = Path(DRIVE_PACKAGE_DIR)
                if not PACKAGE_ROOT.exists():
                    raise FileNotFoundError(f"Configured DRIVE_PACKAGE_DIR does not exist: {{PACKAGE_ROOT}}")
                if not is_valid_bundle_root(PACKAGE_ROOT):
                    raise RuntimeError(f"Configured DRIVE_PACKAGE_DIR is not a valid {{BUNDLE_NAME}} bundle: {{PACKAGE_ROOT}}")
            else:
                PACKAGE_ROOT = autodetect_bundle_root(MYDRIVE_ROOT)

            DATA_DIR = PACKAGE_ROOT / "data"
            RUNNERS_DIR = PACKAGE_ROOT / "runners"
            NOTEBOOKS_DIR = PACKAGE_ROOT / "notebooks"
            DOCS_DIR = PACKAGE_ROOT / "docs"
            OUTPUTS_DIR = PACKAGE_ROOT / "colab_outputs"

            print("PACKAGE_ROOT =", PACKAGE_ROOT)
            print("DATA_DIR =", DATA_DIR)
            print("RUNNERS_DIR =", RUNNERS_DIR)
            print("OUTPUTS_DIR =", OUTPUTS_DIR)
            """
        ),
        code_cell(
            """
            import json
            import shlex
            import subprocess
            import sys
            from pathlib import Path

            REQUIRED_DATA_FILES = [
                "selected_micro_topics.csv",
                "selected_micro_topics_by_subgroup.csv",
                "micro_topic_year_evidence.csv",
                "micro_topic_year_evidence.jsonl",
                "selection_manifest.json",
                "included_corporate_groups.csv",
                "included_noncorporate_groups.csv",
            ]
            REQUIRED_RUNNERS = [
                "run_hf_gemma_micro_topic_year_summaries_colab.py",
                "run_hf_gemma_micro_topic_evolution_synthesis_colab.py",
            ]

            missing_data = [name for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).exists()]
            missing_runners = [name for name in REQUIRED_RUNNERS if not (RUNNERS_DIR / name).exists()]
            if missing_data or missing_runners:
                raise RuntimeError(
                    f"Bundle validation failed. missing_data={{missing_data}} missing_runners={{missing_runners}}"
                )

            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUTS_DIR / "smoke_tests" / "phase_01_year_summaries").mkdir(parents=True, exist_ok=True)
            (OUTPUTS_DIR / "smoke_tests" / "phase_02_inputs_stage1").mkdir(parents=True, exist_ok=True)
            (OUTPUTS_DIR / "smoke_tests" / "phase_02_evolution_summaries").mkdir(parents=True, exist_ok=True)
            (OUTPUTS_DIR / "phase_01_year_summaries").mkdir(parents=True, exist_ok=True)
            (OUTPUTS_DIR / "phase_02_evolution_summaries").mkdir(parents=True, exist_ok=True)

            manifest = json.loads((PACKAGE_ROOT / "colab_bundle_manifest.json").read_text())
            print(json.dumps(manifest, indent=2, ensure_ascii=False))

            def run_cmd(cmd: list[str]) -> None:
                print("RUN:", " ".join(shlex.quote(part) for part in cmd))
                subprocess.run(cmd, check=True)
            """
        ),
        code_cell(
            """
            !pip uninstall -y torch torchvision torchaudio bitsandbytes transformers accelerate
            !pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
            !pip install --no-cache-dir transformers accelerate bitsandbytes huggingface_hub hf_xet pandas tqdm pyarrow openpyxl requests matplotlib
            """
        ),
        code_cell(
            """
            import os
            from huggingface_hub import get_token, login, whoami

            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
            os.environ["HF_HUB_VERBOSITY"] = "warning"

            login()
            token = get_token()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                print("HF token exported to environment.")
                print(whoami())
            else:
                raise RuntimeError("No Hugging Face token found after login.")
            """
        ),
        code_cell(
            f"""
            import accelerate
            import bitsandbytes as bnb
            import huggingface_hub
            import torch
            import transformers

            MODEL_NAME = "google/gemma-4-E4B-it"
            STAGE1_BATCH_SIZE = {DEFAULT_STAGE1_BATCH_SIZE}
            STAGE1_MAX_NEW_TOKENS = {DEFAULT_STAGE1_MAX_NEW_TOKENS}
            STAGE1_SAVE_EVERY = {DEFAULT_STAGE1_SAVE_EVERY}
            STAGE2_MAX_NEW_TOKENS = {DEFAULT_STAGE2_MAX_NEW_TOKENS}
            STAGE2_MAX_ATTEMPTS = {DEFAULT_STAGE2_MAX_ATTEMPTS}

            print("torch", torch.__version__)
            print("torch cuda", torch.version.cuda)
            print("transformers", transformers.__version__)
            print("huggingface_hub", huggingface_hub.__version__)
            print("bitsandbytes", bnb.__version__)
            print("accelerate", accelerate.__version__)
            print("cuda available", torch.cuda.is_available())

            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU detected. Use a GPU runtime in Colab.")

            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print("gpu", gpu_name)
            print("gpu_vram_gb", round(total_vram_gb, 2))

            if "Blackwell" not in gpu_name and total_vram_gb < 80:
                print("Warning: this notebook was tuned for a 96GB Blackwell-class GPU.")

            print("Stage 1 defaults:", STAGE1_BATCH_SIZE, STAGE1_MAX_NEW_TOKENS, STAGE1_SAVE_EVERY)
            print("Stage 2 defaults:", STAGE2_MAX_NEW_TOKENS, STAGE2_MAX_ATTEMPTS)

            PHASE1_SMOKE_DIR = OUTPUTS_DIR / "smoke_tests" / "phase_01_year_summaries"
            PHASE2_SMOKE_STAGE1_DIR = OUTPUTS_DIR / "smoke_tests" / "phase_02_inputs_stage1"
            PHASE2_SMOKE_DIR = OUTPUTS_DIR / "smoke_tests" / "phase_02_evolution_summaries"
            PHASE1_OUTPUT_DIR = OUTPUTS_DIR / "phase_01_year_summaries"
            PHASE2_OUTPUT_DIR = OUTPUTS_DIR / "phase_02_evolution_summaries"
            """
        ),
        markdown_cell("## Smoke test de Stage 1"),
        code_cell(
            """
            run_cmd(
                [
                    sys.executable,
                    str(RUNNERS_DIR / "run_hf_gemma_micro_topic_year_summaries_colab.py"),
                    "--input-csv",
                    str(DATA_DIR / "micro_topic_year_evidence.csv"),
                    "--output-dir",
                    str(PHASE1_SMOKE_DIR),
                    "--model-name",
                    MODEL_NAME,
                    "--batch-size",
                    "1",
                    "--save-every",
                    "1",
                    "--max-new-tokens",
                    str(STAGE1_MAX_NEW_TOKENS),
                    "--max-rows",
                    "1",
                ]
            )
            """
        ),
        code_cell(
            """
            import pandas as pd
            from pandas.errors import EmptyDataError

            print(sorted(path.name for path in PHASE1_SMOKE_DIR.iterdir()))
            smoke_csv = PHASE1_SMOKE_DIR / "micro_topic_year_summaries.csv"
            smoke_err = PHASE1_SMOKE_DIR / "error_log.csv"

            if smoke_csv.exists() and smoke_csv.stat().st_size > 0:
                df = pd.read_csv(smoke_csv)
                print(df.head(5).to_string(index=False))
                print("rows:", len(df))
            else:
                print("No Stage 1 smoke CSV found.")

            if smoke_err.exists() and smoke_err.stat().st_size > 0:
                try:
                    err_df = pd.read_csv(smoke_err)
                    print("error rows:", len(err_df))
                    if not err_df.empty:
                        print(err_df.head(10).to_string(index=False))
                except EmptyDataError:
                    print("error rows: 0")
            """
        ),
        markdown_cell("## Full Stage 1"),
        code_cell(
            """
            run_cmd(
                [
                    sys.executable,
                    str(RUNNERS_DIR / "run_hf_gemma_micro_topic_year_summaries_colab.py"),
                    "--input-csv",
                    str(DATA_DIR / "micro_topic_year_evidence.csv"),
                    "--output-dir",
                    str(PHASE1_OUTPUT_DIR),
                    "--model-name",
                    MODEL_NAME,
                    "--batch-size",
                    str(STAGE1_BATCH_SIZE),
                    "--save-every",
                    str(STAGE1_SAVE_EVERY),
                    "--max-new-tokens",
                    str(STAGE1_MAX_NEW_TOKENS),
                ]
            )
            """
        ),
        code_cell(
            """
            import pandas as pd
            from pandas.errors import EmptyDataError

            print(sorted(path.name for path in PHASE1_OUTPUT_DIR.iterdir()))
            csv_path = PHASE1_OUTPUT_DIR / "micro_topic_year_summaries.csv"
            err_path = PHASE1_OUTPUT_DIR / "error_log.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path)
                print(df.head(5).to_string(index=False))
                print("rows:", len(df))
            else:
                print("No Stage 1 CSV found yet.")

            if err_path.exists() and err_path.stat().st_size > 0:
                try:
                    err_df = pd.read_csv(err_path)
                    print("error rows:", len(err_df))
                    if not err_df.empty:
                        print(err_df.head(20).to_string(index=False))
                except EmptyDataError:
                    print("error rows: 0")
            """
        ),
        markdown_cell("## Smoke test de Stage 2"),
        code_cell(
            """
            import pandas as pd
            from pandas.errors import EmptyDataError

            smoke_inputs_dir = OUTPUTS_DIR / "smoke_tests" / "phase_02_inputs"
            smoke_inputs_dir.mkdir(parents=True, exist_ok=True)

            selected_df = pd.read_csv(DATA_DIR / "selected_micro_topics.csv")
            evidence_df = pd.read_csv(DATA_DIR / "micro_topic_year_evidence.csv")

            smoke_topic = selected_df.head(1).copy()
            smoke_topic_row = smoke_topic.iloc[0]

            smoke_evidence = evidence_df.loc[
                (evidence_df["subgroup"] == smoke_topic_row["subgroup"])
                & (evidence_df["micro_topic_id"].astype(int) == int(smoke_topic_row["micro_topic_id"]))
            ].copy()

            smoke_selected_csv = smoke_inputs_dir / "selected_micro_topics_smoke.csv"
            smoke_evidence_csv = smoke_inputs_dir / "micro_topic_year_evidence_smoke.csv"

            smoke_topic.to_csv(smoke_selected_csv, index=False)
            smoke_evidence.to_csv(smoke_evidence_csv, index=False)

            print("Smoke Stage 2 topic:", smoke_topic_row["subgroup"], int(smoke_topic_row["micro_topic_id"]))
            print("Smoke evidence rows:", len(smoke_evidence))

            run_cmd(
                [
                    sys.executable,
                    str(RUNNERS_DIR / "run_hf_gemma_micro_topic_year_summaries_colab.py"),
                    "--input-csv",
                    str(smoke_evidence_csv),
                    "--output-dir",
                    str(PHASE2_SMOKE_STAGE1_DIR),
                    "--model-name",
                    MODEL_NAME,
                    "--batch-size",
                    "1",
                    "--save-every",
                    "1",
                    "--max-new-tokens",
                    str(STAGE1_MAX_NEW_TOKENS),
                ]
            )

            run_cmd(
                [
                    sys.executable,
                    str(RUNNERS_DIR / "run_hf_gemma_micro_topic_evolution_synthesis_colab.py"),
                    "--selected-topics",
                    str(smoke_selected_csv),
                    "--annual-summaries",
                    str(PHASE2_SMOKE_STAGE1_DIR / "micro_topic_year_summaries.csv"),
                    "--year-evidence",
                    str(smoke_evidence_csv),
                    "--output-dir",
                    str(PHASE2_SMOKE_DIR),
                    "--model-name",
                    MODEL_NAME,
                    "--max-new-tokens",
                    str(STAGE2_MAX_NEW_TOKENS),
                    "--max-attempts",
                    str(STAGE2_MAX_ATTEMPTS),
                    "--max-rows",
                    "1",
                ]
            )
            """
        ),
        code_cell(
            """
            import pandas as pd

            print(sorted(path.name for path in PHASE2_SMOKE_DIR.iterdir()))
            smoke_csv = PHASE2_SMOKE_DIR / "micro_topic_evolution_narratives.csv"
            smoke_err = PHASE2_SMOKE_DIR / "error_log.csv"

            if smoke_csv.exists() and smoke_csv.stat().st_size > 0:
                df = pd.read_csv(smoke_csv)
                print(df.head(5).to_string(index=False))
                print("rows:", len(df))
            else:
                print("No Stage 2 smoke CSV found.")

            if smoke_err.exists() and smoke_err.stat().st_size > 0:
                try:
                    err_df = pd.read_csv(smoke_err)
                    print("error rows:", len(err_df))
                    if not err_df.empty:
                        print(err_df.head(10).to_string(index=False))
                except EmptyDataError:
                    print("error rows: 0")
            """
        ),
        markdown_cell("## Full Stage 2"),
        code_cell(
            """
            run_cmd(
                [
                    sys.executable,
                    str(RUNNERS_DIR / "run_hf_gemma_micro_topic_evolution_synthesis_colab.py"),
                    "--selected-topics",
                    str(DATA_DIR / "selected_micro_topics.csv"),
                    "--annual-summaries",
                    str(PHASE1_OUTPUT_DIR / "micro_topic_year_summaries.csv"),
                    "--year-evidence",
                    str(DATA_DIR / "micro_topic_year_evidence.csv"),
                    "--output-dir",
                    str(PHASE2_OUTPUT_DIR),
                    "--model-name",
                    MODEL_NAME,
                    "--max-new-tokens",
                    str(STAGE2_MAX_NEW_TOKENS),
                    "--max-attempts",
                    str(STAGE2_MAX_ATTEMPTS),
                ]
            )
            """
        ),
        code_cell(
            """
            import pandas as pd
            from pandas.errors import EmptyDataError

            print(sorted(path.name for path in PHASE2_OUTPUT_DIR.iterdir()))
            csv_path = PHASE2_OUTPUT_DIR / "micro_topic_evolution_narratives.csv"
            err_path = PHASE2_OUTPUT_DIR / "error_log.csv"

            if csv_path.exists() and csv_path.stat().st_size > 0:
                df = pd.read_csv(csv_path)
                print(df.head(5).to_string(index=False))
                print("rows:", len(df))
            else:
                print("No Stage 2 CSV found yet.")

            if err_path.exists() and err_path.stat().st_size > 0:
                try:
                    err_df = pd.read_csv(err_path)
                    print("error rows:", len(err_df))
                    if not err_df.empty:
                        print(err_df.head(20).to_string(index=False))
                except EmptyDataError:
                    print("error rows: 0")
            """
        ),
        markdown_cell("## Compactação opcional dos outputs"),
        code_cell(
            """
            import shutil

            archive_base = PACKAGE_ROOT / "corporate_focus_stage12_colab_outputs"
            archive_zip = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUTS_DIR)
            print("Created:", archive_zip)
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_root_readme() -> str:
    return textwrap.dedent(
        f"""
        # {BUNDLE_NAME}

        Este bundle foi preparado para o fluxo Stage 1/2 no Colab com persistência em Google Drive.

        Como usar:
        1. Descompacte este zip localmente.
        2. Suba a pasta `{BUNDLE_NAME}` inteira para `MyDrive`.
        3. Abra o notebook em `notebooks/colab_corporate_focus_stage12_drive_run.ipynb`.
        4. Monte o Drive e rode as células em ordem.

        Este pacote já inclui:
        - o subset com overrides integrados
        - runners `.py` físicos em `runners/`
        - notebook Drive-first
        - documentação do troubleshooting
        """
    ).strip() + "\n"


def build_outputs_readme() -> str:
    return textwrap.dedent(
        """
        # Colab Outputs

        Esta pasta é o destino persistente dos outputs do Colab.

        Estrutura esperada:
        - `smoke_tests/phase_01_year_summaries/`
        - `smoke_tests/phase_02_inputs_stage1/`
        - `smoke_tests/phase_02_evolution_summaries/`
        - `phase_01_year_summaries/`
        - `phase_02_evolution_summaries/`

        Os runners escrevem incrementalmente aqui para que o progresso sobreviva a resets do runtime.
        """
    ).strip() + "\n"


def build_troubleshooting_doc() -> str:
    return textwrap.dedent(
        """
        # Colab Troubleshooting

        ## Problemas que deram erro no fluxo anterior

        1. O pacote antigo assumia `/content` como raiz principal de leitura e escrita.
        2. O notebook gerava os runners Python inline, o que permitia descompasso entre notebook e `.py`.
        3. O `selected_micro_topics.csv` chegou a sair com colunas duplicadas como `share_of_non_outlier_x` e `share_of_non_outlier_y`.
        4. O runner antigo do Stage 2 era frágil no parse/na validação e terminou com `processed_rows=0`.
        5. Algumas células de inspeção usavam heredoc e geraram `NameError: PY`.

        ## O que muda neste bundle

        - os dados e outputs ficam no Google Drive
        - os runners `.py` são arquivos físicos em `runners/`
        - o notebook apenas chama esses arquivos
        - o schema de `selected_micro_topics.csv` já sai corrigido
        - o Stage 2 usa o runner robusto com retries e `--max-attempts`
        - o notebook não usa heredoc nas células de inspeção

        ## Por que o Stage 2 usa 400 / 4

        O Stage 2 ficou mais robusto com:
        - `max_new_tokens = 400`
        - `max_attempts = 4`

        Isso deixa a execução mais lenta, mas reduz bastante o risco de:
        - truncamento do JSON
        - resposta incompleta
        - falha de parse ou schema
        """
    ).strip() + "\n"


def build_bundle_manifest(selection_manifest: dict, source_input_root: Path) -> dict:
    return {
        "bundle_name": BUNDLE_NAME,
        "bundle_version": BUNDLE_VERSION,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_input_root": str(source_input_root),
        "has_manual_overrides": True,
        "selected_micro_topic_count": int(selection_manifest["selected_micro_topic_count"]),
        "annual_evidence_row_count": int(selection_manifest["annual_evidence_row_count"]),
        "relative_paths": {
            "data_dir": "data",
            "runners_dir": "runners",
            "notebooks_dir": "notebooks",
            "docs_dir": "docs",
            "outputs_dir": "colab_outputs",
        },
        "defaults": {
            "stage1": {
                "batch_size": DEFAULT_STAGE1_BATCH_SIZE,
                "max_new_tokens": DEFAULT_STAGE1_MAX_NEW_TOKENS,
                "save_every": DEFAULT_STAGE1_SAVE_EVERY,
            },
            "stage2": {
                "max_new_tokens": DEFAULT_STAGE2_MAX_NEW_TOKENS,
                "max_attempts": DEFAULT_STAGE2_MAX_ATTEMPTS,
                "save_every_topics": 1,
            },
        },
        "recommended_runtime": {
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "vram_gb": 96,
        },
    }


def copy_data_files(source_input_root: Path, bundle_root: Path) -> None:
    data_dir = bundle_root / "data"
    files_to_copy = [
        "selected_micro_topics.csv",
        "selected_micro_topics_by_subgroup.csv",
        "micro_topic_year_evidence.csv",
        "micro_topic_year_evidence.jsonl",
        "selection_manifest.json",
        "included_corporate_groups.csv",
        "included_noncorporate_groups.csv",
    ]
    for filename in files_to_copy:
        copy_file(source_input_root / filename, data_dir / filename)


def validate_source_input_root(source_input_root: Path) -> dict:
    required = [
        source_input_root / "selected_micro_topics.csv",
        source_input_root / "selected_micro_topics_by_subgroup.csv",
        source_input_root / "micro_topic_year_evidence.csv",
        source_input_root / "micro_topic_year_evidence.jsonl",
        source_input_root / "selection_manifest.json",
        source_input_root / "included_corporate_groups.csv",
        source_input_root / "included_noncorporate_groups.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required source input files: {missing}")

    selection_manifest = json.loads((source_input_root / "selection_manifest.json").read_text())
    selected_topics = pd.read_csv(source_input_root / "selected_micro_topics.csv")
    annual_evidence = pd.read_csv(source_input_root / "micro_topic_year_evidence.csv")

    bad_columns = [
        column
        for column in selected_topics.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if bad_columns:
        raise SystemExit(f"selected_micro_topics.csv still has merge suffix columns: {bad_columns}")

    selected_count = int(selected_topics.shape[0])
    annual_count = int(annual_evidence.shape[0])
    if selected_count != EXPECTED_SELECTED_TOPIC_COUNT:
        raise SystemExit(
            f"Unexpected selected topic count: {selected_count} (expected {EXPECTED_SELECTED_TOPIC_COUNT})"
        )
    if annual_count != EXPECTED_ANNUAL_EVIDENCE_ROW_COUNT:
        raise SystemExit(
            f"Unexpected annual evidence row count: {annual_count} (expected {EXPECTED_ANNUAL_EVIDENCE_ROW_COUNT})"
        )

    if int(selection_manifest.get("selected_micro_topic_count", -1)) != EXPECTED_SELECTED_TOPIC_COUNT:
        raise SystemExit("selection_manifest.json has an unexpected selected_micro_topic_count")
    if int(selection_manifest.get("annual_evidence_row_count", -1)) != EXPECTED_ANNUAL_EVIDENCE_ROW_COUNT:
        raise SystemExit("selection_manifest.json has an unexpected annual_evidence_row_count")

    return selection_manifest


def write_bundle(bundle_root: Path, source_input_root: Path) -> None:
    selection_manifest = validate_source_input_root(source_input_root)
    ensure_clean_dir(bundle_root)

    copy_data_files(source_input_root, bundle_root)
    copy_file(STAGE1_RUNNER_SOURCE, bundle_root / "runners" / STAGE1_RUNNER_SOURCE.name)
    copy_file(STAGE2_RUNNER_SOURCE, bundle_root / "runners" / STAGE2_RUNNER_SOURCE.name)

    write_text(bundle_root / "README.md", build_root_readme())
    write_text(bundle_root / "colab_outputs" / "README.md", build_outputs_readme())
    write_text(bundle_root / "docs" / "COLAB_TROUBLESHOOTING.md", build_troubleshooting_doc())
    write_json(bundle_root / "colab_bundle_manifest.json", build_bundle_manifest(selection_manifest, source_input_root))
    write_text(
        bundle_root / "notebooks" / "colab_corporate_focus_stage12_drive_run.ipynb",
        json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n",
    )


def zip_bundle(bundle_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_root.rglob("*")):
            arcname = Path(bundle_root.name) / path.relative_to(bundle_root)
            if path.is_dir():
                continue
            zf.write(path, arcname=str(arcname))


def main() -> None:
    args = parse_args()
    write_bundle(args.bundle_root, args.source_input_root)
    zip_bundle(args.bundle_root, args.zip_path)
    print(args.bundle_root)
    print(args.zip_path)


if __name__ == "__main__":
    main()
