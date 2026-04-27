#!/usr/bin/env python3
"""Restore downloaded Colab outputs into the local drive-first corporate focus bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path("paper_pipeline")
DEFAULT_BUNDLE_ROOT = PIPELINE_ROOT / "outputs" / "corporate_focus_stage12_colab_drive_with_overrides"
DEFAULT_OUTPUTS_DIR = DEFAULT_BUNDLE_ROOT / "colab_outputs"
DEFAULT_ARCHIVE_DIR = DEFAULT_BUNDLE_ROOT / "restored_download_archives"
DEFAULT_STAGING_DIR = DEFAULT_BUNDLE_ROOT / "restored_download_staging"

EXPECTED_STAGE1_ROWS = 4344
EXPECTED_STAGE2_ROWS = 241


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip",
        type=Path,
        required=True,
        help="Zip file downloaded from Google Drive containing colab_outputs or its contents.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Local corporate focus drive-first bundle root.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the extracted staging directory after a successful restore.",
    )
    return parser.parse_args()


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def print_log(message: str) -> None:
    print(f"[restore_colab_outputs] {message}", flush=True)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def prepare_source_snapshot(source_zip: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = archive_dir / f"{now_stamp()}_{source_zip.name}"
    shutil.copy2(source_zip, snapshot_path)
    return snapshot_path


def extract_snapshot(snapshot_zip: Path, staging_root: Path) -> Path:
    extract_dir = staging_root / snapshot_zip.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(snapshot_zip) as handle:
        handle.extractall(extract_dir)
    return extract_dir


def locate_colab_outputs_root(extract_dir: Path) -> Path:
    direct = extract_dir / "colab_outputs"
    if direct.is_dir():
        return direct

    candidates = sorted(
        path for path in extract_dir.rglob("colab_outputs") if path.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidate_list = "\n".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple extracted colab_outputs directories found; restore source is ambiguous:\n"
            f"{candidate_list}"
        )

    required_children = {"phase_01_year_summaries", "phase_02_evolution_summaries"}
    if required_children.issubset({path.name for path in extract_dir.iterdir()}):
        return extract_dir

    for path in extract_dir.rglob("*"):
        if not path.is_dir():
            continue
        child_names = {child.name for child in path.iterdir()}
        if required_children.issubset(child_names):
            return path

    raise RuntimeError(
        "Could not locate an extracted colab_outputs root containing "
        "`phase_01_year_summaries` and `phase_02_evolution_summaries`."
    )


def validate_required_files(colab_outputs_root: Path) -> dict[str, Path]:
    phase1_dir = resolve_stage_dir(
        colab_outputs_root=colab_outputs_root,
        canonical_name="phase_01_year_summaries",
        required_filenames=[
            "micro_topic_year_summaries.csv",
            "run_manifest.json",
            "raw_responses.jsonl",
            "prompt_template.txt",
            "error_log.csv",
        ],
    )
    phase2_dir = resolve_stage_dir(
        colab_outputs_root=colab_outputs_root,
        canonical_name="phase_02_evolution_summaries",
        required_filenames=[
            "micro_topic_evolution_narratives.csv",
            "run_manifest.json",
            "raw_responses.jsonl",
            "prompt_template.txt",
            "error_log.csv",
        ],
    )

    required = {
        "phase1_dir": phase1_dir,
        "phase2_dir": phase2_dir,
        "stage1_csv": phase1_dir / "micro_topic_year_summaries.csv",
        "stage2_csv": phase2_dir / "micro_topic_evolution_narratives.csv",
        "stage1_manifest": phase1_dir / "run_manifest.json",
        "stage2_manifest": phase2_dir / "run_manifest.json",
        "stage1_raw": phase1_dir / "raw_responses.jsonl",
        "stage2_raw": phase2_dir / "raw_responses.jsonl",
        "stage1_prompt": phase1_dir / "prompt_template.txt",
        "stage2_prompt": phase2_dir / "prompt_template.txt",
        "stage2_error": phase2_dir / "error_log.csv",
    }
    for label, path in required.items():
        ensure_exists(path, label)
    return required


def resolve_stage_dir(
    colab_outputs_root: Path,
    canonical_name: str,
    required_filenames: list[str],
) -> Path:
    candidates = [
        colab_outputs_root / canonical_name,
        colab_outputs_root / f"{canonical_name}_runA",
    ]
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in required_filenames):
            return candidate
    for candidate in candidates:
        if any((candidate / filename).exists() for filename in required_filenames):
            missing = [filename for filename in required_filenames if not (candidate / filename).exists()]
            print_log(
                f"Candidate {candidate} is incomplete; missing {missing}. "
                "Will keep searching for a more complete source."
            )
    candidate_list = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a complete stage directory with required files "
        f"{required_filenames} in any expected path:\n{candidate_list}"
    )


def count_csv_rows(path: Path) -> int:
    frame = pd.read_csv(path)
    return int(len(frame))


def count_error_rows(path: Path) -> int:
    if path.stat().st_size == 0:
        return 0
    frame = pd.read_csv(path)
    return int(len(frame))


def validate_counts(required: dict[str, Path]) -> dict[str, int]:
    stage1_rows = count_csv_rows(required["stage1_csv"])
    stage1_error_rows = count_error_rows(required["phase1_dir"] / "error_log.csv")
    stage2_rows = count_csv_rows(required["stage2_csv"])
    stage2_error_rows = count_error_rows(required["stage2_error"])

    if stage1_rows != EXPECTED_STAGE1_ROWS and (stage1_rows + stage1_error_rows) != EXPECTED_STAGE1_ROWS:
        raise RuntimeError(
            "Stage 1 coverage mismatch: expected "
            f"{EXPECTED_STAGE1_ROWS}, got rows={stage1_rows} and error_rows={stage1_error_rows}"
        )
    if stage1_rows != EXPECTED_STAGE1_ROWS:
        print_log(
            "Stage 1 annual summaries are not fully complete in the downloaded archive, "
            f"but rows + error_rows still cover the expected total "
            f"({stage1_rows} + {stage1_error_rows} = {EXPECTED_STAGE1_ROWS})."
        )
    if stage2_rows != EXPECTED_STAGE2_ROWS:
        raise RuntimeError(
            f"Stage 2 row count mismatch: expected {EXPECTED_STAGE2_ROWS}, got {stage2_rows}"
        )
    if stage2_error_rows != 0:
        raise RuntimeError(
            f"Stage 2 error_log.csv still has {stage2_error_rows} row(s); expected 0"
        )

    return {
        "stage1_rows": stage1_rows,
        "stage1_error_rows": stage1_error_rows,
        "stage2_rows": stage2_rows,
        "stage2_error_rows": stage2_error_rows,
    }


def replace_dir(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def sync_restored_outputs(colab_outputs_root: Path, active_outputs_dir: Path) -> None:
    active_outputs_dir.mkdir(parents=True, exist_ok=True)
    phase1_src = resolve_stage_dir(
        colab_outputs_root=colab_outputs_root,
        canonical_name="phase_01_year_summaries",
        required_filenames=[
            "micro_topic_year_summaries.csv",
            "run_manifest.json",
            "raw_responses.jsonl",
            "prompt_template.txt",
            "error_log.csv",
        ],
    )
    phase2_src = resolve_stage_dir(
        colab_outputs_root=colab_outputs_root,
        canonical_name="phase_02_evolution_summaries",
        required_filenames=[
            "micro_topic_evolution_narratives.csv",
            "run_manifest.json",
            "raw_responses.jsonl",
            "prompt_template.txt",
            "error_log.csv",
        ],
    )

    replace_dir(phase1_src, active_outputs_dir / "phase_01_year_summaries")
    replace_dir(phase2_src, active_outputs_dir / "phase_02_evolution_summaries")

    smoke_tests_src = colab_outputs_root / "smoke_tests"
    if smoke_tests_src.exists():
        replace_dir(smoke_tests_src, active_outputs_dir / "smoke_tests")

    # Preserve any extra top-level run folders from the downloaded archive for continuity.
    for extra_dir in sorted(path for path in colab_outputs_root.iterdir() if path.is_dir()):
        if extra_dir.name in {"phase_01_year_summaries", "phase_02_evolution_summaries", "smoke_tests"}:
            continue
        if extra_dir.name.endswith("_runA"):
            replace_dir(extra_dir, active_outputs_dir / extra_dir.name)


def write_restore_manifest(
    bundle_root: Path,
    snapshot_zip: Path,
    staging_dir: Path,
    active_outputs_dir: Path,
    counts: dict[str, int],
) -> Path:
    warnings: list[str] = []
    if counts["stage1_rows"] != EXPECTED_STAGE1_ROWS:
        warnings.append(
            "Stage 1 restored from a run with non-zero error_log coverage; "
            f"rows={counts['stage1_rows']} error_rows={counts['stage1_error_rows']}."
        )
    manifest = {
        "restored_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_zip": str(snapshot_zip),
        "staging_dir": str(staging_dir),
        "active_outputs_dir": str(active_outputs_dir),
        "expected_counts": {
            "stage1_rows": EXPECTED_STAGE1_ROWS,
            "stage1_error_rows": 0,
            "stage2_rows": EXPECTED_STAGE2_ROWS,
            "stage2_error_rows": 0,
        },
        "observed_counts": counts,
        "warnings": warnings,
    }
    manifest_path = bundle_root / "restore_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()

    ensure_exists(args.source_zip, "source zip")
    ensure_exists(args.bundle_root, "bundle root")

    active_outputs_dir = args.bundle_root / "colab_outputs"
    archive_dir = args.bundle_root / "restored_download_archives"
    staging_dir_root = args.bundle_root / "restored_download_staging"

    print_log(f"Creating immutable archive snapshot from {args.source_zip}")
    snapshot_zip = prepare_source_snapshot(args.source_zip, archive_dir)
    print_log(f"Snapshot stored at {snapshot_zip}")

    print_log(f"Extracting snapshot into staging under {staging_dir_root}")
    extract_dir = extract_snapshot(snapshot_zip, staging_dir_root)
    colab_outputs_root = locate_colab_outputs_root(extract_dir)
    print_log(f"Using extracted colab_outputs root: {colab_outputs_root}")

    required = validate_required_files(colab_outputs_root)
    counts = validate_counts(required)
    print_log(
        "Validated counts: "
        f"stage1_rows={counts['stage1_rows']} "
        f"stage2_rows={counts['stage2_rows']} "
        f"stage2_error_rows={counts['stage2_error_rows']}"
    )

    print_log(f"Syncing restored outputs into active bundle outputs at {active_outputs_dir}")
    sync_restored_outputs(colab_outputs_root, active_outputs_dir)

    manifest_path = write_restore_manifest(
        bundle_root=args.bundle_root,
        snapshot_zip=snapshot_zip,
        staging_dir=colab_outputs_root,
        active_outputs_dir=active_outputs_dir,
        counts=counts,
    )
    print_log(f"Wrote restore manifest to {manifest_path}")

    if not args.keep_staging:
        shutil.rmtree(extract_dir)
        print_log(f"Removed staging extract directory {extract_dir}")

    print_log("Restore completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
