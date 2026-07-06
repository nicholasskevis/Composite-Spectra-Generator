#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("hpc_outputs/joint_photometry_spectroscopy")


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _safe_run_label(value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    label = "_".join(part for part in label.split("_") if part)
    if not label:
        raise RuntimeError("--job-name must contain at least one letter or number.")
    return label


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value)


def _run_name(job_name: str, now: datetime | None = None) -> str:
    timestamp = now or datetime.now()
    return f"{timestamp.strftime('%B').lower()}{timestamp.day}_{timestamp:%H%M}_{_safe_run_label(job_name)}"


def _load_manifest_tasks(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"fit_index", "object_id", "COSMOS_ID0"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Manifest must contain {sorted(required)}: {manifest}")
        tasks = [
            {
                "fit_index": row["fit_index"],
                "object_id": row["object_id"],
                "COSMOS_ID0": row["COSMOS_ID0"],
            }
            for row in reader
        ]
    counts = Counter(task["object_id"] for task in tasks)
    duplicates = [object_id for object_id, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Manifest object_id values must be unique; duplicate example: {duplicates[0]!r}")
    return tasks


def _resolve_spectrum_path(spectra_manifest: Path, row: dict[str, str]) -> Path | None:
    raw_path = Path(row.get("spectrum_path", "")).expanduser()
    base_dir = spectra_manifest.parent.resolve()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.append((base_dir / raw_path).resolve())
    parts = raw_path.parts
    if "all_chimera_notebook6_spectra" in parts:
        idx = parts.index("all_chimera_notebook6_spectra")
        rel = Path(*parts[idx + 1 :])
        if rel.parts:
            candidates.append((base_dir / rel).resolve())
    candidates.extend(
        [
            (base_dir / raw_path.name).resolve(),
            (base_dir / "spectra" / raw_path.name).resolve(),
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_spectrum_ids(spectra_manifest: Path) -> tuple[set[str], int]:
    with open(spectra_manifest, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "chimera_id" not in reader.fieldnames:
            raise RuntimeError(f"Spectra manifest must contain chimera_id: {spectra_manifest}")
        ids = set()
        missing_files = 0
        for row in reader:
            chimera_id = str(row.get("chimera_id", "")).strip()
            if not chimera_id or row.get("status", "success") != "success":
                continue
            if _resolve_spectrum_path(spectra_manifest, row) is None:
                missing_files += 1
                continue
            ids.add(chimera_id)
        return ids, missing_files


def _task_stem(task: dict[str, str]) -> str:
    return f"{int(task['fit_index']):05d}_COSMOS{_safe_id(str(task['COSMOS_ID0']))}_{_safe_id(str(task['object_id']))}"


def _filter_missing_tasks(
    tasks: list[dict[str, str]],
    output_dir: Path,
    *,
    rerun_failures: bool,
) -> tuple[list[dict[str, str]], int]:
    selected = []
    skipped = 0
    for task in tasks:
        stem = _task_stem(task)
        success_path = output_dir / "results" / f"{stem}.json"
        failure_path = output_dir / "failures" / f"{stem}.json"
        if success_path.is_file():
            skipped += 1
            continue
        if failure_path.is_file() and not rerun_failures:
            skipped += 1
            continue
        selected.append(task)
    return selected, skipped


def _chunks(values: list[dict[str, str]], chunk_size: int) -> list[tuple[int, int, list[dict[str, str]]]]:
    return [
        (start, min(start + chunk_size, len(values)) - 1, values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]


def _write_task_file(path: Path, tasks: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(f"{task['object_id']}\t{task['fit_index']}\t{task['COSMOS_ID0']}\n")
    tmp_path.replace(path)


def _array_spec(task_count: int) -> str:
    if task_count == 1:
        return "0"
    return f"0-{task_count - 1}"


def _batch_script(
    *,
    job_name: str,
    array: str,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    project_root: Path,
    output_dir: Path,
    manifest: Path,
    spectra_manifest: Path,
    dsps_ssp_fn: Path,
    task_file: Path,
    expected_count: int,
    conda_env: str,
    runner: Path,
    sampler: str,
    n_wave: int,
    optax_steps: int,
    optax_lr: float,
    nuts_warmup: int,
    nuts_samples: int,
    nuts_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    min_valid_spectral_pixels: int,
    progress_bar: bool,
) -> str:
    log_dir = output_dir / "logs"
    progress = " --progress-bar" if progress_bar else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array={array}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err

set -euo pipefail

PROJECT_ROOT={shlex.quote(str(project_root))}
OUTPUT_DIR={shlex.quote(str(output_dir))}
MANIFEST={shlex.quote(str(manifest))}
SPECTRA_MANIFEST={shlex.quote(str(spectra_manifest))}
DSPS_SSP_FN={shlex.quote(str(dsps_ssp_fn))}
TASK_FILE={shlex.quote(str(task_file))}
EXPECTED_COUNT={expected_count}
CONDA_ENV={shlex.quote(conda_env)}
RUNNER={shlex.quote(str(runner))}

mkdir -p "${{OUTPUT_DIR}}/logs" "${{OUTPUT_DIR}}/results" "${{OUTPUT_DIR}}/failures" "${{OUTPUT_DIR}}/sed_pdfs" "${{OUTPUT_DIR}}/corner_pdfs" "${{OUTPUT_DIR}}/trace_pdfs"
cd "${{PROJECT_ROOT}}"

module reset
module load miniconda
conda activate "${{CONDA_ENV}}"

TASK_LINE="$((SLURM_ARRAY_TASK_ID + 1))"
TASK_RECORD="$(sed -n "${{TASK_LINE}}p" "${{TASK_FILE}}")"
if [ -z "${{TASK_RECORD}}" ]; then
  echo "No task found in ${{TASK_FILE}} for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi
IFS=$'\\t' read -r OBJECT_ID FIT_INDEX COSMOS_ID0 <<< "${{TASK_RECORD}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-${{OUTPUT_DIR}}/matplotlib_cache}}"
mkdir -p "${{MPLCONFIGDIR}}"

python "${{RUNNER}}" \\
  --manifest "${{MANIFEST}}" \\
  --spectra-manifest "${{SPECTRA_MANIFEST}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --dsps-ssp-fn "${{DSPS_SSP_FN}}" \\
  --object-id "${{OBJECT_ID}}" \\
  --expected-count "${{EXPECTED_COUNT}}" \\
  --sampler {shlex.quote(sampler)} \\
  --n-wave {n_wave} \\
  --optax-steps {optax_steps} \\
  --optax-lr {optax_lr} \\
  --nuts-warmup {nuts_warmup} \\
  --nuts-samples {nuts_samples} \\
  --nuts-chains {nuts_chains} \\
  --target-accept-prob {target_accept_prob} \\
  --max-tree-depth {max_tree_depth} \\
  --min-valid-spectral-pixels {min_valid_spectral_pixels}{progress}
"""


def _submit(script: str, project_root: Path) -> str:
    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit joint JAXSEDFit photometry+spectroscopy fits for objects with notebook-6 spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--spectra-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--runner", type=Path, default=Path("hpc/run_joint_spectro_manifest_fit.py"))
    parser.add_argument("--max-array-tasks", type=int, default=1000)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--job-name", default="chimera_joint_spectro")
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="04:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="12g")
    parser.add_argument("--conda-env", default="jaxsedfit")
    parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax")
    parser.add_argument("--n-wave", type=int, default=512)
    parser.add_argument("--optax-steps", type=int, default=600)
    parser.add_argument("--optax-lr", type=float, default=5.0e-3)
    parser.add_argument("--nuts-warmup", type=int, default=250)
    parser.add_argument("--nuts-samples", type=int, default=250)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--min-valid-spectral-pixels", type=int, default=50)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    manifest = _resolve_from_root(project_root, args.manifest)
    spectra_manifest = _resolve_from_root(project_root, args.spectra_manifest)
    output_base_dir = _resolve_from_root(project_root, args.output_dir)
    if args.run_dir is not None:
        output_dir = _resolve_from_root(project_root, args.run_dir)
        run_name = output_dir.name
    else:
        run_name = _run_name(args.job_name)
        output_dir = output_base_dir / run_name
    dsps_ssp_fn = _resolve_from_root(project_root, args.dsps_ssp_fn)
    runner = _resolve_from_root(project_root, args.runner)

    for path, label in ((manifest, "manifest"), (spectra_manifest, "spectra manifest"), (dsps_ssp_fn, "DSPS SSP"), (runner, "runner")):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")

    tasks = _load_manifest_tasks(manifest)
    spectrum_ids, missing_spectrum_files = _load_spectrum_ids(spectra_manifest)
    selected = [task for task in tasks if task["object_id"] in spectrum_ids]
    if not selected:
        raise RuntimeError("No manifest rows have matching notebook-6 spectra.")
    selected_without_spectrum = len(tasks) - len(selected)

    for subdir in ("logs", "results", "failures", "sed_pdfs", "corner_pdfs", "trace_pdfs", "slurm_tasks"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if args.only_missing:
        selected, skipped_existing = _filter_missing_tasks(selected, output_dir, rerun_failures=args.rerun_failures)
    else:
        skipped_existing = 0
    if not selected:
        print(f"No tasks to submit. Skipped existing: {skipped_existing}.")
        return 0

    chunks = _chunks(selected, args.max_array_tasks)
    task_dir = output_dir / "slurm_tasks"
    print(f"Project root: {project_root}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest rows: {len(tasks)}")
    print(f"Rows with notebook-6 spectra: {len(selected) + skipped_existing}")
    print(f"Rows without notebook-6 spectra: {selected_without_spectrum}")
    print(f"Spectra manifest rows with missing ECSV files: {missing_spectrum_files}")
    if args.only_missing:
        print(f"Selected missing rows: {len(selected)}")
        print(f"Skipped existing rows: {skipped_existing}")
        print(f"Rerun failures: {args.rerun_failures}")
    print(f"Chunks: {len(chunks)}")

    for chunk_number, (start, end, chunk_tasks) in enumerate(chunks, start=1):
        task_file = task_dir / f"tasks_{start:05d}_{end:05d}.tsv"
        _write_task_file(task_file, chunk_tasks)
        job_name = f"{run_name}-{start:05d}-{end:05d}"
        script = _batch_script(
            job_name=job_name,
            array=_array_spec(len(chunk_tasks)),
            partition=args.partition,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            project_root=project_root,
            output_dir=output_dir,
            manifest=manifest,
            spectra_manifest=spectra_manifest,
            dsps_ssp_fn=dsps_ssp_fn,
            task_file=task_file,
            expected_count=len(tasks),
            conda_env=args.conda_env,
            runner=runner,
            sampler=args.sampler,
            n_wave=args.n_wave,
            optax_steps=args.optax_steps,
            optax_lr=args.optax_lr,
            nuts_warmup=args.nuts_warmup,
            nuts_samples=args.nuts_samples,
            nuts_chains=args.nuts_chains,
            target_accept_prob=args.target_accept_prob,
            max_tree_depth=args.max_tree_depth,
            min_valid_spectral_pixels=args.min_valid_spectral_pixels,
            progress_bar=args.progress_bar,
        )
        print(f"[{chunk_number}/{len(chunks)}] rows {start}-{end}: {len(chunk_tasks)} tasks, task_file={task_file}")
        if args.dry_run:
            print(f"[dry-run] would submit job {job_name!r}")
            continue
        print(_submit(script, project_root))

    if args.dry_run:
        print("Dry run complete; task files were written, but no jobs were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
