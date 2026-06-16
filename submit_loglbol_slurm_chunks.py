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


DEFAULT_OUTPUT_DIR = Path("hpc_outputs/loglbol_mass_retrieval")


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


def _run_name(job_name: str, now: datetime | None = None) -> str:
    timestamp = now or datetime.now()
    return f"{timestamp.strftime('%B').lower()}{timestamp.day}_{timestamp:%H%M}_{_safe_run_label(job_name)}"


def _load_object_ids(manifest: Path) -> list[str]:
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "object_id" not in reader.fieldnames:
            raise RuntimeError(f"Manifest must contain an object_id column: {manifest}")
        object_ids = [row["object_id"] for row in reader]

    missing_count = sum(1 for object_id in object_ids if not object_id)
    if missing_count:
        raise RuntimeError(f"Manifest contains {missing_count} rows with empty object_id values.")

    counts = Counter(object_ids)
    duplicates = [object_id for object_id, count in counts.items() if count > 1]
    if duplicates:
        preview = ", ".join(repr(object_id) for object_id in duplicates[:5])
        raise RuntimeError(
            f"Manifest object_id values must be unique; found {len(duplicates)} duplicated values. "
            f"Examples: {preview}"
        )

    return object_ids


def _chunks(values: list[str], chunk_size: int) -> list[tuple[int, int, list[str]]]:
    return [
        (start, min(start + chunk_size, len(values)) - 1, values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]


def _write_task_file(path: Path, object_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
        for object_id in object_ids:
            fh.write(f"{object_id}\n")
    tmp_path.replace(path)


def _array_spec(task_count: int) -> str:
    if task_count < 1:
        raise ValueError("task_count must be positive")
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
    dsps_ssp_fn: Path,
    task_file: Path,
    expected_count: int,
    conda_env: str,
) -> str:
    log_dir = output_dir / "logs"
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
DSPS_SSP_FN={shlex.quote(str(dsps_ssp_fn))}
MANIFEST={shlex.quote(str(manifest))}
TASK_FILE={shlex.quote(str(task_file))}
EXPECTED_COUNT={expected_count}
CONDA_ENV={shlex.quote(conda_env)}

mkdir -p "${{OUTPUT_DIR}}/logs" "${{OUTPUT_DIR}}/results" "${{OUTPUT_DIR}}/failures" "${{OUTPUT_DIR}}/sed_pdfs" "${{OUTPUT_DIR}}/corner_pdfs" "${{OUTPUT_DIR}}/trace_pdfs"
cd "${{PROJECT_ROOT}}"

module reset
module load miniconda
conda activate "${{CONDA_ENV}}"

if [ ! -f "${{MANIFEST}}" ]; then
  echo "Manifest not found: ${{MANIFEST}}" >&2
  exit 1
fi

if [ ! -f "${{TASK_FILE}}" ]; then
  echo "Task file not found: ${{TASK_FILE}}" >&2
  exit 1
fi

TASK_LINE="$((SLURM_ARRAY_TASK_ID + 1))"
OBJECT_ID="$(sed -n "${{TASK_LINE}}p" "${{TASK_FILE}}")"
if [ -z "${{OBJECT_ID}}" ]; then
  echo "No object_id found in ${{TASK_FILE}} for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-${{OUTPUT_DIR}}/matplotlib_cache}}"
mkdir -p "${{MPLCONFIGDIR}}"

SAMPLER="${{SAMPLER:-optax+nuts}}"
NS_ARGS=()
ns_flag_enabled() {{
  case "${{1,,}}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}}
if [ -n "${{NS_LIVE_POINTS:-}}" ]; then
  NS_ARGS+=(--ns-live-points "${{NS_LIVE_POINTS}}")
fi
if [ -n "${{NS_MAX_SAMPLES:-}}" ]; then
  NS_ARGS+=(--ns-max-samples "${{NS_MAX_SAMPLES}}")
fi
if [ -n "${{NS_DLOGZ:-}}" ]; then
  NS_ARGS+=(--ns-dlogz "${{NS_DLOGZ}}")
fi
if ns_flag_enabled "${{NS_DIFFICULT_MODEL:-}}"; then
  NS_ARGS+=(--ns-difficult-model)
fi
if ns_flag_enabled "${{NS_PARAMETER_ESTIMATION:-}}"; then
  NS_ARGS+=(--ns-parameter-estimation)
fi
if [ -n "${{NS_NUM_PARALLEL_WORKERS:-}}" ]; then
  NS_ARGS+=(--ns-num-parallel-workers "${{NS_NUM_PARALLEL_WORKERS}}")
fi
if [ -n "${{NS_INIT_EFFICIENCY_THRESHOLD:-}}" ]; then
  NS_ARGS+=(--ns-init-efficiency-threshold "${{NS_INIT_EFFICIENCY_THRESHOLD}}")
fi
if [ -n "${{NS_MAX_LIKELIHOOD_EVALS:-}}" ]; then
  NS_ARGS+=(--ns-max-likelihood-evals "${{NS_MAX_LIKELIHOOD_EVALS}}")
fi
if [ -n "${{NS_EFFICIENCY_THRESHOLD:-}}" ]; then
  NS_ARGS+=(--ns-efficiency-threshold "${{NS_EFFICIENCY_THRESHOLD}}")
fi

python "${{PROJECT_ROOT}}/run_manifest_fit.py" \\
  --manifest "${{MANIFEST}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --dsps-ssp-fn "${{DSPS_SSP_FN}}" \\
  --object-id "${{OBJECT_ID}}" \\
  --expected-count "${{EXPECTED_COUNT}}" \\
  --sampler "${{SAMPLER}}" \\
  --optax-steps "${{OPTAX_STEPS:-300}}" \\
  --optax-lr "${{OPTAX_LR:-1.0e-2}}" \\
  --nuts-warmup "${{NUTS_WARMUP:-500}}" \\
  --nuts-samples "${{NUTS_SAMPLES:-300}}" \\
  --nuts-chains "${{NUTS_CHAINS:-1}}" \\
  --target-accept-prob "${{TARGET_ACCEPT_PROB:-0.85}}" \\
  "${{NS_ARGS[@]}}"
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
        description="Submit logLbol manifest fits as Slurm arrays chunked by object_id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory; a timestamped run subdirectory is created inside it.",
    )
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--max-array-tasks", type=int, default=10_000)
    parser.add_argument("--job-name", default="nicholas", help="Label appended to the timestamped Slurm job and run directory name.")
    parser.add_argument("--partition", default="day_amd")
    parser.add_argument("--time", default="02:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="8g")
    parser.add_argument("--conda-env", default="nicholas")
    parser.add_argument("--dry-run", action="store_true", help="Write task files and print sbatch plans without submitting.")
    args = parser.parse_args(argv)

    if args.max_array_tasks < 1:
        raise RuntimeError("--max-array-tasks must be at least 1.")
    if args.cpus_per_task < 1:
        raise RuntimeError("--cpus-per-task must be at least 1.")

    project_root = Path.cwd().resolve()
    manifest = _resolve_from_root(project_root, args.manifest)
    output_base_dir = _resolve_from_root(project_root, args.output_dir)
    run_name = _run_name(args.job_name)
    output_dir = output_base_dir / run_name
    dsps_ssp_fn = _resolve_from_root(project_root, args.dsps_ssp_fn)

    if not manifest.is_file():
        raise RuntimeError(f"Manifest not found: {manifest}")
    if not dsps_ssp_fn.is_file():
        raise RuntimeError(f"DSPS SSP file not found: {dsps_ssp_fn}")

    object_ids = _load_object_ids(manifest)
    if not object_ids:
        raise RuntimeError(f"Manifest contains no rows: {manifest}")

    task_dir = output_dir / "slurm_tasks"
    for subdir in ("logs", "results", "failures", "sed_pdfs", "corner_pdfs", "trace_pdfs", "slurm_tasks"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    chunks = _chunks(object_ids, args.max_array_tasks)
    print(f"Project root: {project_root}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest rows: {len(object_ids)}")
    print(f"Max array tasks: {args.max_array_tasks}")
    print(f"Chunks: {len(chunks)}")

    for chunk_number, (start, end, chunk_object_ids) in enumerate(chunks, start=1):
        task_file = task_dir / f"object_ids_{start:05d}_{end:05d}.txt"
        _write_task_file(task_file, chunk_object_ids)

        job_name = f"{run_name}-{start:05d}-{end:05d}"
        array = _array_spec(len(chunk_object_ids))
        script = _batch_script(
            job_name=job_name,
            array=array,
            partition=args.partition,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            project_root=project_root,
            output_dir=output_dir,
            manifest=manifest,
            dsps_ssp_fn=dsps_ssp_fn,
            task_file=task_file,
            expected_count=len(object_ids),
            conda_env=args.conda_env,
        )

        print(
            f"[{chunk_number}/{len(chunks)}] rows {start}-{end}: "
            f"{len(chunk_object_ids)} tasks, array={array}, task_file={task_file}"
        )
        if args.dry_run:
            print(f"[dry-run] would submit job {job_name!r}")
            continue

        output = _submit(script, project_root)
        print(output)

    if args.dry_run:
        print("Dry run complete; task files were written, but no jobs were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
