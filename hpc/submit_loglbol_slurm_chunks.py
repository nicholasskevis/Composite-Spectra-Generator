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
DEFAULT_GRAHSP_SAMPLER_SCRIPT = Path("../GRAHSP/GRAHSP-run/dualsampler.py")
DEFAULT_GRAHSP_CIGALE_ROOT = Path("../GRAHSP/GRAHSP")
DEFAULT_GRAHSP_MASS_MAX = 13.0
BACKEND_ALIASES = {
    "jaxsed": "grahspj",
    "jaxsedfit": "grahspj",
    "grahspj": "grahspj",
    "grahsp": "grahsp",
}


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _resolve_first_existing(project_root: Path, paths: list[Path], *, kind: str) -> Path:
    resolved = [_resolve_from_root(project_root, path) for path in paths]
    for path in resolved:
        if kind == "file" and path.is_file():
            return path
        if kind == "dir" and path.is_dir():
            return path
    return resolved[0]


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


def _normalize_backend(value: str) -> str:
    try:
        return BACKEND_ALIASES[value.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKEND_ALIASES))
        raise ValueError(f"Unsupported backend {value!r}; choose one of: {choices}") from exc


def _load_tasks(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required_columns = {"fit_index", "object_id", "COSMOS_ID0"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            columns = ", ".join(sorted(required_columns))
            raise RuntimeError(f"Manifest must contain columns {columns}: {manifest}")
        tasks = [
            {
                "fit_index": row["fit_index"],
                "object_id": row["object_id"],
                "COSMOS_ID0": row["COSMOS_ID0"],
            }
            for row in reader
        ]

    missing_count = sum(1 for task in tasks if not task["object_id"] or not task["fit_index"] or not task["COSMOS_ID0"])
    if missing_count:
        raise RuntimeError(f"Manifest contains {missing_count} rows with empty fit_index, object_id, or COSMOS_ID0 values.")

    object_ids = [task["object_id"] for task in tasks]
    counts = Counter(object_ids)
    duplicates = [object_id for object_id, count in counts.items() if count > 1]
    if duplicates:
        preview = ", ".join(repr(object_id) for object_id in duplicates[:5])
        raise RuntimeError(
            f"Manifest object_id values must be unique; found {len(duplicates)} duplicated values. "
            f"Examples: {preview}"
        )

    return tasks


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
    if task_count < 1:
        raise ValueError("task_count must be positive")
    if task_count == 1:
        return "0"
    return f"0-{task_count - 1}"


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
    backend: str,
    grahsp_runner: Path,
    grahsp_sampler_script: Path,
    grahsp_cigale_root: Path,
    grahsp_mass_max: float,
) -> str:
    backend = _normalize_backend(backend)
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
BACKEND={shlex.quote(backend)}
GRAHSP_RUNNER={shlex.quote(str(grahsp_runner))}
GRAHSP_SAMPLER_SCRIPT={shlex.quote(str(grahsp_sampler_script))}
GRAHSP_CIGALE_ROOT={shlex.quote(str(grahsp_cigale_root))}
GRAHSP_MASS_MAX={grahsp_mass_max:g}

mkdir -p "${{OUTPUT_DIR}}/logs" "${{OUTPUT_DIR}}/results" "${{OUTPUT_DIR}}/failures" "${{OUTPUT_DIR}}/sed_pdfs" "${{OUTPUT_DIR}}/sed_lum_pdfs" "${{OUTPUT_DIR}}/corner_pdfs" "${{OUTPUT_DIR}}/trace_pdfs" "${{OUTPUT_DIR}}/posteriors_pdfs" "${{OUTPUT_DIR}}/derived_pdfs" "${{OUTPUT_DIR}}/sed_csvs" "${{OUTPUT_DIR}}/photometry_csvs"
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
TASK_RECORD="$(sed -n "${{TASK_LINE}}p" "${{TASK_FILE}}")"
if [ -z "${{TASK_RECORD}}" ]; then
  echo "No task found in ${{TASK_FILE}} for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi
IFS=$'\\t' read -r OBJECT_ID FIT_INDEX COSMOS_ID0 <<< "${{TASK_RECORD}}"
if [ -z "${{OBJECT_ID}}" ] || [ -z "${{FIT_INDEX}}" ] || [ -z "${{COSMOS_ID0}}" ]; then
  echo "Malformed task record in ${{TASK_FILE}} line ${{TASK_LINE}}: ${{TASK_RECORD}}" >&2
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

case "${{BACKEND}}" in
  grahspj)
    python "${{PROJECT_ROOT}}/hpc/run_manifest_fit.py" \\
      --manifest "${{MANIFEST}}" \\
      --output-dir "${{OUTPUT_DIR}}" \\
      --dsps-ssp-fn "${{DSPS_SSP_FN}}" \\
      --object-id "${{OBJECT_ID}}" \\
      --expected-count "${{EXPECTED_COUNT}}" \\
      --backend grahspj \\
      --sampler "${{SAMPLER}}" \\
      --optax-steps "${{OPTAX_STEPS:-300}}" \\
      --optax-lr "${{OPTAX_LR:-1.0e-2}}" \\
      --nuts-warmup "${{NUTS_WARMUP:-500}}" \\
      --nuts-samples "${{NUTS_SAMPLES:-300}}" \\
      --nuts-chains "${{NUTS_CHAINS:-1}}" \\
      --target-accept-prob "${{TARGET_ACCEPT_PROB:-0.85}}" \\
      "${{NS_ARGS[@]}}"
    ;;
  grahsp)
    if [ ! -f "${{GRAHSP_RUNNER}}" ]; then
      echo "GRAHSP runner not found: ${{GRAHSP_RUNNER}}" >&2
      exit 1
    fi
    python "${{GRAHSP_RUNNER}}" \\
      --manifest "${{MANIFEST}}" \\
      --output-dir "${{OUTPUT_DIR}}" \\
      --fit-index "${{FIT_INDEX}}" \\
      --expected-count "${{EXPECTED_COUNT}}" \\
      --sampler-script "${{GRAHSP_SAMPLER_SCRIPT}}" \\
      --cigale-root "${{GRAHSP_CIGALE_ROOT}}" \\
      --cores "${{SLURM_CPUS_PER_TASK:-1}}" \\
      --num-live-points "${{GRAHSP_NUM_LIVE_POINTS:-800}}" \\
      --num-posterior-samples "${{GRAHSP_NUM_POSTERIOR_SAMPLES:-3000}}" \\
      --cache-max "${{GRAHSP_CACHE_MAX:-5000}}"
    ;;
  *)
    echo "Unsupported BACKEND=${{BACKEND}}" >&2
    exit 1
    ;;
esac
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
    parser.add_argument("--max-array-tasks", type=int, default=4_000)
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact existing or new run directory. Overrides timestamped --output-dir/<run_name>.")
    parser.add_argument("--only-missing", action="store_true", help="Submit only rows without existing result/failure JSONs in the selected run directory.")
    parser.add_argument("--rerun-failures", action="store_true", help="With --only-missing, include rows that already have failure JSONs.")
    parser.add_argument("--job-name", default="nicholas", help="Label appended to the timestamped Slurm job and run directory name.")
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="02:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="8g")
    parser.add_argument("--conda-env", default="jaxsedfit")
    parser.add_argument(
        "--backend",
        choices=("jaxsedfit", "jaxsed", "grahspj", "grahsp"),
        default="grahspj",
        help="Fit backend: grahspj/jaxsedfit use hpc/run_manifest_fit.py; grahsp uses hpc/run_grahsp_manifest_fit.py.",
    )
    parser.add_argument("--grahsp-runner", type=Path, default=Path("hpc/run_grahsp_manifest_fit.py"))
    parser.add_argument("--grahsp-sampler-script", type=Path, default=DEFAULT_GRAHSP_SAMPLER_SCRIPT)
    parser.add_argument("--grahsp-cigale-root", type=Path, default=DEFAULT_GRAHSP_CIGALE_ROOT)
    parser.add_argument(
        "--grahsp-mass-max",
        type=float,
        default=DEFAULT_GRAHSP_MASS_MAX,
        help="External GRAHSP mass-max written to pcigale.ini; default matches notebook 13.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write task files and print sbatch plans without submitting.")
    args = parser.parse_args(argv)
    args.backend = _normalize_backend(args.backend)

    if args.max_array_tasks < 1:
        raise RuntimeError("--max-array-tasks must be at least 1.")
    if args.cpus_per_task < 1:
        raise RuntimeError("--cpus-per-task must be at least 1.")

    project_root = Path(__file__).resolve().parents[1]
    manifest = _resolve_from_root(project_root, args.manifest)
    output_base_dir = _resolve_from_root(project_root, args.output_dir)
    if args.run_dir is not None:
        output_dir = _resolve_from_root(project_root, args.run_dir)
        run_name = output_dir.name
    else:
        run_name = _run_name(args.job_name)
        output_dir = output_base_dir / run_name
    dsps_ssp_fn = _resolve_from_root(project_root, args.dsps_ssp_fn)
    grahsp_runner = _resolve_from_root(project_root, args.grahsp_runner)
    grahsp_sampler_script = _resolve_first_existing(
        project_root,
        [
            args.grahsp_sampler_script,
            DEFAULT_GRAHSP_SAMPLER_SCRIPT,
            Path("../GRAHSP-run/dualsampler.py"),
            Path("../sampler/dualsampler.py"),
        ],
        kind="file",
    )
    grahsp_cigale_root = _resolve_first_existing(
        project_root,
        [
            args.grahsp_cigale_root,
            DEFAULT_GRAHSP_CIGALE_ROOT,
            Path("../cigale"),
        ],
        kind="dir",
    )

    if not manifest.is_file():
        raise RuntimeError(f"Manifest not found: {manifest}")
    if args.backend == "grahspj" and not dsps_ssp_fn.is_file():
        raise RuntimeError(f"DSPS SSP file not found: {dsps_ssp_fn}")
    if args.backend == "grahsp":
        if not grahsp_runner.is_file():
            raise RuntimeError(f"GRAHSP runner not found: {grahsp_runner}")
        if not grahsp_sampler_script.is_file():
            raise RuntimeError(f"GRAHSP sampler script not found: {grahsp_sampler_script}")
        if not grahsp_cigale_root.is_dir():
            raise RuntimeError(f"GRAHSP CIGALE root not found: {grahsp_cigale_root}")

    tasks = _load_tasks(manifest)
    if not tasks:
        raise RuntimeError(f"Manifest contains no rows: {manifest}")
    total_task_count = len(tasks)

    task_dir = output_dir / "slurm_tasks"
    for subdir in (
        "logs",
        "results",
        "failures",
        "sed_pdfs",
        "sed_lum_pdfs",
        "corner_pdfs",
        "trace_pdfs",
        "posteriors_pdfs",
        "derived_pdfs",
        "sed_csvs",
        "photometry_csvs",
        "slurm_tasks",
    ):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    if args.only_missing:
        tasks, skipped_count = _filter_missing_tasks(tasks, output_dir, rerun_failures=args.rerun_failures)
    else:
        skipped_count = 0

    if not tasks:
        print(f"No tasks to submit. Manifest rows: {total_task_count}; skipped existing: {skipped_count}.")
        return 0

    chunks = _chunks(tasks, args.max_array_tasks)
    print(f"Project root: {project_root}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest rows: {total_task_count}")
    if args.only_missing:
        print(f"Selected missing rows: {len(tasks)}")
        print(f"Skipped existing rows: {skipped_count}")
        print(f"Rerun failures: {args.rerun_failures}")
    print(f"Backend: {args.backend}")
    print(f"Max array tasks: {args.max_array_tasks}")
    print(f"Chunks: {len(chunks)}")

    for chunk_number, (start, end, chunk_tasks) in enumerate(chunks, start=1):
        task_file = task_dir / f"tasks_{start:05d}_{end:05d}.tsv"
        _write_task_file(task_file, chunk_tasks)

        job_name = f"{run_name}-{start:05d}-{end:05d}"
        array = _array_spec(len(chunk_tasks))
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
            expected_count=total_task_count,
            conda_env=args.conda_env,
            backend=args.backend,
            grahsp_runner=grahsp_runner,
            grahsp_sampler_script=grahsp_sampler_script,
            grahsp_cigale_root=grahsp_cigale_root,
            grahsp_mass_max=args.grahsp_mass_max,
        )

        print(
            f"[{chunk_number}/{len(chunks)}] rows {start}-{end}: "
            f"{len(chunk_tasks)} tasks, array={array}, task_file={task_file}"
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
