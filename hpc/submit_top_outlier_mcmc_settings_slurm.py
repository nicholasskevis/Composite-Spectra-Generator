#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("hpc_outputs/top_outlier_mcmc_setting_optimization")
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _safe_label(value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    label = "_".join(part for part in label.split("_") if part)
    if not label:
        raise ValueError("job name must contain at least one letter or number")
    return label


def _run_name(job_name: str) -> str:
    now = datetime.now()
    return f"{now.strftime('%B').lower()}{now.day}_{now:%H%M}_{_safe_label(job_name)}"


def _chunks(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, total) - 1) for start in range(0, total, size)]


def _write_task_file(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _submit(script_path: Path, dry_run: bool) -> str:
    if dry_run:
        return f"DRY-RUN {script_path}"
    result = subprocess.run(
        ["sbatch", str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}")
    return result.stdout.strip()


def _batch_script(
    *,
    job_name: str,
    array: str,
    partition: str,
    time_limit: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    conda_env: str,
    output_dir: Path,
    task_file: Path,
    manifest: Path,
    dsps_ssp_fn: Path,
    jaxsedfit_root: Path,
    outliers_csv: Path,
    warmup: str,
    samples: str,
    target_accept: str,
    dense_mass: str,
    tree_depth: str,
    map_steps: str,
    learning_rate: str,
    chains: int,
    seed: int,
    rank_column: str,
    limit: int,
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

PROJECT_ROOT={shlex.quote(str(PROJECT_ROOT))}
OUTPUT_DIR={shlex.quote(str(output_dir))}
TASK_FILE={shlex.quote(str(task_file))}
MANIFEST={shlex.quote(str(manifest))}
DSPS_SSP_FN={shlex.quote(str(dsps_ssp_fn))}
JAXSEDFIT_ROOT={shlex.quote(str(jaxsedfit_root))}
OUTLIERS_CSV={shlex.quote(str(outliers_csv))}
CONDA_ENV={shlex.quote(conda_env)}

mkdir -p "${{OUTPUT_DIR}}/logs" "${{OUTPUT_DIR}}/trials"
cd "${{PROJECT_ROOT}}"

module reset
module load miniconda
conda activate "${{CONDA_ENV}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-${{OUTPUT_DIR}}/matplotlib_cache}}"
mkdir -p "${{MPLCONFIGDIR}}"

python optimize_mcmc_settings_top_outliers.py \\
  --task-file "${{TASK_FILE}}" \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --outliers-csv "${{OUTLIERS_CSV}}" \\
  --limit {limit} \\
  --rank-column {shlex.quote(rank_column)} \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --jaxsedfit-root "${{JAXSEDFIT_ROOT}}" \\
  --manifest "${{MANIFEST}}" \\
  --dsps-ssp-fn "${{DSPS_SSP_FN}}" \\
  --warmup {shlex.quote(warmup)} \\
  --samples {shlex.quote(samples)} \\
  --target-accept {shlex.quote(target_accept)} \\
  --dense-mass {shlex.quote(dense_mass)} \\
  --tree-depth {shlex.quote(tree_depth)} \\
  --map-steps {shlex.quote(map_steps)} \\
  --learning-rate {shlex.quote(learning_rate)} \\
  --chains {chains} \\
  --seed {seed}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit top-outlier MCMC setting optimization as Slurm arrays.")
    parser.add_argument("--outliers-csv", type=Path, default=Path("top100_mass_retrieval_outliers_per_logLbol_bin.csv"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--rank-column", default="abs_residual_log_ratio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, help="Exact output run directory. Overrides timestamped directory.")
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--jaxsedfit-root", type=Path, default=PROJECT_ROOT.parent / "grahspj_latest")
    parser.add_argument("--job-name", default="top_outlier_mcmc")
    parser.add_argument("--max-array-tasks", type=int, default=4000)
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="02:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="8g")
    parser.add_argument("--conda-env", default="jaxsedfit")
    parser.add_argument("--warmup", default="500,1000")
    parser.add_argument("--samples", default="300,500")
    parser.add_argument("--target-accept", default="0.8,0.85,0.9,0.95")
    parser.add_argument("--dense-mass", default="false,true")
    parser.add_argument("--tree-depth", default="6,8,10")
    parser.add_argument("--map-steps", default="300,500")
    parser.add_argument("--learning-rate", default="0.003,0.005")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = _resolve(args.output_dir)
    output_dir = _resolve(args.run_dir) if args.run_dir else output_root / _run_name(args.job_name)
    manifest = _resolve(args.manifest)
    dsps_ssp_fn = _resolve(args.dsps_ssp_fn)
    outliers_csv = _resolve(args.outliers_csv)
    jaxsedfit_root = _resolve(args.jaxsedfit_root)

    import optimize_mcmc_settings_top_outliers as optimizer

    optimizer_args = optimizer.build_parser().parse_args([
        "--outliers-csv", str(outliers_csv),
        "--limit", str(args.limit),
        "--rank-column", args.rank_column,
        "--warmup", args.warmup,
        "--samples", args.samples,
        "--target-accept", args.target_accept,
        "--dense-mass", args.dense_mass,
        "--tree-depth", args.tree_depth,
        "--map-steps", args.map_steps,
        "--learning-rate", args.learning_rate,
        "--chains", str(args.chains),
        "--seed", str(args.seed),
    ])
    task_records = optimizer.build_task_records(optimizer_args)
    if not task_records:
        raise RuntimeError("No optimization tasks selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_file = output_dir / "slurm_tasks" / "top_outlier_mcmc_tasks.jsonl"
    _write_task_file(task_file, task_records)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Task file: {task_file}")
    print(f"Objects: {len({record['object_id'] for record in task_records})}")
    print(f"Tasks: {len(task_records)}")
    print(f"Max array tasks: {args.max_array_tasks}")
    print(f"Chunks: {len(_chunks(len(task_records), args.max_array_tasks))}")

    for chunk_number, (start, end) in enumerate(_chunks(len(task_records), args.max_array_tasks), start=1):
        script = _batch_script(
            job_name=f"{args.job_name}_{chunk_number:02d}",
            array=f"{start}-{end}",
            partition=args.partition,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            conda_env=args.conda_env,
            output_dir=output_dir,
            task_file=task_file,
            manifest=manifest,
            dsps_ssp_fn=dsps_ssp_fn,
            jaxsedfit_root=jaxsedfit_root,
            outliers_csv=outliers_csv,
            warmup=args.warmup,
            samples=args.samples,
            target_accept=args.target_accept,
            dense_mass=args.dense_mass,
            tree_depth=args.tree_depth,
            map_steps=args.map_steps,
            learning_rate=args.learning_rate,
            chains=args.chains,
            seed=args.seed,
            rank_column=args.rank_column,
            limit=args.limit,
        )
        script_path = output_dir / "slurm_scripts" / f"submit_{start:06d}_{end:06d}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        print(f"[{chunk_number}] tasks {start}-{end}: {script_path}")
        print(_submit(script_path, args.dry_run))

    print("After jobs finish, summarize with:")
    print(f"python optimize_mcmc_settings_top_outliers.py --summarize-only --output-dir {shlex.quote(str(output_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
