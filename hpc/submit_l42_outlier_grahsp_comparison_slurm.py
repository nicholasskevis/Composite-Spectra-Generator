#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/home/ns2385/project_pi_pn38/ns2385/grahsp_vs_grahspj_l42_outliers")


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


def _load_l42_outliers(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("luminosity_bin") == "L < 42"]
    rows.sort(key=lambda row: float(row.get("abs_residual_log_ratio") or 0.0), reverse=True)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    records = []
    for rank, row in enumerate(rows, start=1):
        records.append(
            {
                "rank": rank,
                "object_id": row["object_id"],
                "COSMOS_ID0": row.get("COSMOS_ID0", ""),
                "luminosity_bin": row.get("luminosity_bin", ""),
                "log_stellar_mass_truth": row.get("log_stellar_mass_truth", ""),
                "previous_recovered_logm": row.get("recovered_logm", ""),
                "previous_residual_log_ratio": row.get("residual_log_ratio", ""),
                "previous_abs_residual_log_ratio": row.get("abs_residual_log_ratio", ""),
            }
        )
    return records


def _write_task_file(path: Path, records: list[dict[str, Any]]) -> None:
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
    n_wave: int,
    sampler: str,
    optax_steps: int,
    optax_lr: float,
    nuts_warmup: int,
    nuts_samples: int,
    nuts_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    grahsp_mass_max: float,
    grahsp_live_points: int,
    grahsp_posterior_samples: int,
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
CONDA_ENV={shlex.quote(conda_env)}

mkdir -p "${{OUTPUT_DIR}}/logs"
cd "${{PROJECT_ROOT}}"
export TASK_FILE

module reset
module load miniconda
conda activate "${{CONDA_ENV}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-${{OUTPUT_DIR}}/matplotlib_cache}}"
mkdir -p "${{MPLCONFIGDIR}}"

OBJECT_ID="$(python -c 'import json, os, pathlib; lines=pathlib.Path(os.environ["TASK_FILE"]).read_text().splitlines(); print(json.loads(lines[int(os.environ["SLURM_ARRAY_TASK_ID"])])["object_id"])')"

python hpc/run_grahsp_jaxsedfit_comparison.py \\
  --manifest "${{MANIFEST}}" \\
  --object-id "${{OBJECT_ID}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --dsps-ssp-fn "${{DSPS_SSP_FN}}" \\
  --jaxsedfit-root "${{JAXSEDFIT_ROOT}}" \\
  --n-wave {n_wave} \\
  --sampler {shlex.quote(sampler)} \\
  --optax-steps {optax_steps} \\
  --optax-lr {optax_lr:g} \\
  --nuts-warmup {nuts_warmup} \\
  --nuts-samples {nuts_samples} \\
  --nuts-chains {nuts_chains} \\
  --target-accept-prob {target_accept_prob:g} \\
  --no-dense-mass \\
  --max-tree-depth {max_tree_depth} \\
  --grahsp-mass-max {grahsp_mass_max:g} \\
  --grahsp-live-points {grahsp_live_points} \\
  --grahsp-posterior-samples {grahsp_posterior_samples}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit L<42 top-outlier GRAHSP vs GRAHSPJ/JAXSEDFit comparisons.")
    parser.add_argument("--outliers-csv", type=Path, default=Path("top100_mass_retrieval_outliers_per_logLbol_bin.csv"))
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of L<42 outliers to run. Use 0 for all selected rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, help="Exact output run directory. Overrides timestamped directory.")
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--jaxsedfit-root", type=Path, default=Path("/home/ns2385/jaxsedfit/jaxsedfit"))
    parser.add_argument("--job-name", default="l42_grahsp_compare")
    parser.add_argument("--max-array-tasks", type=int, default=1000)
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="04:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="12g")
    parser.add_argument("--conda-env", default="jaxsedfit")
    parser.add_argument("--n-wave", type=int, default=1024)
    parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax+nuts")
    parser.add_argument("--optax-steps", type=int, default=500)
    parser.add_argument("--optax-lr", type=float, default=0.003)
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--target-accept-prob", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--grahsp-mass-max", type=float, default=13.0)
    parser.add_argument("--grahsp-live-points", type=int, default=800)
    parser.add_argument("--grahsp-posterior-samples", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = _resolve(args.output_dir)
    output_dir = _resolve(args.run_dir) if args.run_dir else output_root / _run_name(args.job_name)
    manifest = _resolve(args.manifest)
    dsps_ssp_fn = _resolve(args.dsps_ssp_fn)
    outliers_csv = _resolve(args.outliers_csv)
    jaxsedfit_root = args.jaxsedfit_root.expanduser().resolve()

    limit = None if args.limit == 0 else args.limit
    task_records = _load_l42_outliers(outliers_csv, limit=limit)
    if not task_records:
        raise RuntimeError(f"No L < 42 outliers selected from {outliers_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task_file = output_dir / "slurm_tasks" / "l42_outlier_comparison_tasks.jsonl"
    _write_task_file(task_file, task_records)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Task file: {task_file}")
    print(f"L < 42 objects: {len(task_records)}")
    print(f"JAXSEDFit/GRAHSPJ root: {jaxsedfit_root}")
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
            n_wave=args.n_wave,
            sampler=args.sampler,
            optax_steps=args.optax_steps,
            optax_lr=args.optax_lr,
            nuts_warmup=args.nuts_warmup,
            nuts_samples=args.nuts_samples,
            nuts_chains=args.nuts_chains,
            target_accept_prob=args.target_accept_prob,
            max_tree_depth=args.max_tree_depth,
            grahsp_mass_max=args.grahsp_mass_max,
            grahsp_live_points=args.grahsp_live_points,
            grahsp_posterior_samples=args.grahsp_posterior_samples,
        )
        script_path = output_dir / "slurm_scripts" / f"submit_{start:06d}_{end:06d}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        print(f"[{chunk_number}] tasks {start}-{end}: {script_path}")
        print(_submit(script_path, args.dry_run))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
