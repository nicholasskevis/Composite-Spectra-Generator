#!/usr/bin/env xonsh
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _root_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


# -----------------------------------------------------------------------------
# Common fit settings
# -----------------------------------------------------------------------------

MANIFEST = _root_path("fit_manifest.csv")
DSPS_SSP_FN = _root_path("tempdata.h5")
OBJECT_ID = "013549.53+241149.7_243632_0.0001"
EXPECTED_COUNT = 13558

OUTPUT_ROOT = _root_path("hpc_outputs", "loglbol_mass_retrieval")
OUTPUT_LABEL = "manual_single_013549"
ALL_OBJECTS_JOB_NAME = "chimera_jaxsedfit"
ALL_OBJECTS_BACKEND = "grahspj"
MAX_ARRAY_TASKS = 4_000
SLURM_PARTITION = "day_amd"
SLURM_TIME = "02:00:00"
SLURM_CPUS_PER_TASK = 1
SLURM_MEM_PER_CPU = "8g"
SLURM_CONDA_ENV = "nicholas"

OPTAX_STEPS = 300
OPTAX_LR = "1.0e-2"
TARGET_ACCEPT_PROB = 0.85


# -----------------------------------------------------------------------------
# NUTS settings, used when SAMPLER is "optax+nuts"
# -----------------------------------------------------------------------------

NUTS_WARMUP = 2000
NUTS_SAMPLES = 1000
NUTS_CHAINS = 8


# -----------------------------------------------------------------------------
# Nested sampler settings, used when SAMPLER is "ns"
# -----------------------------------------------------------------------------

NS_LIVE_POINTS = 1000
NS_MAX_SAMPLES = None
NS_DLOGZ = 0.1
NS_RESAMPLES = 2000

# Advanced JAXNS DefaultNestedSampler tuning. Leave as False/None to use the
# library defaults.
NS_DIFFICULT_MODEL = False
NS_PARAMETER_ESTIMATION = False
NS_NUM_PARALLEL_WORKERS = None
NS_INIT_EFFICIENCY_THRESHOLD = None
NS_MAX_LIKELIHOOD_EVALS = None
NS_EFFICIENCY_THRESHOLD = None


def _sampler_output_dir(sampler: str) -> Path:
    safe_sampler = sampler.replace("+", "_")
    return OUTPUT_ROOT / f"{OUTPUT_LABEL}_{safe_sampler}"


def _build_single_object_command(sampler: str, output_dir: Path, dry_run: bool, ns_resamples: int) -> list[str]:
    cmd = [
        "python",
        str(_root_path("hpc", "run_manifest_fit.py")),
        "--manifest",
        str(MANIFEST),
        "--progress-bar",
        "--output-dir",
        str(output_dir),
        "--dsps-ssp-fn",
        str(DSPS_SSP_FN),
        "--object-id",
        OBJECT_ID,
        "--expected-count",
        str(EXPECTED_COUNT),
        "--sampler",
        sampler,
        "--optax-steps",
        str(OPTAX_STEPS),
        "--optax-lr",
        str(OPTAX_LR),
        "--target-accept-prob",
        str(TARGET_ACCEPT_PROB),
    ]

    if sampler == "optax+nuts":
        cmd.extend(
            [
                "--nuts-warmup",
                str(NUTS_WARMUP),
                "--nuts-samples",
                str(NUTS_SAMPLES),
                "--nuts-chains",
                str(NUTS_CHAINS),
            ]
        )
    elif sampler == "ns":
        cmd.extend(
            [
                "--ns-live-points",
                str(NS_LIVE_POINTS),
                "--ns-dlogz",
                str(NS_DLOGZ),
                "--ns-resamples",
                str(ns_resamples),
            ]
        )
        if NS_MAX_SAMPLES is not None:
            cmd.extend(["--ns-max-samples", str(NS_MAX_SAMPLES)])
        if NS_DIFFICULT_MODEL:
            cmd.append("--ns-difficult-model")
        if NS_PARAMETER_ESTIMATION:
            cmd.append("--ns-parameter-estimation")
        if NS_NUM_PARALLEL_WORKERS is not None:
            cmd.extend(["--ns-num-parallel-workers", str(NS_NUM_PARALLEL_WORKERS)])
        if NS_INIT_EFFICIENCY_THRESHOLD is not None:
            cmd.extend(["--ns-init-efficiency-threshold", str(NS_INIT_EFFICIENCY_THRESHOLD)])
        if NS_MAX_LIKELIHOOD_EVALS is not None:
            cmd.extend(["--ns-max-likelihood-evals", str(NS_MAX_LIKELIHOOD_EVALS)])
        if NS_EFFICIENCY_THRESHOLD is not None:
            cmd.extend(["--ns-efficiency-threshold", str(NS_EFFICIENCY_THRESHOLD)])
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    if dry_run:
        cmd.append("--dry-run")

    return cmd


def _build_all_objects_command(args: argparse.Namespace) -> list[str]:
    output_root = args.output_dir if args.output_dir is not None else OUTPUT_ROOT
    cmd = [
        "python",
        str(_root_path("hpc", "submit_loglbol_slurm_chunks.py")),
        "--manifest",
        str(MANIFEST),
        "--output-dir",
        str(output_root),
        "--dsps-ssp-fn",
        str(DSPS_SSP_FN),
        "--backend",
        args.backend,
        "--job-name",
        args.job_name,
        "--max-array-tasks",
        str(args.max_array_tasks),
        "--partition",
        args.partition,
        "--time",
        args.time_limit,
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem-per-cpu",
        args.mem_per_cpu,
        "--conda-env",
        args.conda_env,
    ]
    if args.run_dir is not None:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if args.only_missing:
        cmd.append("--only-missing")
    if args.rerun_failures:
        cmd.append("--rerun-failures")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


parser = argparse.ArgumentParser(description="Run one configured fit, or submit all manifest rows as Slurm chunks.")
parser.add_argument("--all-objects", action="store_true", help="Submit every manifest row as chunked Slurm arrays.")
parser.add_argument("--sampler", choices=("optax+nuts", "ns"), default="optax+nuts")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Single-object output directory, or all-object output base directory. Defaults to the configured hpc_outputs path.",
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--ns-resamples", type=int, default=NS_RESAMPLES)
parser.add_argument("--backend", choices=("jaxsedfit", "jaxsed", "grahspj", "grahsp"), default=ALL_OBJECTS_BACKEND)
parser.add_argument("--job-name", default=ALL_OBJECTS_JOB_NAME)
parser.add_argument("--max-array-tasks", type=int, default=MAX_ARRAY_TASKS)
parser.add_argument("--run-dir", type=Path, default=None, help="Exact existing or new all-object run directory.")
parser.add_argument("--only-missing", action="store_true", help="Submit only rows without an existing result/failure JSON in --run-dir.")
parser.add_argument("--rerun-failures", action="store_true", help="With --only-missing, include rows that have failure JSONs.")
parser.add_argument("--partition", default=SLURM_PARTITION)
parser.add_argument("--time", default=SLURM_TIME, dest="time_limit")
parser.add_argument("--cpus-per-task", type=int, default=SLURM_CPUS_PER_TASK)
parser.add_argument("--mem-per-cpu", default=SLURM_MEM_PER_CPU)
parser.add_argument("--conda-env", default=SLURM_CONDA_ENV)
args = parser.parse_args()

if args.all_objects:
    cmd = _build_all_objects_command(args)
else:
    output_dir = args.output_dir if args.output_dir is not None else _sampler_output_dir(args.sampler)
    cmd = _build_single_object_command(args.sampler, output_dir, args.dry_run, args.ns_resamples)

print("Running:", flush=True)
print(" ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
