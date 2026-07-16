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

OUTPUT_ROOT = Path("/home/ns2385/project_pi_pn38/ns2385/grahspj_loglbol_mass_retrieval")
OUTPUT_LABEL = "manual_single_013549"
ALL_OBJECTS_JOB_NAME = "chimera_jaxsedfit"
ALL_OBJECTS_BACKEND = "grahspj"
MAX_ARRAY_TASKS = 4_000
SLURM_PARTITION = "day"
SLURM_TIME = "02:00:00"
SLURM_CPUS_PER_TASK = 1
SLURM_MEM_PER_CPU = "8g"
SLURM_CONDA_ENV = "jaxsedfit"

OPTAX_STEPS = 500
OPTAX_LR = "5.0e-3"
TARGET_ACCEPT_PROB = 0.90
NUTS_DENSE_MASS = False
NUTS_MAX_TREE_DEPTH = 8
COMPARE_OBJECT_ID = "022754.38-073455.0_869049_0.0001"
COMPARE_N_WAVE = 1024
COMPARE_GRAHSP_MASS_MAX = 13.0
SPECTRA_MANIFEST = _root_path(
    "notebook_outputs",
    "all_chimera_notebook6_spectra",
    "chimera_notebook6_spectra_manifest.csv",
)
JOINT_OUTPUT_ROOT = _root_path("hpc_outputs", "joint_photometry_spectroscopy")
JOINT_JOB_NAME = "chimera_joint_spectro"
JOINT_MAX_ARRAY_TASKS = 1000
JOINT_SLURM_PARTITION = "day"
JOINT_SLURM_TIME = "04:00:00"
JOINT_SLURM_CPUS_PER_TASK = 1
JOINT_SLURM_MEM_PER_CPU = "12g"
JOINT_SLURM_CONDA_ENV = "jaxsedfit"
JOINT_SAMPLER = "optax"
JOINT_N_WAVE = 512
JOINT_OPTAX_STEPS = 600
JOINT_OPTAX_LR = "5.0e-3"
JOINT_NUTS_WARMUP = 250
JOINT_NUTS_SAMPLES = 250
JOINT_NUTS_CHAINS = 1
JOINT_MAX_TREE_DEPTH = 8
TOP_OUTLIER_OUTPUT_ROOT = _root_path("hpc_outputs", "top_outlier_mcmc_setting_optimization")
TOP_OUTLIER_JOB_NAME = "top_outlier_mcmc"
TOP_OUTLIER_LIMIT = 100
TOP_OUTLIER_MAX_ARRAY_TASKS = 4000
TOP_OUTLIER_WARMUP = "500,1000"
TOP_OUTLIER_SAMPLES = "300,500"
TOP_OUTLIER_TARGET_ACCEPT = "0.8,0.85,0.9,0.95"
TOP_OUTLIER_DENSE_MASS = "false,true"
TOP_OUTLIER_TREE_DEPTH = "6,8,10"
TOP_OUTLIER_MAP_STEPS = "300,500"
TOP_OUTLIER_LEARNING_RATE = "0.003,0.005"


# -----------------------------------------------------------------------------
# NUTS settings, used when SAMPLER is "optax+nuts"
# -----------------------------------------------------------------------------

NUTS_WARMUP = 1000
NUTS_SAMPLES = 500
NUTS_CHAINS = 1


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


def _build_single_object_command(
    sampler: str,
    output_dir: Path,
    dry_run: bool,
    ns_resamples: int,
    use_map_init: bool | None,
    disable_agn: bool,
) -> list[str]:
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
        "--sampler",
        sampler,
        "--optax-steps",
        str(OPTAX_STEPS),
        "--optax-lr",
        str(OPTAX_LR),
        "--target-accept-prob",
        str(TARGET_ACCEPT_PROB),
    ]
    if disable_agn:
        cmd.append("--disable-agn")

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
    if use_map_init is not None:
        cmd.append("--use-map-init" if use_map_init else "--no-use-map-init")

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
        "--sampler",
        args.sampler,
        "--optax-steps",
        str(args.optax_steps),
        "--optax-lr",
        str(args.optax_lr),
        "--nuts-warmup",
        str(args.nuts_warmup),
        "--nuts-samples",
        str(args.nuts_samples),
        "--nuts-chains",
        str(args.nuts_chains),
        "--target-accept-prob",
        str(args.target_accept_prob),
    ]
    if args.dense_mass is not None:
        cmd.append("--dense-mass" if args.dense_mass else "--no-dense-mass")
    if args.max_tree_depth is not None:
        cmd.extend(["--max-tree-depth", str(args.max_tree_depth)])
    if args.use_map_init is not None:
        cmd.append("--use-map-init" if args.use_map_init else "--no-use-map-init")
    if args.disable_agn:
        cmd.append("--disable-agn")
    if args.luminosity_bin is not None:
        cmd.extend(["--luminosity-bin", args.luminosity_bin])
    if args.run_dir is not None:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if args.only_missing:
        cmd.append("--only-missing")
    if args.rerun_failures:
        cmd.append("--rerun-failures")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_comparison_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        str(_root_path("hpc", "run_grahsp_jaxsedfit_comparison.py")),
        "--manifest",
        str(MANIFEST),
        "--object-id",
        args.object_id,
        "--output-dir",
        str(args.output_dir if args.output_dir is not None else OUTPUT_ROOT / "grahsp_vs_jaxsedfit_single"),
        "--dsps-ssp-fn",
        str(DSPS_SSP_FN),
        "--n-wave",
        str(args.n_wave),
        "--sampler",
        args.sampler,
        "--optax-steps",
        str(OPTAX_STEPS),
        "--optax-lr",
        str(OPTAX_LR),
        "--target-accept-prob",
        str(TARGET_ACCEPT_PROB),
        "--nuts-warmup",
        str(args.nuts_warmup),
        "--nuts-samples",
        str(args.nuts_samples),
        "--nuts-chains",
        str(args.nuts_chains),
        "--grahsp-mass-max",
        str(args.grahsp_mass_max),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.progress_bar:
        cmd.append("--progress-bar")
    if args.skip_jaxsedfit:
        cmd.append("--skip-jaxsedfit")
    if args.skip_grahsp:
        cmd.append("--skip-grahsp")
    if args.skip_jaxsedfit_plots:
        cmd.append("--skip-jaxsedfit-plots")
    return cmd


def _build_joint_spectra_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir if args.output_dir is not None else JOINT_OUTPUT_ROOT
    common = [
        "--manifest",
        str(MANIFEST),
        "--spectra-manifest",
        str(args.spectra_manifest),
        "--output-dir",
        str(output_dir),
        "--dsps-ssp-fn",
        str(DSPS_SSP_FN),
        "--sampler",
        args.joint_sampler,
        "--n-wave",
        str(args.n_wave),
        "--optax-steps",
        str(args.joint_optax_steps),
        "--optax-lr",
        str(args.joint_optax_lr),
        "--nuts-warmup",
        str(args.nuts_warmup),
        "--nuts-samples",
        str(args.nuts_samples),
        "--nuts-chains",
        str(args.nuts_chains),
        "--target-accept-prob",
        str(TARGET_ACCEPT_PROB),
        "--max-tree-depth",
        str(args.max_tree_depth),
    ]
    if args.all_objects:
        cmd = [
            "python",
            str(_root_path("hpc", "submit_joint_spectro_slurm_chunks.py")),
            *common,
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
    else:
        cmd = [
            "python",
            str(_root_path("hpc", "run_joint_spectro_manifest_fit.py")),
            *common,
            "--object-id",
            args.object_id,
        ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.progress_bar:
        cmd.append("--progress-bar")
    return cmd


def _build_top_outlier_optimization_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir if args.output_dir is not None else TOP_OUTLIER_OUTPUT_ROOT
    cmd = [
        "python",
        str(_root_path("hpc", "submit_top_outlier_mcmc_settings_slurm.py")),
        "--outliers-csv",
        str(args.outliers_csv),
        "--limit",
        str(args.limit),
        "--output-dir",
        str(output_dir),
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
        "--warmup",
        args.grid_warmup,
        "--samples",
        args.grid_samples,
        "--target-accept",
        args.grid_target_accept,
        "--dense-mass",
        args.grid_dense_mass,
        "--tree-depth",
        args.grid_tree_depth,
        "--map-steps",
        args.grid_map_steps,
        "--learning-rate",
        args.grid_learning_rate,
        "--chains",
        str(args.nuts_chains),
    ]
    if args.run_dir is not None:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


parser = argparse.ArgumentParser(description="Run one configured fit, or submit all manifest rows as Slurm chunks.")
parser.add_argument("--all-objects", action="store_true", help="Submit every manifest row as chunked Slurm arrays.")
parser.add_argument("--compare-backends", action="store_true", help="Run one object through both JAXSEDFit and external GRAHSP.")
parser.add_argument("--joint-spectra", action="store_true", help="Run JAXSEDFit jointly on Chimera photometry and notebook-6 spectra.")
parser.add_argument("--optimize-top-outliers", action="store_true", help="Submit the top-outlier MCMC settings grid as Slurm arrays.")
parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax+nuts")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Single-object output directory, or all-object output base directory. Defaults to the configured hpc_outputs path.",
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--progress-bar", action="store_true", help="Show fitter progress bars in single comparison mode.")
parser.add_argument("--object-id", default=COMPARE_OBJECT_ID, help="Object id for --compare-backends or --joint-spectra.")
parser.add_argument("--n-wave", type=int, default=COMPARE_N_WAVE, help="JAXSEDFit wavelength grid size for --compare-backends/--joint-spectra.")
parser.add_argument("--optax-steps", type=int, default=OPTAX_STEPS)
parser.add_argument("--optax-lr", type=float, default=float(OPTAX_LR))
parser.add_argument("--nuts-warmup", type=int, default=NUTS_WARMUP, help="NUTS warmup draws for --compare-backends.")
parser.add_argument("--nuts-samples", type=int, default=NUTS_SAMPLES, help="NUTS posterior draws for --compare-backends.")
parser.add_argument("--nuts-chains", type=int, default=NUTS_CHAINS, help="NUTS chains for --compare-backends.")
parser.add_argument("--target-accept-prob", type=float, default=TARGET_ACCEPT_PROB)
parser.add_argument("--disable-agn", action="store_true", help="Fit Chimera photometry with the JAXSEDFit AGN component switched off.")
parser.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=NUTS_DENSE_MASS)
parser.add_argument(
    "--use-map-init",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="For NUTS/optax+nuts, initialize NUTS from the MAP solution. Use --no-use-map-init for difficult failed objects.",
)
parser.add_argument("--grahsp-mass-max", type=float, default=COMPARE_GRAHSP_MASS_MAX, help="External GRAHSP pcigale.ini mass-max for --compare-backends.")
parser.add_argument("--skip-jaxsedfit", action="store_true", help="With --compare-backends, do not run JAXSEDFit.")
parser.add_argument("--skip-grahsp", action="store_true", help="With --compare-backends, do not run external GRAHSP.")
parser.add_argument("--skip-jaxsedfit-plots", action="store_true", help="With --compare-backends, skip JAXSEDFit sed/corner/trace PDFs.")
parser.add_argument("--ns-resamples", type=int, default=NS_RESAMPLES)
parser.add_argument("--backend", choices=("jaxsedfit", "jaxsed", "grahspj", "grahsp"), default=ALL_OBJECTS_BACKEND)
parser.add_argument("--job-name", default=ALL_OBJECTS_JOB_NAME)
parser.add_argument("--max-array-tasks", type=int, default=MAX_ARRAY_TASKS)
parser.add_argument("--run-dir", type=Path, default=None, help="Exact existing or new all-object run directory.")
parser.add_argument("--luminosity-bin", default=None, help="Submit only all-object rows matching this luminosity_bin value, e.g. 'L < 42'.")
parser.add_argument("--only-missing", action="store_true", help="Submit only rows without an existing result/failure JSON in --run-dir.")
parser.add_argument("--rerun-failures", action="store_true", help="With --only-missing, include rows that have failure JSONs.")
parser.add_argument("--partition", default=SLURM_PARTITION)
parser.add_argument("--time", default=SLURM_TIME, dest="time_limit")
parser.add_argument("--cpus-per-task", type=int, default=SLURM_CPUS_PER_TASK)
parser.add_argument("--mem-per-cpu", default=SLURM_MEM_PER_CPU)
parser.add_argument("--conda-env", default=SLURM_CONDA_ENV)
parser.add_argument("--spectra-manifest", type=Path, default=SPECTRA_MANIFEST)
parser.add_argument("--outliers-csv", type=Path, default=_root_path("top100_mass_retrieval_outliers_per_logLbol_bin.csv"))
parser.add_argument("--limit", type=int, default=TOP_OUTLIER_LIMIT)
parser.add_argument("--grid-warmup", default=TOP_OUTLIER_WARMUP)
parser.add_argument("--grid-samples", default=TOP_OUTLIER_SAMPLES)
parser.add_argument("--grid-target-accept", default=TOP_OUTLIER_TARGET_ACCEPT)
parser.add_argument("--grid-dense-mass", default=TOP_OUTLIER_DENSE_MASS)
parser.add_argument("--grid-tree-depth", default=TOP_OUTLIER_TREE_DEPTH)
parser.add_argument("--grid-map-steps", default=TOP_OUTLIER_MAP_STEPS)
parser.add_argument("--grid-learning-rate", default=TOP_OUTLIER_LEARNING_RATE)
parser.add_argument("--joint-sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default=JOINT_SAMPLER)
parser.add_argument("--joint-optax-steps", type=int, default=JOINT_OPTAX_STEPS)
parser.add_argument("--joint-optax-lr", type=float, default=JOINT_OPTAX_LR)
parser.add_argument("--max-tree-depth", type=int, default=NUTS_MAX_TREE_DEPTH)
args = parser.parse_args()

if args.joint_spectra:
    if args.backend == ALL_OBJECTS_BACKEND:
        args.backend = "jaxsedfit"
    if args.job_name == ALL_OBJECTS_JOB_NAME:
        args.job_name = JOINT_JOB_NAME
    if args.max_array_tasks == MAX_ARRAY_TASKS:
        args.max_array_tasks = JOINT_MAX_ARRAY_TASKS
    if args.partition == SLURM_PARTITION:
        args.partition = JOINT_SLURM_PARTITION
    if args.time_limit == SLURM_TIME:
        args.time_limit = JOINT_SLURM_TIME
    if args.cpus_per_task == SLURM_CPUS_PER_TASK:
        args.cpus_per_task = JOINT_SLURM_CPUS_PER_TASK
    if args.mem_per_cpu == SLURM_MEM_PER_CPU:
        args.mem_per_cpu = JOINT_SLURM_MEM_PER_CPU
    if args.conda_env == SLURM_CONDA_ENV:
        args.conda_env = JOINT_SLURM_CONDA_ENV
    if args.output_dir is None:
        args.output_dir = JOINT_OUTPUT_ROOT
    if args.n_wave == COMPARE_N_WAVE:
        args.n_wave = JOINT_N_WAVE
    if args.nuts_warmup == NUTS_WARMUP:
        args.nuts_warmup = JOINT_NUTS_WARMUP
    if args.nuts_samples == NUTS_SAMPLES:
        args.nuts_samples = JOINT_NUTS_SAMPLES
    if args.nuts_chains == NUTS_CHAINS:
        args.nuts_chains = JOINT_NUTS_CHAINS

if sum(bool(x) for x in (args.compare_backends, args.joint_spectra, args.optimize_top_outliers)) > 1:
    raise SystemExit("--compare-backends, --joint-spectra, and --optimize-top-outliers cannot be used together.")
if args.all_objects and args.compare_backends:
    raise SystemExit("--all-objects and --compare-backends cannot be used together.")

if args.optimize_top_outliers:
    if args.job_name == ALL_OBJECTS_JOB_NAME:
        args.job_name = TOP_OUTLIER_JOB_NAME
    if args.max_array_tasks == MAX_ARRAY_TASKS:
        args.max_array_tasks = TOP_OUTLIER_MAX_ARRAY_TASKS
    if args.output_dir is None:
        args.output_dir = TOP_OUTLIER_OUTPUT_ROOT

if args.joint_spectra:
    cmd = _build_joint_spectra_command(args)
elif args.optimize_top_outliers:
    cmd = _build_top_outlier_optimization_command(args)
elif args.compare_backends:
    cmd = _build_comparison_command(args)
elif args.all_objects:
    cmd = _build_all_objects_command(args)
else:
    output_dir = args.output_dir if args.output_dir is not None else _sampler_output_dir(args.sampler)
    cmd = _build_single_object_command(
        args.sampler,
        output_dir,
        args.dry_run,
        args.ns_resamples,
        args.use_map_init,
        args.disable_agn,
    )

print("Running:", flush=True)
print(" ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
