#!/usr/bin/env xonsh
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


# -----------------------------------------------------------------------------
# Common fit settings
# -----------------------------------------------------------------------------

MANIFEST = Path("fit_manifest.csv")
DSPS_SSP_FN = Path("tempdata.h5")
OBJECT_ID = "013549.53+241149.7_243632_0.0001"
EXPECTED_COUNT = 13558

OUTPUT_ROOT = Path("hpc_outputs/loglbol_mass_retrieval")
OUTPUT_LABEL = "manual_single_013549"

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


def _build_command(sampler: str, output_dir: Path, dry_run: bool, ns_resamples: int) -> list[str]:
    cmd = [
        "python",
        "run_manifest_fit.py",
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


parser = argparse.ArgumentParser(description="Run one manifest fit with either optax+nuts or nested sampling.")
parser.add_argument("--sampler", choices=("optax+nuts", "ns"), default="optax+nuts")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Optional explicit output directory. Defaults to a sampler-specific folder.",
)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--ns-resamples", type=int, default=NS_RESAMPLES)
args = parser.parse_args()

output_dir = args.output_dir if args.output_dir is not None else _sampler_output_dir(args.sampler)
cmd = _build_command(args.sampler, output_dir, args.dry_run, args.ns_resamples)

print("Running:", flush=True)
print(" ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
