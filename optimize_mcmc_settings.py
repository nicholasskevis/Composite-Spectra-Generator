#!/usr/bin/env python3
"""Grid-search JAXSEDFit NUTS settings against a Chimera mass truth value."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_JAXSEDFIT_ROOT = WORKSPACE / "grahspj_latest"
DEFAULT_CHIMERA_ID = "161651.82+385324.4_507867_0.0003"


def _csv_values(text: str, cast: Any) -> list[Any]:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _bool_value(text: str) -> bool:
    value = text.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _find_dsps_file(jaxsedfit_root: Path, explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else [
        jaxsedfit_root / "tempdata.h5",
        jaxsedfit_root.parent / "jaxqsofit" / "tempdata.h5",
    ]
    match = next((path.expanduser().resolve() for path in candidates if path and path.expanduser().is_file()), None)
    if match is None:
        raise FileNotFoundError("DSPS SSP file not found; pass --dsps-ssp-fn explicitly")
    return match


def _select_row(rows: Iterable[dict[str, Any]], lookup: str, object_id: str) -> dict[str, Any]:
    key = "ID_COSMOS" if lookup == "COSMOS_ID" else "id"
    row = next((candidate for candidate in rows if str(candidate[key]) == str(object_id)), None)
    if row is None:
        raise ValueError(f"no Chimera row found with {key}={object_id!r}")
    return row


def _settings_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = ("num_warmup", "num_samples", "target_accept_prob", "dense_mass", "max_tree_depth")
    values = (args.warmup, args.samples, args.target_accept, args.dense_mass, args.tree_depth)
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def _run_one(row: dict[str, Any], dsps_ssp_fn: Path, settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from jaxsedfit.benchmark import build_chimera_fit_config
    from jaxsedfit.core import JAXSEDFit

    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(dsps_ssp_fn))
    cfg.inference.method = "optax+nuts"
    cfg.inference.map_steps = args.map_steps
    cfg.inference.learning_rate = args.learning_rate
    cfg.inference.num_chains = args.chains
    cfg.inference.seed = args.seed
    for name, value in settings.items():
        setattr(cfg.inference, name, value)
    cfg.likelihood.use_local_line_photometry = True
    cfg.output.plot_fig = False
    cfg.output.save_fig = False
    cfg.output.save_result = False

    fitter = JAXSEDFit(cfg)
    fitter.fit(progress_bar=False)
    samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise RuntimeError("fit returned no finite log_stellar_mass samples")
    p16, median, p84 = np.percentile(samples, [16.0, 50.0, 84.0])
    truth = float(row["log_stellar_mass_truth"])
    residual = float(median - truth)
    return {
        "settings": {**settings, "map_steps": args.map_steps, "learning_rate": args.learning_rate,
                     "num_chains": args.chains, "seed": args.seed},
        "residual_dex": residual,
        "absolute_residual_dex": abs(residual),
        "recovered_log_stellar_mass": float(median),
        "posterior_16_84": [float(p16), float(p84)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", choices=("CHIMERA_ID", "COSMOS_ID"), default="CHIMERA_ID")
    parser.add_argument("--object-id", default=DEFAULT_CHIMERA_ID)
    parser.add_argument("--jaxsedfit-root", type=Path, default=DEFAULT_JAXSEDFIT_ROOT)
    parser.add_argument("--dsps-ssp-fn", type=Path)
    parser.add_argument("--warmup", type=lambda x: _csv_values(x, int), default=[500, 1000])
    parser.add_argument("--samples", type=lambda x: _csv_values(x, int), default=[500, 1000])
    parser.add_argument("--target-accept", type=lambda x: _csv_values(x, float), default=[0.8, 0.9])
    parser.add_argument("--dense-mass", type=lambda x: _csv_values(x, _bool_value), default=[False, True])
    parser.add_argument("--tree-depth", type=lambda x: _csv_values(x, int), default=[8])
    parser.add_argument("--map-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.jaxsedfit_root.expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    from jaxsedfit.benchmark import load_chimera_benchmark_dataset

    row = _select_row(load_chimera_benchmark_dataset(root).rows, args.lookup, args.object_id)
    dsps_ssp_fn = _find_dsps_file(root, args.dsps_ssp_fn)
    grid = _settings_grid(args)
    results = []
    for index, settings in enumerate(grid, start=1):
        print(f"running MCMC configuration {index}/{len(grid)}: {settings}", file=sys.stderr, flush=True)
        try:
            results.append(_run_one(row, dsps_ssp_fn, settings, args))
        except Exception as exc:
            print(f"configuration failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if not results:
        raise RuntimeError("every MCMC configuration failed")
    best = min(results, key=lambda result: result["absolute_residual_dex"])
    output = {"object_id": str(row["id"]), "truth_log_stellar_mass": float(row["log_stellar_mass_truth"]), **best}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
