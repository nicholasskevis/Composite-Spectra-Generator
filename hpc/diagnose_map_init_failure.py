from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import jax
import numpy as np
from numpyro.handlers import seed, substitute, trace

from run_manifest_fit import (
    _atomic_write_json,
    _json_sanitize,
    _load_backend,
    _load_manifest,
    _normalize_backend,
    _patch_backend_config_compat,
    _row_from_manifest,
    _set_if_present,
)


def _is_finite_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in "biufc":
            return True
        return bool(np.all(np.isfinite(arr)))
    except Exception:
        return False


def _array_summary(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        return {"repr": repr(value), "array_error": str(exc)}
    out: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.dtype.kind in "biufc" and arr.size:
        finite = np.isfinite(arr)
        out.update(
            {
                "all_finite": bool(np.all(finite)),
                "n_nonfinite": int(np.size(arr) - np.count_nonzero(finite)),
            }
        )
        if np.any(finite):
            finite_values = arr[finite]
            out.update(
                {
                    "min": float(np.min(finite_values)),
                    "median": float(np.median(finite_values)),
                    "max": float(np.max(finite_values)),
                }
            )
    elif arr.size == 1:
        out["value"] = arr.item()
    return out


def _log_prob_summary(site: dict[str, Any]) -> dict[str, Any]:
    fn = site.get("fn")
    value = site.get("value")
    if fn is None:
        return {"available": False}
    try:
        log_prob = fn.log_prob(value)
        arr = np.asarray(log_prob)
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    finite = np.isfinite(arr)
    out: dict[str, Any] = {
        "available": True,
        "shape": list(arr.shape),
        "all_finite": bool(np.all(finite)),
        "n_nonfinite": int(np.size(arr) - np.count_nonzero(finite)),
    }
    if np.any(finite):
        finite_values = arr[finite]
        total = np.sum(finite_values)
        out.update(
            {
                "sum_finite": float(total),
                "min": float(np.min(finite_values)),
                "median": float(np.median(finite_values)),
                "max": float(np.max(finite_values)),
            }
        )
    return out


def _select_row(manifest: Path, object_id: str) -> dict[str, str]:
    matches = [row for row in _load_manifest(manifest) if str(row["object_id"]) == str(object_id)]
    if len(matches) != 1:
        raise RuntimeError(f"object_id={object_id!r} matched {len(matches)} manifest rows.")
    return matches[0]


def _configure(args: argparse.Namespace):
    filter_names, build_chimera_fit_config, fitter_cls = _load_backend(args.backend)
    raw = _select_row(args.manifest, args.object_id)
    row = _row_from_manifest(raw, filter_names)
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(args.dsps_ssp_fn))
    _patch_backend_config_compat(cfg)
    cfg.inference.seed = int(args.seed_base + row["fit_index"])
    cfg.inference.method = "optax+nuts"
    _set_if_present(cfg.inference, "map_steps", int(args.optax_steps))
    _set_if_present(cfg.inference, "learning_rate", float(args.optax_lr))
    _set_if_present(cfg.inference, "num_warmup", int(args.nuts_warmup))
    _set_if_present(cfg.inference, "num_samples", int(args.nuts_samples))
    _set_if_present(cfg.inference, "num_chains", int(args.nuts_chains))
    _set_if_present(cfg.inference, "target_accept_prob", float(args.target_accept_prob))
    _set_if_present(cfg.inference, "dense_mass", args.dense_mass)
    _set_if_present(cfg.inference, "max_tree_depth", args.max_tree_depth)
    _set_if_present(cfg.inference, "use_map_init", True)
    return row, cfg, fitter_cls


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    row, cfg, fitter_cls = _configure(args)
    fitter = fitter_cls(cfg)
    payload: dict[str, Any] = {
        "object_id": str(row["id"]),
        "fit_index": int(row["fit_index"]),
        "COSMOS_ID0": str(row["COSMOS_ID0"]),
        "redshift": float(row["redshift"]),
        "backend": _normalize_backend(args.backend),
        "settings": {
            "optax_steps": int(args.optax_steps),
            "optax_lr": float(args.optax_lr),
            "nuts_warmup": int(args.nuts_warmup),
            "nuts_samples": int(args.nuts_samples),
            "nuts_chains": int(args.nuts_chains),
            "target_accept_prob": float(args.target_accept_prob),
            "dense_mass": args.dense_mass,
            "max_tree_depth": args.max_tree_depth,
            "seed": int(cfg.inference.seed),
        },
    }

    try:
        fitter.fit_map(steps=args.optax_steps, learning_rate=args.optax_lr, progress_bar=args.progress_bar)
    except Exception as exc:
        payload.update(
            {
                "status": "map_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return payload

    map_values = {name: np.asarray(value) for name, value in fitter.map_result["median"].items()}
    payload["map_status"] = "success"
    payload["map_loss_initial"] = (
        float(np.asarray(fitter.map_result["losses"])[0])
        if np.asarray(fitter.map_result["losses"]).size
        else math.nan
    )
    payload["map_loss_final"] = (
        float(np.asarray(fitter.map_result["losses"])[-1])
        if np.asarray(fitter.map_result["losses"]).size
        else math.nan
    )
    payload["map_nonfinite_values"] = {
        name: _array_summary(value)
        for name, value in map_values.items()
        if not _is_finite_array(value)
    }

    model = substitute(fitter._model, data=map_values)
    try:
        tr = trace(seed(model, jax.random.PRNGKey(int(cfg.inference.seed) + 1))).get_trace()
    except Exception as exc:
        payload.update(
            {
                "status": "trace_failed_at_map",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "map_value_summaries": {name: _array_summary(value) for name, value in map_values.items()},
            }
        )
        return payload

    sample_sites: dict[str, Any] = {}
    deterministic_nonfinite: dict[str, Any] = {}
    suspicious_sites: dict[str, Any] = {}
    total_log_prob = 0.0
    total_log_prob_finite = True

    for name, site in tr.items():
        site_type = site.get("type")
        if site_type == "sample":
            lp = _log_prob_summary(site)
            value_summary = _array_summary(site.get("value"))
            entry = {
                "fn": site.get("fn").__class__.__name__ if site.get("fn") is not None else None,
                "is_observed": bool(site.get("is_observed", False)),
                "value": value_summary,
                "log_prob": lp,
            }
            sample_sites[name] = entry
            if lp.get("available") and lp.get("all_finite"):
                total_log_prob += float(lp.get("sum_finite", 0.0))
            else:
                total_log_prob_finite = False
            if (not value_summary.get("all_finite", True)) or (not lp.get("all_finite", True)):
                suspicious_sites[name] = entry
        elif site_type == "deterministic" and not _is_finite_array(site.get("value")):
            deterministic_nonfinite[name] = _array_summary(site.get("value"))

    payload.update(
        {
            "status": "diagnosed",
            "total_sample_log_prob_finite": total_log_prob_finite,
            "total_sample_log_prob_sum_finite_terms": total_log_prob,
            "n_sample_sites": len(sample_sites),
            "n_suspicious_sample_sites": len(suspicious_sites),
            "n_nonfinite_deterministic_sites": len(deterministic_nonfinite),
            "suspicious_sample_sites": suspicious_sites,
            "nonfinite_deterministic_sites": deterministic_nonfinite,
            "sample_sites": sample_sites,
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a MAP-initialized NUTS failure for one manifest object.")
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--backend", choices=("jaxsedfit", "jaxsed", "grahspj"), default="grahspj")
    parser.add_argument("--seed-base", type=int, default=20231011)
    parser.add_argument("--optax-steps", type=int, default=500)
    parser.add_argument("--optax-lr", type=float, default=0.003)
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--target-accept-prob", type=float, default=0.95)
    parser.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    args.manifest = args.manifest.expanduser().resolve()
    args.dsps_ssp_fn = args.dsps_ssp_fn.expanduser().resolve()
    payload = diagnose(args)
    text = json.dumps(_json_sanitize(payload), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        _atomic_write_json(args.output.expanduser().resolve(), payload)
    return 0 if payload.get("status") == "diagnosed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
