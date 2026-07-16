from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ALIASES = {
    "jaxsed": "jaxsedfit",
    "jaxsedfit": "jaxsedfit",
    "grahspj": "jaxsedfit",
}

SFR_SAMPLE_KEYS = (
    "log_sfr",
    "log_SFR",
    "log10_sfr",
    "log10_SFR",
    "sfr",
    "SFR",
    "sfh_sfr",
    "sfh.sfr",
    "sfr100Myrs",
    "sfh.sfr100Myrs",
    "sfr_100myr",
    "sfr100",
)

OPTIONAL_MANIFEST_FLOAT_FIELDS = (
    "logL5100_QSO",
    "e_logL5100_QSO",
    "logL3000_QSO",
    "e_logL3000_QSO",
    "logL1350_QSO",
    "e_logL1350_QSO",
    "SFR_BEST_GAL",
    "SFR_MED_GAL",
    "SFR_MED_MIN68_GAL",
    "SFR_MED_MAX68_GAL",
    "SSFR_BEST_GAL",
    "SSFR_MED_GAL",
    "SSFR_MED_MIN68_GAL",
    "SSFR_MED_MAX68_GAL",
)


def _normalize_backend(value: str) -> str:
    try:
        return BACKEND_ALIASES[value.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKEND_ALIASES))
        raise ValueError(f"Unsupported backend {value!r}; choose one of: {choices}") from exc


def _load_backend(backend: str) -> tuple[list[str], Any, type]:
    backend = _normalize_backend(backend)
    benchmark = importlib.import_module(f"{backend}.benchmark")
    core = importlib.import_module(f"{backend}.core")
    fitter_cls = getattr(core, "GRAHSPJ", None) or getattr(core, "JAXSEDFit")
    return list(benchmark.CHIMERA_FILTER_NAMES), benchmark.build_chimera_fit_config, fitter_cls


def _patch_backend_config_compat(cfg: Any) -> None:
    filters = getattr(cfg, "filters", None)
    if filters is not None and not hasattr(filters, "speclite_names"):
        filters.speclite_names = {}


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if value is not None and hasattr(obj, name):
        setattr(obj, name, value)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(_json_sanitize(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value)


def _sample_percentiles(samples: Any) -> tuple[float, float, float]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(p16), float(p50), float(p84)


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _optional_float(raw: dict[str, str], key: str) -> float | None:
    if key not in raw or raw[key] == "":
        return None
    value = _finite_float(raw[key])
    return None if not np.isfinite(value) else value


def _percentile_payload(prefix: str, values: Any, *, add_erg_s: bool = False) -> dict[str, float]:
    p16, p50, p84 = _sample_percentiles(values)
    out = {
        f"{prefix}16": p16,
        prefix: p50,
        f"{prefix}84": p84,
    }
    if add_erg_s:
        out.update(
            {
                f"{prefix}_erg_s16": p16 + 7.0 if np.isfinite(p16) else float("nan"),
                f"{prefix}_erg_s": p50 + 7.0 if np.isfinite(p50) else float("nan"),
                f"{prefix}_erg_s84": p84 + 7.0 if np.isfinite(p84) else float("nan"),
            }
        )
    return out


def _predictive_or_empty(fitter: Any) -> dict[str, Any]:
    try:
        predictive = fitter.predict(kind="photometry")
    except Exception as exc:
        return {"predictive_summary_error": f"{type(exc).__name__}: {exc}"}
    return predictive if isinstance(predictive, dict) else {}


def _extract_predictive_summary(fitter: Any) -> dict[str, Any]:
    predictive = _predictive_or_empty(fitter)
    if "predictive_summary_error" in predictive:
        return predictive

    out: dict[str, Any] = {}
    if "log_agn_bol_luminosity_fit" in predictive:
        out.update(_percentile_payload("log_agn_bol_luminosity_fit", predictive["log_agn_bol_luminosity_fit"], add_erg_s=True))

    if "log_disk_luminosity_fit" in predictive:
        out.update(_percentile_payload("log_agn_l5100_fit", predictive["log_disk_luminosity_fit"], add_erg_s=True))
        lambda_l5100 = np.asarray(predictive["log_disk_luminosity_fit"], dtype=float) + np.log10(5100.0)
        out.update(_percentile_payload("log_agn_lambda_l5100_fit", lambda_l5100, add_erg_s=True))

    if "fracAGN_5100_fit" in predictive:
        out.update(_percentile_payload("fracAGN_5100_fit", predictive["fracAGN_5100_fit"]))

    if "log_dust_luminosity_fit" in predictive:
        out.update(_percentile_payload("log_dust_luminosity_fit", predictive["log_dust_luminosity_fit"], add_erg_s=True))

    if "gal_sfr_table" in predictive:
        sfr_table = np.asarray(predictive["gal_sfr_table"], dtype=float)
        if sfr_table.size:
            current_sfr = sfr_table[..., -1]
            out.update(_sfr_payload_from_linear("sfr_current_fit", current_sfr))
            n_recent = min(100, sfr_table.shape[-1])
            if n_recent > 0:
                out.update(_sfr_payload_from_linear("sfr_100myr_fit", np.nanmean(sfr_table[..., -n_recent:], axis=-1)))

    return out


def _sfr_payload_from_linear(prefix: str, values: Any) -> dict[str, float]:
    p16, p50, p84 = _sample_percentiles(values)
    positive = np.asarray(values, dtype=float).reshape(-1)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        log16, log50, log84 = np.percentile(np.log10(positive), [16.0, 50.0, 84.0])
    else:
        log16 = log50 = log84 = float("nan")
    return {
        f"{prefix}16": p16,
        prefix: p50,
        f"{prefix}84": p84,
        f"log_{prefix}16": float(log16),
        f"log_{prefix}": float(log50),
        f"log_{prefix}84": float(log84),
    }


def _extract_sfr_summary(samples: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(np, "isfinite"):
        return {"sfr_sample_key": None}

    for key in SFR_SAMPLE_KEYS:
        if key not in samples:
            continue

        p16, p50, p84 = _sample_percentiles(samples[key])
        out: dict[str, Any] = {"sfr_sample_key": key}
        key_lower = key.lower()
        if key_lower.startswith("log"):
            out.update(
                {
                    "log_sfr16": p16,
                    "log_sfr": p50,
                    "log_sfr84": p84,
                    "sfr16": float(10.0**p16) if np.isfinite(p16) else float("nan"),
                    "sfr": float(10.0**p50) if np.isfinite(p50) else float("nan"),
                    "sfr84": float(10.0**p84) if np.isfinite(p84) else float("nan"),
                }
            )
        else:
            values = np.asarray(samples[key], dtype=float).reshape(-1)
            positive = values[np.isfinite(values) & (values > 0.0)]
            if positive.size:
                log16, log50, log84 = np.percentile(np.log10(positive), [16.0, 50.0, 84.0])
            else:
                log16 = log50 = log84 = float("nan")
            out.update(
                {
                    "sfr16": p16,
                    "sfr": p50,
                    "sfr84": p84,
                    "log_sfr16": float(log16),
                    "log_sfr": float(log50),
                    "log_sfr84": float(log84),
                }
            )
        return out

    return {"sfr_sample_key": None}


def _truth_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    qso_weight = _finite_float(row.get("chimera_QSO_weight"))
    log_weight = np.log10(qso_weight) if np.isfinite(qso_weight) and qso_weight > 0.0 else float("nan")

    for key in OPTIONAL_MANIFEST_FLOAT_FIELDS:
        value = row.get(key)
        if value is not None:
            out[key] = float(value)

    for key in ("logL5100_QSO", "logL3000_QSO", "logL1350_QSO"):
        if key in row and np.isfinite(row[key]) and np.isfinite(log_weight):
            out[key.replace("_QSO", "_chimera")] = float(row[key] + log_weight)

    if "SFR_MED_GAL" in row and np.isfinite(row["SFR_MED_GAL"]):
        out["sfr_truth"] = float(row["SFR_MED_GAL"])
        out["log_sfr_truth"] = float(np.log10(row["SFR_MED_GAL"])) if row["SFR_MED_GAL"] > 0.0 else float("nan")
    return out


def _fit_result_summary(fit_result: Any) -> Any:
    if isinstance(fit_result, dict):
        return fit_result.get("summary", {})

    summary = getattr(fit_result, "summary", None)
    if summary is None:
        return {}
    if hasattr(summary, "items"):
        return dict(summary)
    return summary


def _is_nuts_initialization_error(exc: BaseException) -> bool:
    message = str(exc)
    return "Cannot find valid initial parameters" in message


def _call_fit_compat(fitter: Any, **kwargs: Any) -> Any:
    """Call fit() with only the keywords supported by the installed backend."""
    signature = inspect.signature(fitter.fit)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_var_kwargs:
        return fitter.fit(**kwargs)

    supported = {name for name, param in parameters.items() if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    fit_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return fitter.fit(**fit_kwargs)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_from_manifest(raw: dict[str, str], filter_names: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    if filter_names is None:
        filter_names, _, _ = _load_backend("jaxsedfit")
    row: dict[str, Any] = {
        "id": raw["object_id"],
        "ID_COSMOS": raw["COSMOS_ID0"],
        "COSMOS_ID0": raw["COSMOS_ID0"],
        "redshift": float(raw["redshift"]),
        "chimera_QSO_weight": float(raw["chimera_QSO_weight"]),
        "resample_weight": float(raw["resample_weight"]),
        "log_stellar_mass_truth": float(raw["log_stellar_mass_truth"]),
        "logLbol_QSO": float(raw["logLbol_QSO"]),
        "logLbol_chimera": float(raw["logLbol_chimera"]),
        "luminosity_bin": raw.get("luminosity_bin", ""),
        "fit_index": int(raw["fit_index"]),
    }
    for key in OPTIONAL_MANIFEST_FLOAT_FIELDS:
        value = _optional_float(raw, key)
        if value is not None:
            row[key] = value
    for name in filter_names:
        row[name] = float(raw[name])
        row[f"{name}_err"] = float(raw[f"{name}_err"])
    return row


def _select_manifest_entry(args: argparse.Namespace) -> dict[str, str]:
    rows = _load_manifest(args.manifest)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} manifest rows, found {len(rows)}.")

    if not args.object_id:
        raise RuntimeError("--object-id is required; array-index and scheduler environment selection are not supported.")

    matches = [row for row in rows if str(row["object_id"]) == str(args.object_id)]
    if len(matches) != 1:
        if not matches:
            raise RuntimeError(f"object_id={args.object_id!r} matched 0 manifest rows.")
        raise RuntimeError(
            f"object_id={args.object_id!r} matched {len(matches)} manifest rows; "
            "object_id values must be unique."
        )
    return matches[0]


def _run_fit(
    row: dict[str, Any],
    args: argparse.Namespace,
    sed_pdf_path: Path,
    corner_pdf_path: Path,
    trace_pdf_path: Path,
) -> dict[str, Any]:
    _, build_chimera_fit_config, fitter_cls = _load_backend(args.backend)
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(args.dsps_ssp_fn))
    _patch_backend_config_compat(cfg)
    dense_mass = getattr(args, "dense_mass", None)
    max_tree_depth = getattr(args, "max_tree_depth", None)
    cfg.inference.seed = int(args.seed_base + row["fit_index"])
    cfg.inference.method = args.sampler
    _set_if_present(cfg.inference, "map_steps", int(args.optax_steps))
    _set_if_present(cfg.inference, "learning_rate", float(args.optax_lr))
    _set_if_present(cfg.inference, "num_warmup", int(args.nuts_warmup))
    _set_if_present(cfg.inference, "num_samples", int(args.nuts_samples))
    _set_if_present(cfg.inference, "num_chains", int(args.nuts_chains))
    _set_if_present(cfg.inference, "target_accept_prob", float(args.target_accept_prob))
    _set_if_present(cfg.inference, "dense_mass", dense_mass)
    _set_if_present(cfg.inference, "max_tree_depth", max_tree_depth)
    if hasattr(cfg, "output"):
        _set_if_present(cfg.output, "output_dir", str(args.output_dir))
        _set_if_present(cfg.output, "fig_path", str(sed_pdf_path))
        _set_if_present(cfg.output, "plot_fig", False)
        _set_if_present(cfg.output, "save_fig", True)
        _set_if_present(cfg.output, "save_result", False)
        _set_if_present(cfg.output, "show_plot", False)

    def _fit_once(use_map_init: bool | None) -> tuple[Any, Any]:
        if use_map_init is not None:
            _set_if_present(cfg.inference, "use_map_init", bool(use_map_init))
        fitter = fitter_cls(cfg)
        fit_result = _call_fit_compat(
            fitter,
            prior_config=getattr(cfg, "prior_config", None),
            dsps_ssp_fn=cfg.galaxy.dsps_ssp_fn,
            optax_steps=args.optax_steps,
            optax_lr=args.optax_lr,
            nuts_warmup=args.nuts_warmup,
            nuts_samples=args.nuts_samples,
            nuts_chains=args.nuts_chains,
            dense_mass=dense_mass,
            nuts_dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            nuts_max_tree_depth=max_tree_depth,
            use_map_init=use_map_init,
            ns_live_points=args.ns_live_points,
            ns_max_samples=args.ns_max_samples,
            ns_dlogz=args.ns_dlogz,
            ns_resamples=args.ns_resamples,
            ns_difficult_model=args.ns_difficult_model,
            ns_parameter_estimation=args.ns_parameter_estimation,
            ns_num_parallel_workers=args.ns_num_parallel_workers,
            ns_init_efficiency_threshold=args.ns_init_efficiency_threshold,
            ns_max_likelihood_evals=args.ns_max_likelihood_evals,
            ns_efficiency_threshold=args.ns_efficiency_threshold,
            target_accept_prob=args.target_accept_prob,
            plot_fig=False,
            save_fig=True,
            fig_path=sed_pdf_path,
            save_result=False,
            progress_bar=args.progress_bar,
        )
        return fitter, fit_result

    initial_use_map_init = True if args.use_map_init is None else bool(args.use_map_init)
    used_map_init = initial_use_map_init
    map_init_fallback = False
    try:
        fitter, fit_result = _fit_once(initial_use_map_init)
    except RuntimeError as exc:
        should_retry = args.sampler == "optax+nuts" and initial_use_map_init and _is_nuts_initialization_error(exc)
        if not should_retry:
            raise
        print("[manifest-fit] MAP-initialized NUTS failed; retrying with NumPyro initialization", flush=True)
        map_init_fallback = True
        used_map_init = False
        fitter, fit_result = _fit_once(False)

    fitter.plot_corner(output_path=corner_pdf_path)
    fitter.plot_trace(output_path=trace_pdf_path)
    logm_samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    logm16, recovered_logm, logm84 = np.percentile(logm_samples, [16.0, 50.0, 84.0])
    truth_logm = float(row["log_stellar_mass_truth"])
    sfr_summary = _extract_sfr_summary(fitter.samples)
    predictive_summary = _extract_predictive_summary(fitter)
    truth_summary = _truth_payload(row)
    payload = {
        "status": "success",
        "fit_index": int(row["fit_index"]),
        "object_id": str(row["id"]),
        "COSMOS_ID0": str(row["COSMOS_ID0"]),
        "ID_COSMOS": str(row["ID_COSMOS"]),
        "redshift": float(row["redshift"]),
        "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
        "resample_weight": float(row["resample_weight"]),
        "log_stellar_mass_truth": truth_logm,
        "logLbol_QSO": float(row["logLbol_QSO"]),
        "logLbol_chimera": float(row["logLbol_chimera"]),
        "luminosity_bin": str(row["luminosity_bin"]),
        **truth_summary,
        "recovered_logm": float(recovered_logm),
        "logm16": float(logm16),
        "logm84": float(logm84),
        "residual_log_ratio": float(recovered_logm - truth_logm),
        **sfr_summary,
        **predictive_summary,
        "sed_pdf_path": str(sed_pdf_path),
        "corner_pdf_path": str(corner_pdf_path),
        "trace_pdf_path": str(trace_pdf_path),
        "fit_summary": _fit_result_summary(fit_result),
        "sampler": args.sampler,
        "backend": _normalize_backend(args.backend),
        "optax_steps": int(args.optax_steps),
        "optax_lr": float(args.optax_lr),
        "nuts_warmup": int(args.nuts_warmup),
        "nuts_samples": int(args.nuts_samples),
        "nuts_chains": int(args.nuts_chains),
        "dense_mass": dense_mass,
        "max_tree_depth": max_tree_depth,
        "use_map_init": used_map_init,
        "map_init_fallback": map_init_fallback,
        "target_accept_prob": float(args.target_accept_prob),
    }
    if args.sampler == "ns" or any(
        value is not None
        for value in (
            args.ns_live_points,
            args.ns_max_samples,
            args.ns_dlogz,
            args.ns_resamples,
            args.ns_num_parallel_workers,
            args.ns_init_efficiency_threshold,
            args.ns_max_likelihood_evals,
            args.ns_efficiency_threshold,
        )
    ) or args.ns_difficult_model or args.ns_parameter_estimation:
        payload.update(
            {
                "ns_live_points": None if args.ns_live_points is None else int(args.ns_live_points),
                "ns_max_samples": None if args.ns_max_samples is None else int(args.ns_max_samples),
                "ns_dlogz": None if args.ns_dlogz is None else float(args.ns_dlogz),
                "ns_resamples": None if args.ns_resamples is None else int(args.ns_resamples),
            }
        )
        if args.ns_difficult_model:
            payload["ns_difficult_model"] = True
        if args.ns_parameter_estimation:
            payload["ns_parameter_estimation"] = True
        if args.ns_num_parallel_workers is not None:
            payload["ns_num_parallel_workers"] = int(args.ns_num_parallel_workers)
        if args.ns_init_efficiency_threshold is not None:
            payload["ns_init_efficiency_threshold"] = float(args.ns_init_efficiency_threshold)
        if args.ns_max_likelihood_evals is not None:
            payload["ns_max_likelihood_evals"] = int(args.ns_max_likelihood_evals)
        if args.ns_efficiency_threshold is not None:
            payload["ns_efficiency_threshold"] = float(args.ns_efficiency_threshold)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one jaxsedfit/grahspj fit from a precomputed manifest row.")
    parser.add_argument("--manifest", type=Path, default=Path("hpc_outputs/loglbol_mass_retrieval/fit_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("hpc_outputs/loglbol_mass_retrieval"))
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=20231011)
    parser.add_argument(
        "--backend",
        choices=tuple(sorted(BACKEND_ALIASES)),
        default="jaxsedfit",
        help="GRAHSPJ/JAXSEDFit backend name; jaxsed and jaxsedfit are accepted aliases.",
    )
    parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax+nuts")
    parser.add_argument("--optax-steps", type=int, default=300)
    parser.add_argument("--optax-lr", type=float, default=1.0e-2)
    parser.add_argument("--nuts-warmup", type=int, default=300)
    parser.add_argument("--nuts-samples", type=int, default=300)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument(
        "--use-map-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Initialize NUTS from the Optax/MAP solution. Use --no-use-map-init to let NumPyro initialize NUTS independently.",
    )
    parser.add_argument("--ns-live-points", type=int, default=None)
    parser.add_argument("--ns-max-samples", type=int, default=None)
    parser.add_argument("--ns-dlogz", type=float, default=None)
    parser.add_argument("--ns-resamples", type=int, default=None)
    parser.add_argument("--ns-difficult-model", action="store_true")
    parser.add_argument("--ns-parameter-estimation", action="store_true")
    parser.add_argument("--ns-num-parallel-workers", type=int, default=None)
    parser.add_argument("--ns-init-efficiency-threshold", type=float, default=None)
    parser.add_argument("--ns-max-likelihood-evals", type=int, default=None)
    parser.add_argument("--ns-efficiency-threshold", type=float, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dsps_ssp_fn = args.dsps_ssp_fn.expanduser().resolve()
    args.backend = _normalize_backend(args.backend)

    filter_names, _, _ = _load_backend(args.backend)
    raw = _select_manifest_entry(args)
    row = _row_from_manifest(raw, filter_names)
    stem = f"{row['fit_index']:05d}_COSMOS{_safe_id(str(row['COSMOS_ID0']))}_{_safe_id(str(row['id']))}"
    success_path = args.output_dir / "results" / f"{stem}.json"
    failure_path = args.output_dir / "failures" / f"{stem}.json"
    sed_pdf_path = args.output_dir / "sed_pdfs" / f"{stem}.pdf"
    corner_pdf_path = args.output_dir / "corner_pdfs" / f"{stem}.pdf"
    trace_pdf_path = args.output_dir / "trace_pdfs" / f"{stem}.pdf"

    print(
        f"[manifest-fit] fit_index={row['fit_index']} COSMOS_ID0={row['COSMOS_ID0']} "
        f"object_id={row['id']} backend={args.backend}",
        flush=True,
    )

    if args.dry_run:
        print(json.dumps({"status": "dry_run", **{k: row[k] for k in ("fit_index", "id", "COSMOS_ID0", "redshift", "logLbol_chimera")}}, indent=2))
        return 0

    try:
        payload = _run_fit(row, args, sed_pdf_path, corner_pdf_path, trace_pdf_path)
        _atomic_write_json(success_path, payload)
        print(f"[manifest-fit] wrote {success_path}", flush=True)
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "fit_index": int(row["fit_index"]),
            "object_id": str(row["id"]),
            "COSMOS_ID0": str(row["COSMOS_ID0"]),
            "ID_COSMOS": str(row["ID_COSMOS"]),
            "redshift": float(row["redshift"]),
            "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
            "log_stellar_mass_truth": float(row["log_stellar_mass_truth"]),
            "logLbol_QSO": float(row["logLbol_QSO"]),
            "logLbol_chimera": float(row["logLbol_chimera"]),
            "luminosity_bin": str(row["luminosity_bin"]),
            "backend": _normalize_backend(args.backend),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(failure_path, payload)
        print(f"[manifest-fit] failed; wrote {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
