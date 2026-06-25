from __future__ import annotations

import argparse
import csv
import importlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ALIASES = {
    "jaxsed": "grahspj",
    "jaxsedfit": "grahspj",
    "grahspj": "grahspj",
}


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
    cfg.inference.seed = int(args.seed_base + row["fit_index"])
    fitter = fitter_cls(cfg)
    fit_result = fitter.fit(
        fit_method=args.sampler,
        prior_config=cfg.prior_config,
        dsps_ssp_fn=cfg.galaxy.dsps_ssp_fn,
        optax_steps=args.optax_steps,
        optax_lr=args.optax_lr,
        nuts_warmup=args.nuts_warmup,
        nuts_samples=args.nuts_samples,
        nuts_chains=args.nuts_chains,
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
    fitter.plot_corner(output_path=corner_pdf_path)
    fitter.plot_trace(output_path=trace_pdf_path)
    logm_samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    logm16, recovered_logm, logm84 = np.percentile(logm_samples, [16.0, 50.0, 84.0])
    truth_logm = float(row["log_stellar_mass_truth"])
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
        "recovered_logm": float(recovered_logm),
        "logm16": float(logm16),
        "logm84": float(logm84),
        "residual_log_ratio": float(recovered_logm - truth_logm),
        "sed_pdf_path": str(sed_pdf_path),
        "corner_pdf_path": str(corner_pdf_path),
        "trace_pdf_path": str(trace_pdf_path),
        "fit_summary": fit_result.get("summary", {}),
        "sampler": args.sampler,
        "backend": _normalize_backend(args.backend),
        "optax_steps": int(args.optax_steps),
        "optax_lr": float(args.optax_lr),
        "nuts_warmup": int(args.nuts_warmup),
        "nuts_samples": int(args.nuts_samples),
        "nuts_chains": int(args.nuts_chains),
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
    parser.add_argument("--expected-count", type=int, default=13558)
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
