from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_OBJECT_ID = "022754.38-073455.0_869049_0.0001"
DEFAULT_N_WAVE = 1024

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hpc import run_grahsp_manifest_fit as grahsp_runner
from hpc import run_manifest_fit as jaxsedfit_runner


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_sanitize(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value)


def _find_jaxsedfit_root(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            REPO_ROOT / "jaxsedfit",
            REPO_ROOT / "grahspj_latest",
            REPO_ROOT / "grahspj",
        ]
    )
    for root in candidates:
        root = root.expanduser().resolve()
        if (root / "src" / "jaxsedfit").is_dir():
            return root
    return None


def _prepare_jaxsedfit_import(jaxsedfit_root: Path | None) -> None:
    if jaxsedfit_root is None:
        return
    src_root = jaxsedfit_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _select_manifest_row(manifest: Path, object_id: str) -> dict[str, str]:
    rows = jaxsedfit_runner._load_manifest(manifest)
    matches = [row for row in rows if str(row["object_id"]) == str(object_id)]
    if len(matches) != 1:
        if not matches:
            raise RuntimeError(f"object_id={object_id!r} matched 0 manifest rows in {manifest}.")
        raise RuntimeError(f"object_id={object_id!r} matched {len(matches)} manifest rows in {manifest}.")
    return matches[0]


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _run_jaxsedfit(row: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    filter_names, build_chimera_fit_config, fitter_cls = jaxsedfit_runner._load_backend("jaxsedfit")
    row = jaxsedfit_runner._row_from_manifest(row, filter_names)

    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(args.dsps_ssp_fn))
    jaxsedfit_runner._patch_backend_config_compat(cfg)
    cfg.galaxy.n_wave = int(args.n_wave)
    cfg.inference.seed = int(args.seed_base + row["fit_index"])
    cfg.inference.method = args.sampler
    cfg.inference.map_steps = int(args.optax_steps)
    cfg.inference.learning_rate = float(args.optax_lr)
    cfg.inference.num_warmup = int(args.nuts_warmup)
    cfg.inference.num_samples = int(args.nuts_samples)
    cfg.inference.num_chains = int(args.nuts_chains)
    cfg.inference.target_accept_prob = float(args.target_accept_prob)
    _set_if_present(cfg.inference, "max_tree_depth", int(args.max_tree_depth))
    cfg.inference.ns_num_live_points = args.ns_live_points
    cfg.inference.ns_max_samples = args.ns_max_samples
    cfg.inference.ns_dlogz = args.ns_dlogz
    cfg.inference.ns_resamples = args.ns_resamples

    fitter = fitter_cls(cfg)
    output_cfg = getattr(cfg, "output", None)
    if output_cfg is None:
        fit_result = fitter.fit(
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
            target_accept_prob=args.target_accept_prob,
            plot_fig=False,
            save_fig=False,
            save_result=False,
            progress_bar=args.progress_bar,
        )
    else:
        output_cfg.plot_fig = False
        output_cfg.save_fig = False
        output_cfg.save_result = False
        fit_result = fitter.fit(progress_bar=args.progress_bar)
    pred = fitter.predict()
    samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    logm16, recovered_logm, logm84 = np.percentile(samples, [16.0, 50.0, 84.0])
    truth_logm = float(row["log_stellar_mass_truth"])

    paths: dict[str, str] = {}
    if not args.skip_jaxsedfit_plots:
        paths["sed_pdf_path"] = str(output_dir / "jaxsedfit_sed.pdf")
        paths["corner_pdf_path"] = str(output_dir / "jaxsedfit_corner.pdf")
        paths["trace_pdf_path"] = str(output_dir / "jaxsedfit_trace.pdf")
        fitter.plot_sed(output_path=paths["sed_pdf_path"])
        fitter.plot_corner(output_path=paths["corner_pdf_path"], max_params=args.corner_max_params)
        fitter.plot_trace(output_path=paths["trace_pdf_path"])

    payload = {
        "status": "success",
        "backend": "jaxsedfit",
        "fit_method": "jaxsedfit",
        "fit_index": int(row["fit_index"]),
        "object_id": str(row["id"]),
        "COSMOS_ID0": str(row["COSMOS_ID0"]),
        "ID_COSMOS": str(row["ID_COSMOS"]),
        "redshift": float(row["redshift"]),
        "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
        "log_stellar_mass_truth": truth_logm,
        "logLbol_QSO": float(row["logLbol_QSO"]),
        "logLbol_chimera": float(row["logLbol_chimera"]),
        "luminosity_bin": str(row["luminosity_bin"]),
        "recovered_logm": float(recovered_logm),
        "logm16": float(logm16),
        "logm84": float(logm84),
        "residual_log_ratio": float(recovered_logm - truth_logm),
        "n_wave": int(args.n_wave),
        "sampler": args.sampler,
        "fit_summary": jaxsedfit_runner._fit_result_summary(fit_result),
        **paths,
    }
    return payload, fitter, pred


def _run_grahsp(row: dict[str, str], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    grahsp_args = argparse.Namespace(
        manifest=args.manifest,
        output_dir=output_dir,
        expected_count=args.expected_count,
        python_executable=args.python_executable,
        sampler_script=args.grahsp_sampler_script,
        cigale_root=args.grahsp_cigale_root,
        cores=args.grahsp_cores,
        num_live_points=args.grahsp_live_points,
        num_posterior_samples=args.grahsp_posterior_samples,
        mass_max=args.grahsp_mass_max,
        cache_max=args.grahsp_cache_max,
        keep_pdfs=args.keep_grahsp_pdfs,
    )
    filter_names, _, _ = jaxsedfit_runner._load_backend("jaxsedfit")
    manifest_row = jaxsedfit_runner._row_from_manifest(row, filter_names)
    grahsp_row = grahsp_runner._row_from_manifest(row)
    # Keep the exact jaxsedfit filter-derived row values when possible.
    grahsp_row.update({key: value for key, value in manifest_row.items() if key in grahsp_row})
    return grahsp_runner._run_grahsp(grahsp_row, grahsp_args)


def _positive(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.isfinite(arr) & (arr > 0.0)


def _mass_title(label: str, recovered: float, lo: float, hi: float, truth: float) -> str:
    values = np.asarray([recovered, lo, hi, truth], dtype=float)
    if not np.all(np.isfinite(values)):
        return label
    return f"{label}\nrecovered logM={recovered:.3f} (+{hi - recovered:.2f}/-{recovered - lo:.2f}), delta={recovered - truth:+.3f}"


def _plot_side_by_side(output_path: Path, row: dict[str, str], fitter: Any, pred: dict[str, Any], grahsp_payload: dict[str, Any]) -> None:
    from jaxsedfit.plotting import (
        _COMPONENT_STYLE,
        _median_site,
        _median_site_sum,
        _percentile_site_sum,
        _to_display_flux_density,
    )

    sed_path = grahsp_payload.get("sed_mjy_csv_path")
    if not sed_path:
        return
    grahsp_sed = Table.read(sed_path, format="ascii.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True, layout="constrained")
    y_parts: list[np.ndarray] = []

    obs_wave_a = _median_site(pred, "obs_wave")
    phot_wave_a = np.asarray([flt.effective_wavelength for flt in fitter.context.filters], dtype=float)
    obs_flux = np.asarray(fitter.config.photometry.fluxes, dtype=float)
    obs_err = np.asarray(fitter.config.photometry.errors, dtype=float)
    model_flux = _median_site(pred, "pred_fluxes")
    y_parts.extend([obs_flux, model_flux])
    truth_logm = float(row["log_stellar_mass_truth"])
    jax_samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    jax_logm16, jax_recovered_logm, jax_logm84 = np.percentile(jax_samples, [16.0, 50.0, 84.0])

    labels_seen = set()
    for keys, label, color, lw in _COMPONENT_STYLE:
        key_tuple = tuple(keys if isinstance(keys, (list, tuple, set)) else [keys])
        if not any(key in pred for key in key_tuple):
            continue
        component = _to_display_flux_density(obs_wave_a, _median_site_sum(pred, key_tuple))
        if not np.any(_positive(component)):
            continue
        lo = _to_display_flux_density(obs_wave_a, _percentile_site_sum(pred, key_tuple, 16.0))
        hi = _to_display_flux_density(obs_wave_a, _percentile_site_sum(pred, key_tuple, 84.0))
        y_parts.extend([component, lo, hi])
        plot_label = label if label not in labels_seen else "_nolegend_"
        labels_seen.add(label)
        if np.any(_positive(lo) & _positive(hi)):
            axes[0].fill_between(
                obs_wave_a,
                np.clip(np.minimum(lo, hi), 1e-300, None),
                np.clip(np.maximum(lo, hi), 1e-300, None),
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
        ls = "-" if "total_obs_sed" in key_tuple else ("--" if "host_obs_sed" in key_tuple else ":")
        axes[0].plot(obs_wave_a, component, color=color, lw=lw, ls=ls, alpha=0.9, label=plot_label)
    axes[0].errorbar(phot_wave_a, obs_flux, yerr=obs_err, fmt="o", color="#c53030", ms=5, capsize=2, label="Observed photometry")
    axes[0].scatter(phot_wave_a, model_flux, color="#111111", marker="s", s=28, label="Model photometry")
    axes[0].set_title(_mass_title("JAXSEDFit", jax_recovered_logm, jax_logm16, jax_logm84, truth_logm))

    wave_a = np.asarray(grahsp_sed["wavelength"], dtype=float) * 1.0e4
    for col, label, color, lw, ls in (
        ("total", "Model total", "#000000", 2.0, "-"),
        ("Stellar (attenuated)", "Host stellar", "#2b6cb0", 1.6, "--"),
        ("Nebular emission", "Nebular emission", "#319795", 1.1, ":"),
        ("Dust", "Host dust", "#b7791f", 1.5, ":"),
        ("AGN disk", "AGN disk", "#c05621", 1.2, ":"),
        ("AGN torus", "Torus", "#805ad5", 1.2, ":"),
        ("AGN lines", "AGN lines", "#d53f8c", 1.0, ":"),
    ):
        if col not in grahsp_sed.colnames:
            continue
        values = np.asarray(grahsp_sed[col], dtype=float)
        if not np.any(_positive(values)):
            continue
        y_parts.append(values)
        axes[1].plot(wave_a, values, color=color, lw=lw, ls=ls, alpha=0.95, label=label)
    axes[1].errorbar(phot_wave_a, obs_flux, yerr=obs_err, fmt="o", color="#c53030", ms=5, capsize=2, label="Observed photometry")
    axes[1].set_title(
        _mass_title(
            "External GRAHSP",
            float(grahsp_payload.get("recovered_logm", np.nan)),
            float(grahsp_payload.get("logm16", np.nan)),
            float(grahsp_payload.get("logm84", np.nan)),
            truth_logm,
        )
    )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Observed-frame wavelength (A)")
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.set_xlim(1e2, 1e6)
    axes[0].set_ylabel("Flux density (mJy)")

    finite_y = np.concatenate([np.ravel(np.asarray(y, dtype=float)) for y in y_parts])
    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0.0)]
    if finite_y.size:
        ymax = float(np.nanmax(finite_y))
        ymin = float(np.nanmin(finite_y[finite_y >= ymax * 1e-7])) if np.any(finite_y >= ymax * 1e-7) else float(np.nanmin(finite_y))
        for ax in axes:
            ax.set_ylim(ymin * 0.7, ymax * 1.8)

    fig.suptitle(
        f"{row['object_id']} | z={float(row['redshift']):.4f} | "
        f"Chimera logM={float(row['log_stellar_mass_truth']):.3f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one JAXSEDFit vs external GRAHSP comparison from fit_manifest.csv.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "fit_manifest.csv")
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--expected-count", type=int, default=13558)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "hpc_outputs" / "grahsp_vs_jaxsedfit_single")
    parser.add_argument("--dsps-ssp-fn", type=Path, default=PROJECT_ROOT / "tempdata.h5")
    parser.add_argument("--jaxsedfit-root", type=Path, default=None)
    parser.add_argument("--n-wave", type=int, default=DEFAULT_N_WAVE)
    parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax+nuts")
    parser.add_argument("--seed-base", type=int, default=20231011)
    parser.add_argument("--optax-steps", type=int, default=400)
    parser.add_argument("--optax-lr", type=float, default=5.0e-3)
    parser.add_argument("--nuts-warmup", type=int, default=400)
    parser.add_argument("--nuts-samples", type=int, default=400)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--ns-live-points", type=int, default=700)
    parser.add_argument("--ns-max-samples", type=int, default=8000)
    parser.add_argument("--ns-dlogz", type=float, default=10.0)
    parser.add_argument("--ns-resamples", type=int, default=None)
    parser.add_argument("--grahsp-sampler-script", type=Path, default=grahsp_runner.SAMPLER_SCRIPT)
    parser.add_argument("--grahsp-cigale-root", type=Path, default=grahsp_runner.CIGALE_ROOT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--grahsp-cores", type=int, default=1)
    parser.add_argument("--grahsp-live-points", type=int, default=800)
    parser.add_argument("--grahsp-posterior-samples", type=int, default=3000)
    parser.add_argument(
        "--grahsp-mass-max",
        type=float,
        default=13.0,
        help="Maximum stellar mass written to external GRAHSP pcigale.ini; default matches notebook 13.",
    )
    parser.add_argument("--grahsp-cache-max", type=int, default=5000)
    parser.add_argument("--keep-grahsp-pdfs", action="store_true", help="Keep and copy GRAHSP PDFs. By default only CSV products are retained.")
    parser.add_argument("--corner-max-params", type=int, default=8)
    parser.add_argument("--skip-jaxsedfit", action="store_true")
    parser.add_argument("--skip-grahsp", action="store_true")
    parser.add_argument("--skip-jaxsedfit-plots", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dsps_ssp_fn = args.dsps_ssp_fn.expanduser().resolve()
    args.grahsp_sampler_script = args.grahsp_sampler_script.expanduser().resolve()
    args.grahsp_cigale_root = args.grahsp_cigale_root.expanduser().resolve()
    args.python_executable = args.python_executable.expanduser().resolve()

    jaxsedfit_root = _find_jaxsedfit_root(args.jaxsedfit_root)
    _prepare_jaxsedfit_import(jaxsedfit_root)

    raw = _select_manifest_row(args.manifest, args.object_id)
    stem = f"{int(raw['fit_index']):05d}_COSMOS{_safe_id(str(raw['COSMOS_ID0']))}_{_safe_id(str(raw['object_id']))}"
    run_dir = args.output_dir / stem

    dry_payload = {
        "status": "dry_run",
        "project_root": PROJECT_ROOT,
        "jaxsedfit_root": jaxsedfit_root,
        "manifest": args.manifest,
        "object_id": raw["object_id"],
        "fit_index": int(raw["fit_index"]),
        "COSMOS_ID0": raw["COSMOS_ID0"],
        "output_dir": run_dir,
        "dsps_ssp_fn": args.dsps_ssp_fn,
        "grahsp_sampler_script": args.grahsp_sampler_script,
        "grahsp_cigale_root": args.grahsp_cigale_root,
        "grahsp_backend": "external_grahsp",
        "grahsp_mass_max": float(args.grahsp_mass_max),
        "n_wave": int(args.n_wave),
    }
    if args.dry_run:
        print(json.dumps(_json_sanitize(dry_payload), indent=2, sort_keys=True))
        return 0

    if jaxsedfit_root is None:
        print("[comparison] no sibling jaxsedfit checkout found; using installed jaxsedfit from the active environment", flush=True)
    if not args.dsps_ssp_fn.is_file():
        raise FileNotFoundError(f"Missing DSPS SSP file: {args.dsps_ssp_fn}")
    if not args.skip_grahsp:
        if not args.grahsp_sampler_script.is_file():
            raise FileNotFoundError(f"Missing GRAHSP sampler script: {args.grahsp_sampler_script}")
        if not args.grahsp_cigale_root.is_dir():
            raise FileNotFoundError(f"Missing GRAHSP/CIGALE root: {args.grahsp_cigale_root}")

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[comparison] object_id={raw['object_id']} fit_index={raw['fit_index']}", flush=True)
    print(f"[comparison] output_dir={run_dir}", flush=True)

    jax_payload = None
    fitter = None
    pred = None
    if not args.skip_jaxsedfit:
        print("[comparison] running JAXSEDFit", flush=True)
        jax_payload, fitter, pred = _run_jaxsedfit(raw, args, run_dir)
        _write_json(run_dir / "jaxsedfit_result.json", jax_payload)
    else:
        print("[comparison] skipping JAXSEDFit", flush=True)

    grahsp_payload = None
    if not args.skip_grahsp:
        print("[comparison] running external GRAHSP", flush=True)
        grahsp_payload = _run_grahsp(raw, args, run_dir / "grahsp")
        _write_json(run_dir / "grahsp_result.json", grahsp_payload)
    else:
        print("[comparison] skipping external GRAHSP", flush=True)

    comparison_plot = ""
    if fitter is not None and pred is not None and grahsp_payload is not None:
        comparison_plot = str(run_dir / "grahsp_vs_jaxsedfit_sed.png")
        _plot_side_by_side(Path(comparison_plot), raw, fitter, pred, grahsp_payload)

    summary = {
        "status": "success",
        "object_id": raw["object_id"],
        "fit_index": int(raw["fit_index"]),
        "COSMOS_ID0": raw["COSMOS_ID0"],
        "luminosity_bin": raw.get("luminosity_bin", ""),
        "redshift": float(raw["redshift"]),
        "log_stellar_mass_truth": float(raw["log_stellar_mass_truth"]),
        "jaxsedfit_backend": "jaxsedfit",
        "grahsp_backend": "external_grahsp",
        "grahsp_mass_max": float(args.grahsp_mass_max),
        "jaxsedfit": jax_payload,
        "grahsp": grahsp_payload,
        "comparison_plot": comparison_plot,
    }
    _write_json(run_dir / "comparison_summary.json", summary)
    print(f"[comparison] wrote {run_dir / 'comparison_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
