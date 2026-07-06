from __future__ import annotations

import argparse
import csv
import importlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

import run_manifest_fit as manifest_fit


def _load_spectra_manifest(path: Path) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        chimera_id = str(row.get("chimera_id", "")).strip()
        if chimera_id and row.get("status", "success") == "success":
            out[chimera_id] = row
    return out


def _resolve_spectrum_path(spectra_manifest: Path, row: dict[str, str]) -> Path:
    raw_path = Path(row["spectrum_path"]).expanduser()
    base_dir = spectra_manifest.parent.resolve()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.append((base_dir / raw_path).resolve())

    parts = raw_path.parts
    if "all_chimera_notebook6_spectra" in parts:
        idx = parts.index("all_chimera_notebook6_spectra")
        rel = Path(*parts[idx + 1 :])
        if rel.parts:
            candidates.append((base_dir / rel).resolve())

    candidates.extend(
        [
            (base_dir / raw_path.name).resolve(),
            (base_dir / "spectra" / raw_path.name).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve notebook-6 spectrum path from spectra manifest. "
        f"Manifest value was {row.get('spectrum_path')!r}. Searched:\n  {searched}"
    )


def _load_notebook6_spectrum(path: Path, *, min_valid_pixels: int) -> tuple[Any, dict[str, Any]]:
    config = importlib.import_module("jaxsedfit.config")
    spectroscopy_data_cls = getattr(config, "SpectroscopyData")
    table = Table.read(path, format="ascii.ecsv")
    required = {"wave_obs", "flux_mjy", "flux_err_mjy", "mask"}
    missing = sorted(required.difference(table.colnames))
    if missing:
        raise RuntimeError(f"Spectrum table is missing columns {missing}: {path}")

    wave = np.asarray(table["wave_obs"], dtype=float)
    flux = np.asarray(table["flux_mjy"], dtype=float)
    err = np.asarray(table["flux_err_mjy"], dtype=float)
    mask = np.asarray(table["mask"], dtype=bool)
    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0.0) & mask
    if int(np.count_nonzero(valid)) < min_valid_pixels:
        raise RuntimeError(
            f"Spectrum has {int(np.count_nonzero(valid))} valid pixels; need at least {min_valid_pixels}: {path}"
        )

    aperture = table.meta.get("aperture_diameter_arcsec")
    try:
        aperture = float(aperture)
        if not np.isfinite(aperture):
            aperture = None
    except (TypeError, ValueError):
        aperture = None

    spec = spectroscopy_data_cls(
        wave_obs=wave.tolist(),
        fluxes=flux.tolist(),
        errors=err.tolist(),
        mask=valid.tolist(),
        instrument=str(table.meta.get("instrument", "ChimeraComposite")),
        aperture_diameter_arcsec=aperture,
    )
    return spec, {
        "spectrum_path": str(path),
        "spectrum_n_pixels": int(len(wave)),
        "spectrum_n_valid_pixels": int(np.count_nonzero(valid)),
        "spectrum_wave_min": float(np.nanmin(wave[valid])),
        "spectrum_wave_max": float(np.nanmax(wave[valid])),
        "spectrum_flux_unit": table.meta.get("flux_unit", "mJy"),
        "spectrum_wave_unit": table.meta.get("wave_unit", "Angstrom"),
    }


def _configure_joint_fit(cfg: Any, args: argparse.Namespace, spectrum: Any) -> None:
    config = importlib.import_module("jaxsedfit.config")
    spectroscopy_config_cls = getattr(config, "SpectroscopyConfig")
    jaxqsofit_config_cls = getattr(config, "JaxQSOFitConfig")

    cfg.spectroscopy = spectrum
    cfg.spectroscopy_config = spectroscopy_config_cls(
        enabled=True,
        backend=args.spectroscopy_backend,
        fit_scale=not args.no_fit_spectrum_scale,
        scale_prior_sigma_dex=args.scale_prior_sigma_dex,
        systematics_width=args.spectrum_systematics_width,
        student_t_df=args.spectrum_student_t_df,
        likelihood_weight_mode=args.spectrum_weight_mode,
        resolving_power=args.resolving_power,
        jaxqsofit=jaxqsofit_config_cls(
            use_spectral_lines=not args.no_spectral_lines,
            use_tied_lines=not args.no_tied_lines,
            use_spectral_smart_priors=not args.no_spectral_smart_priors,
            use_spectral_feii=not args.no_spectral_feii,
            use_spectral_balmer_continuum=not args.no_spectral_balmer_continuum,
            line_flux_scale_mjy=args.line_flux_scale_mjy,
        ),
    )

    cfg.galaxy.n_wave = int(args.n_wave)
    cfg.galaxy.fit_host_kinematics = True
    cfg.galaxy.rest_wave_min = float(args.rest_wave_min)
    cfg.galaxy.rest_wave_max = float(args.rest_wave_max)
    cfg.agn.agn_type = int(args.agn_type)
    cfg.agn.broad_line_width_kms_default = float(args.broad_line_width_kms)
    cfg.agn.narrow_line_width_kms_default = float(args.narrow_line_width_kms)
    cfg.agn.broad_lines_strength_default = float(args.broad_lines_strength)
    cfg.agn.narrow_lines_strength_default = float(args.narrow_lines_strength)
    cfg.agn.feii_strength_default = float(args.feii_strength)
    cfg.agn.fit_balmer_continuum = False
    cfg.agn.balmer_continuum_default = float(args.balmer_continuum)
    cfg.likelihood.use_host_capture_model = True
    cfg.likelihood.systematics_width = float(args.phot_systematics_width)
    cfg.likelihood.fit_intrinsic_scatter = not args.no_fit_intrinsic_scatter

    cfg.inference.method = args.sampler
    cfg.inference.map_steps = int(args.optax_steps)
    cfg.inference.learning_rate = float(args.optax_lr)
    cfg.inference.num_warmup = int(args.nuts_warmup)
    cfg.inference.num_samples = int(args.nuts_samples)
    cfg.inference.num_chains = int(args.nuts_chains)
    cfg.inference.target_accept_prob = float(args.target_accept_prob)
    cfg.inference.max_tree_depth = int(args.max_tree_depth)
    cfg.inference.ns_num_live_points = args.ns_live_points
    cfg.inference.ns_max_samples = args.ns_max_samples
    cfg.inference.ns_dlogz = args.ns_dlogz
    cfg.inference.ns_resamples = args.ns_resamples

    try:
        import numpyro.distributions as dist

        cfg.prior_config.agn.log_broad_line_width_kms = dist.TruncatedNormal(
            loc=np.log(float(args.broad_line_width_kms)),
            scale=0.4,
            low=np.log(1000.0),
            high=np.log(15000.0),
        )
        cfg.prior_config.agn.log_narrow_line_width_kms = dist.TruncatedNormal(
            loc=np.log(float(args.narrow_line_width_kms)),
            scale=0.3,
            low=np.log(100.0),
            high=np.log(1500.0),
        )
    except Exception:
        pass

    if hasattr(cfg, "validate"):
        cfg.validate()


def _sample_percentiles(samples: Any) -> tuple[float, float, float]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
    return float(p16), float(p50), float(p84)


def _run_joint_fit(row: dict[str, Any], args: argparse.Namespace, spectrum_path: Path) -> dict[str, Any]:
    _, build_chimera_fit_config, fitter_cls = manifest_fit._load_backend(args.backend)
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(args.dsps_ssp_fn))
    manifest_fit._patch_backend_config_compat(cfg)
    cfg.inference.seed = int(args.seed_base + row["fit_index"])
    spectrum, spectrum_payload = _load_notebook6_spectrum(
        spectrum_path,
        min_valid_pixels=args.min_valid_spectral_pixels,
    )
    _configure_joint_fit(cfg, args, spectrum)

    fitter = fitter_cls(cfg)
    fit_result = fitter.fit(progress_bar=args.progress_bar)
    if args.save_sed_pdf:
        args.sed_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fitter.plot_sed(output_path=args.sed_pdf_path)
    if args.save_corner_pdf:
        args.corner_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fitter.plot_corner(output_path=args.corner_pdf_path)
    if args.save_trace_pdf:
        args.trace_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fitter.plot_trace(output_path=args.trace_pdf_path)

    logm16, recovered_logm, logm84 = _sample_percentiles(fitter.samples["log_stellar_mass"])
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
        "fit_summary": manifest_fit._fit_result_summary(fit_result),
        "sampler": args.sampler,
        "backend": manifest_fit._normalize_backend(args.backend),
        "fit_method": "jaxsedfit_joint_photometry_spectroscopy",
        "n_wave": int(args.n_wave),
        "spectroscopy_backend": args.spectroscopy_backend,
        "spectrum_weight_mode": args.spectrum_weight_mode,
        "spectrum_systematics_width": float(args.spectrum_systematics_width),
        "phot_systematics_width": float(args.phot_systematics_width),
        "scale_prior_sigma_dex": float(args.scale_prior_sigma_dex),
        **spectrum_payload,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one joint JAXSEDFit photometry+spectroscopy Chimera fit.")
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--spectra-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("hpc_outputs/joint_spectro"))
    parser.add_argument("--dsps-ssp-fn", type=Path, default=Path("tempdata.h5"))
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--expected-count", type=int, default=13558)
    parser.add_argument("--seed-base", type=int, default=20231011)
    parser.add_argument("--backend", choices=tuple(sorted(manifest_fit.BACKEND_ALIASES)), default="jaxsedfit")
    parser.add_argument("--sampler", choices=("optax", "nuts", "optax+nuts", "ns"), default="optax")
    parser.add_argument("--optax-steps", type=int, default=600)
    parser.add_argument("--optax-lr", type=float, default=5.0e-3)
    parser.add_argument("--nuts-warmup", type=int, default=250)
    parser.add_argument("--nuts-samples", type=int, default=250)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--target-accept-prob", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--ns-live-points", type=int, default=None)
    parser.add_argument("--ns-max-samples", type=int, default=None)
    parser.add_argument("--ns-dlogz", type=float, default=None)
    parser.add_argument("--ns-resamples", type=int, default=None)
    parser.add_argument("--n-wave", type=int, default=512)
    parser.add_argument("--rest-wave-min", type=float, default=100.0)
    parser.add_argument("--rest-wave-max", type=float, default=3.0e6)
    parser.add_argument("--spectroscopy-backend", choices=("jaxqsofit", "jaxsedfit"), default="jaxqsofit")
    parser.add_argument("--spectrum-weight-mode", default="resolution_elements")
    parser.add_argument("--resolving-power", type=float, default=2000.0)
    parser.add_argument("--spectrum-systematics-width", type=float, default=0.08)
    parser.add_argument("--spectrum-student-t-df", type=float, default=5.0)
    parser.add_argument("--scale-prior-sigma-dex", type=float, default=0.4)
    parser.add_argument("--phot-systematics-width", type=float, default=0.08)
    parser.add_argument("--line-flux-scale-mjy", type=float, default=0.1)
    parser.add_argument("--min-valid-spectral-pixels", type=int, default=50)
    parser.add_argument("--agn-type", type=int, default=1)
    parser.add_argument("--broad-line-width-kms", type=float, default=3000.0)
    parser.add_argument("--narrow-line-width-kms", type=float, default=500.0)
    parser.add_argument("--broad-lines-strength", type=float, default=1.0)
    parser.add_argument("--narrow-lines-strength", type=float, default=1.0)
    parser.add_argument("--feii-strength", type=float, default=1.0)
    parser.add_argument("--balmer-continuum", type=float, default=0.1)
    parser.add_argument("--no-fit-spectrum-scale", action="store_true")
    parser.add_argument("--no-fit-intrinsic-scatter", action="store_true")
    parser.add_argument("--no-spectral-lines", action="store_true")
    parser.add_argument("--no-tied-lines", action="store_true")
    parser.add_argument("--no-spectral-smart-priors", action="store_true")
    parser.add_argument("--no-spectral-feii", action="store_true")
    parser.add_argument("--no-spectral-balmer-continuum", action="store_true")
    parser.add_argument("--save-sed-pdf", action="store_true")
    parser.add_argument("--save-corner-pdf", action="store_true")
    parser.add_argument("--save-trace-pdf", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.manifest = args.manifest.expanduser().resolve()
    args.spectra_manifest = args.spectra_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dsps_ssp_fn = args.dsps_ssp_fn.expanduser().resolve()
    args.backend = manifest_fit._normalize_backend(args.backend)

    filter_names, _, _ = manifest_fit._load_backend(args.backend)
    raw = manifest_fit._select_manifest_entry(args)
    row = manifest_fit._row_from_manifest(raw, filter_names)
    spectra_by_id = _load_spectra_manifest(args.spectra_manifest)
    if str(row["id"]) not in spectra_by_id:
        raise RuntimeError(f"No notebook-6 spectrum is available for object_id={row['id']!r}.")
    spectrum_path = _resolve_spectrum_path(args.spectra_manifest, spectra_by_id[str(row["id"])])

    stem = f"{row['fit_index']:05d}_COSMOS{manifest_fit._safe_id(str(row['COSMOS_ID0']))}_{manifest_fit._safe_id(str(row['id']))}"
    success_path = args.output_dir / "results" / f"{stem}.json"
    failure_path = args.output_dir / "failures" / f"{stem}.json"
    args.sed_pdf_path = args.output_dir / "sed_pdfs" / f"{stem}.pdf"
    args.corner_pdf_path = args.output_dir / "corner_pdfs" / f"{stem}.pdf"
    args.trace_pdf_path = args.output_dir / "trace_pdfs" / f"{stem}.pdf"

    print(
        f"[joint-fit] fit_index={row['fit_index']} COSMOS_ID0={row['COSMOS_ID0']} "
        f"object_id={row['id']} spectrum={spectrum_path}",
        flush=True,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "fit_index": row["fit_index"],
                    "object_id": row["id"],
                    "spectrum_path": str(spectrum_path),
                },
                indent=2,
            )
        )
        return 0

    try:
        payload = _run_joint_fit(row, args, spectrum_path)
        manifest_fit._atomic_write_json(success_path, payload)
        print(f"[joint-fit] wrote {success_path}", flush=True)
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
            "backend": args.backend,
            "fit_method": "jaxsedfit_joint_photometry_spectroscopy",
            "spectrum_path": str(spectrum_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        manifest_fit._atomic_write_json(failure_path, payload)
        print(f"[joint-fit] failed; wrote {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
