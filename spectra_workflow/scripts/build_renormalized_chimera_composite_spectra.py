#!/usr/bin/env python
"""Build publication-ready Chimera spectra with optional galaxy renormalization.

This is a production version of the optional notebook-18 test:

    composite = galaxy_scale * galaxy_spectrum + qso_weight * qso_spectrum

where ``galaxy_scale`` is measured from overlapping benchmark photometry bands.
The output ECSV files use the columns expected by the joint spectroscopy runner:
``wave_obs``, ``flux_mjy``, ``flux_err_mjy``, and ``mask``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


C_MS = 299792458.0
FILTER_WAVELENGTHS_A = {
    "r_sdss": 6231.0,
    "i_sdss": 7625.0,
    "z_sdss": 9134.0,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=None)
    parser.add_argument("--input-spectra-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--renorm-mode", choices=("galaxy-to-total", "galaxy-to-residual", "none"), default="galaxy-to-total")
    parser.add_argument("--renorm-bands", nargs="+", default=["r_sdss", "i_sdss", "z_sdss"])
    parser.add_argument("--min-renorm-bands", type=int, default=1)
    parser.add_argument("--local-window-a", type=float, default=150.0)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--error-floor-fraction", type=float, default=0.10)
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def flux_lambda_to_mjy(wavelength_a: np.ndarray | float, flux_lambda: np.ndarray | float) -> np.ndarray | float:
    wave = np.asarray(wavelength_a, dtype=float)
    flux = np.asarray(flux_lambda, dtype=float)
    out = flux * 1.0e7 * (wave * 1.0e-10) ** 2 / C_MS / 1.0e-29
    if np.ndim(out) == 0:
        return float(out)
    return out


def load_component_module(project_root: Path):
    candidates = [
        Path(__file__).resolve().with_name("chimera_composite_spectra.py"),
        project_root / "chimera_composite_spectra.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("chimera_composite_spectra", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chimera_composite_spectra"] = module
    spec.loader.exec_module(module)
    return module


def local_flux(wave: np.ndarray, flux: np.ndarray, wavelength: float, *, window_a: float) -> float:
    valid = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    wave_valid = wave[valid]
    flux_valid = flux[valid]
    if not (float(np.nanmin(wave_valid)) <= wavelength <= float(np.nanmax(wave_valid))):
        return float("nan")
    nearby = np.abs(wave_valid - wavelength) <= window_a
    if np.count_nonzero(nearby) >= 3:
        return float(np.nanmedian(flux_valid[nearby]))
    return float(np.interp(wavelength, wave_valid, flux_valid))


def load_fit_rows(path: Path) -> dict[str, dict[str, str]]:
    return {row["object_id"]: row for row in read_csv_rows(path)}


def source_path(row: dict[str, str], key: str) -> Path:
    path = Path(row[key]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path


def prepare_components(module: Any, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    chimera_redshift = parse_float(row["chimera_redshift"])
    dr7q_redshift = parse_float(row["dr7q_redshift"])
    qso_weight = parse_float(row["chimera_qso_weight"])
    cosmos_ebv = parse_float(row.get("cosmos_ebv"))
    if not math.isfinite(cosmos_ebv):
        cosmos_ebv = 0.0

    galaxy = module.shift_flux_density_to_redshift(
        module.load_zcosmos_spectrum(source_path(row, "galaxy_spectrum_path"), chimera_redshift),
        chimera_redshift,
    )
    qso = module.shift_flux_density_to_redshift(
        module.load_sdss_dr7q_spectrum(source_path(row, "qso_spectrum_path"), dr7q_redshift),
        chimera_redshift,
    )
    for spec in (galaxy, qso):
        attenuated, _ = module.apply_foreground_extinction(
            spec["wavelength_target_obs_angstrom"],
            spec["flux_density_target_obs"],
            cosmos_ebv,
            apply=True,
        )
        spec["flux_density_extincted"] = attenuated

    common_wave = module.make_common_grid([galaxy, qso])
    galaxy_flux = module.interp_flux(galaxy, common_wave)
    qso_flux = module.interp_flux(qso, common_wave)
    meta = {
        "chimera_redshift": chimera_redshift,
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "cosmos_ebv": cosmos_ebv,
    }
    return common_wave, galaxy_flux, qso_flux, qso_weight, meta


def estimate_galaxy_scale(
    wave: np.ndarray,
    galaxy_flux: np.ndarray,
    weighted_qso_flux: np.ndarray,
    fit_row: dict[str, str],
    *,
    mode: str,
    bands: list[str],
    window_a: float,
) -> tuple[float, list[dict[str, Any]]]:
    if mode == "none":
        return 1.0, []

    ratios: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for band in bands:
        wavelength = FILTER_WAVELENGTHS_A.get(band)
        if wavelength is None:
            continue
        phot = parse_float(fit_row.get(band))
        galaxy_mjy = flux_lambda_to_mjy(wavelength, local_flux(wave, galaxy_flux, wavelength, window_a=window_a))
        qso_mjy = flux_lambda_to_mjy(wavelength, local_flux(wave, weighted_qso_flux, wavelength, window_a=window_a))
        target = phot
        if mode == "galaxy-to-residual":
            target = phot - qso_mjy
        ratio = target / galaxy_mjy if galaxy_mjy > 0.0 and target > 0.0 else float("nan")
        diagnostics.append(
            {
                "band": band,
                "band_wavelength_a": wavelength,
                "manifest_flux_mjy": phot,
                "galaxy_flux_mjy": galaxy_mjy,
                "weighted_qso_flux_mjy": qso_mjy,
                "target_galaxy_flux_mjy": target,
                "galaxy_scale_band": ratio,
            }
        )
        if math.isfinite(ratio) and ratio > 0.0:
            ratios.append(ratio)

    if not ratios:
        return float("nan"), diagnostics
    return float(10.0 ** np.median(np.log10(ratios))), diagnostics


def build_one(
    module: Any,
    row: dict[str, str],
    fit_row: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    chimera_id = row["chimera_id"]
    wave, galaxy_flux, qso_flux, qso_weight, meta = prepare_components(module, row)
    weighted_qso_flux = qso_weight * qso_flux
    galaxy_scale, scale_diagnostics = estimate_galaxy_scale(
        wave,
        galaxy_flux,
        weighted_qso_flux,
        fit_row,
        mode=args.renorm_mode,
        bands=list(args.renorm_bands),
        window_a=float(args.local_window_a),
    )
    if int(sum(math.isfinite(item.get("galaxy_scale_band", float("nan"))) for item in scale_diagnostics)) < args.min_renorm_bands and args.renorm_mode != "none":
        raise RuntimeError(f"only {len(scale_diagnostics)} renormalization diagnostics; need {args.min_renorm_bands}")
    if not math.isfinite(galaxy_scale):
        raise RuntimeError("could not estimate galaxy scale")
    if not (args.min_scale <= galaxy_scale <= args.max_scale):
        raise RuntimeError(f"galaxy scale {galaxy_scale:.6g} outside allowed range {args.min_scale:g}-{args.max_scale:g}")

    renorm_galaxy_flux = galaxy_scale * galaxy_flux
    composite_flux_lambda = renorm_galaxy_flux + weighted_qso_flux
    flux_mjy = flux_lambda_to_mjy(wave, composite_flux_lambda)
    finite = np.isfinite(wave) & np.isfinite(flux_mjy)
    positive = finite & (flux_mjy > 0.0)
    mask = positive
    if int(np.count_nonzero(mask)) < args.min_valid_pixels:
        raise RuntimeError(f"only {int(np.count_nonzero(mask))} valid positive pixels")

    err_mjy = np.full_like(np.asarray(flux_mjy, dtype=float), np.nan)
    err_mjy[finite] = np.maximum(np.abs(np.asarray(flux_mjy, dtype=float)[finite]) * args.error_floor_fraction, 1.0e-30)

    safe_id = safe_filename(chimera_id)
    spectrum_path = output_dir / "spectra" / f"{safe_id}_renorm_spectrum.ecsv"
    if args.overwrite or not spectrum_path.exists():
        table = Table(
            {
                "wave_obs": wave,
                "flux_mjy": flux_mjy,
                "flux_err_mjy": err_mjy,
                "mask": mask,
            }
        )
        table.meta.update(
            {
                "chimera_id": chimera_id,
                "cosmos_id": row.get("cosmos_id", ""),
                "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
                "chimera_redshift": meta["chimera_redshift"],
                "dr7q_redshift": meta["dr7q_redshift"],
                "chimera_qso_weight": qso_weight,
                "cosmos_ebv": meta["cosmos_ebv"],
                "galaxy_spectrum_path": row["galaxy_spectrum_path"],
                "qso_spectrum_path": row["qso_spectrum_path"],
                "renorm_mode": args.renorm_mode,
                "galaxy_scale": galaxy_scale,
                "error_floor_fraction": args.error_floor_fraction,
                "format": "renormalized Chimera publication workflow SpectroscopyData input",
                "flux_unit": "mJy",
                "wave_unit": "Angstrom",
                "instrument": "ChimeraCompositeRenormalized",
                "aperture_diameter_arcsec": np.nan,
            }
        )
        spectrum_path.parent.mkdir(parents=True, exist_ok=True)
        table.write(spectrum_path, format="ascii.ecsv", overwrite=True)

    payload: dict[str, Any] = {
        "status": "success",
        "action": "created",
        "chimera_id": chimera_id,
        "row_index": row.get("row_index", ""),
        "cosmos_id": row.get("cosmos_id", ""),
        "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
        "chimera_redshift": row.get("chimera_redshift", ""),
        "dr7q_redshift": row.get("dr7q_redshift", ""),
        "chimera_qso_weight": qso_weight,
        "renorm_mode": args.renorm_mode,
        "galaxy_scale": galaxy_scale,
        "spectrum_path": str(spectrum_path),
        "n_pixels": int(len(wave)),
        "n_valid_pixels": int(np.count_nonzero(mask)),
        "wave_min": float(np.nanmin(wave[mask])),
        "wave_max": float(np.nanmax(wave[mask])),
        "galaxy_spectrum_path": row["galaxy_spectrum_path"],
        "qso_spectrum_path": row["qso_spectrum_path"],
    }
    for item in scale_diagnostics:
        band = item["band"]
        payload[f"{band}_scale"] = item["galaxy_scale_band"]
        payload[f"{band}_manifest_mjy"] = item["manifest_flux_mjy"]
        payload[f"{band}_galaxy_mjy"] = item["galaxy_flux_mjy"]
        payload[f"{band}_weighted_qso_mjy"] = item["weighted_qso_flux_mjy"]
    return payload


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    fit_manifest = (args.fit_manifest or project_root / "fit_manifest.csv").expanduser().resolve()
    input_manifest = (
        args.input_spectra_manifest
        or WORKFLOW_ROOT
        / "outputs"
        / "all_chimera_spectra"
        / "chimera_spectra_manifest.csv"
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or WORKFLOW_ROOT / "outputs" / "renormalized_chimera_spectra"
    ).expanduser().resolve()

    print(f"Project root: {project_root}")
    print(f"Fit manifest: {fit_manifest}")
    print(f"Input spectra manifest: {input_manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Renormalization mode: {args.renorm_mode}")

    module = load_component_module(project_root)
    fit_rows = load_fit_rows(fit_manifest)
    input_rows = [row for row in read_csv_rows(input_manifest) if row.get("status", "success") == "success"]
    if args.max_rows is not None:
        input_rows = input_rows[: args.max_rows]

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(input_rows, start=1):
        chimera_id = row.get("chimera_id", "")
        try:
            fit_row = fit_rows.get(chimera_id)
            if fit_row is None:
                raise KeyError(f"no fit_manifest row for {chimera_id!r}")
            successes.append(build_one(module, row, fit_row, args, output_dir))
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "chimera_id": chimera_id,
                    "row_index": row.get("row_index", ""),
                    "cosmos_id": row.get("cosmos_id", ""),
                    "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(input_rows)}; successes={len(successes)} failures={len(failures)}")

    manifest_path = output_dir / "renormalized_chimera_spectra_manifest.csv"
    failures_path = output_dir / "renormalized_chimera_spectra_failures.csv"
    write_csv(manifest_path, successes)
    write_csv(failures_path, failures)
    print(f"Done. successes={len(successes)} failures={len(failures)}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote failures: {failures_path}")
    print(f"Spectra are under: {output_dir / 'spectra'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
