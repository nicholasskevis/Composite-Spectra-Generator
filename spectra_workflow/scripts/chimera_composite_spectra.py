#!/usr/bin/env python
"""Build a Chimera composite spectrum from local galaxy and DR7Q spectra.

This script is the command-line version of notebook 18. It reads
``chimera_provenance.csv``, finds the matching zCOSMOS/CESAM-style galaxy
spectrum and DR7Q spectrum, standardizes wavelength/flux-density units, shifts
the QSO spectrum to the best galaxy spectroscopic redshift, matches the
higher-resolution component down to the lower resolving power, optionally
applies COSMOS foreground extinction, and writes observed-frame/rest-frame
tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


DEFAULT_CHIMERA_ID = "165523.09+184708.4_701398_0.01"
FLUX_DENSITY_SCALE = 1e17
FLUX_DENSITY_DISPLAY_UNIT = r"$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ Angstrom$^{-1}$"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DATA_DIR = Path("/home/nicho/GRAHSP_my/data")
ZCOSMOS_ERROR_COLUMNS = (
    "ERR",
    "ERROR",
    "FLUX_ERR",
    "FLUX_ERROR",
    "SIGMA",
    "NOISE",
)
GALAXY_REDSHIFT_COLUMNS = (
    "galaxy_spectroscopic_redshift",
    "redshift_GAL",
    "zspec_GAL",
    "z_spec_GAL",
    "zCOSMOS_redshift",
    "z_zcosmos",
    "chimera_redshift",
    "redshift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Chimera galaxy + weighted-QSO composite spectrum."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository/project root. Defaults to the inferred repo root.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing dr7q_spectra, zCOSMOS_data, and COSMOS2015. "
            "Default: /home/nicho/GRAHSP_my/data if present, otherwise project-root/data."
        ),
    )
    parser.add_argument(
        "--chimera-id",
        default=DEFAULT_CHIMERA_ID,
        help=(
            "Chimera ID to process. The default is a local row with both zCOSMOS "
            "and DR7Q spectra available."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to spectra_workflow/outputs/18_chimera_composite_spectra.",
    )
    parser.add_argument(
        "--no-extinction",
        action="store_true",
        help="Skip COSMOS foreground attenuation.",
    )
    parser.add_argument(
        "--no-rest",
        action="store_true",
        help="Skip rest-frame table and plot.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables only; skip PNG plots.",
    )
    parser.add_argument(
        "--error-floor-fraction",
        type=float,
        default=0.10,
        help=(
            "Minimum 1-sigma error as a fraction of the absolute composite flux. "
            "The written error is max(propagated component error, this floor)."
        ),
    )
    parser.add_argument(
        "--resampling-method",
        choices=("flux-conserving", "interp"),
        default="flux-conserving",
        help="How to put components on the common grid after resolution matching. Default: flux-conserving.",
    )
    parser.add_argument(
        "--no-resolution-match",
        action="store_true",
        help="Disable Gaussian LSF matching before combining galaxy and QSO spectra.",
    )
    parser.add_argument(
        "--galaxy-resolving-power",
        type=float,
        default=600.0,
        help="Assumed zCOSMOS/CESAM resolving power R=lambda/dlambda. Default: 600.",
    )
    parser.add_argument(
        "--qso-resolving-power",
        type=float,
        default=2000.0,
        help="Assumed SDSS DR7Q resolving power R=lambda/dlambda. Default: 2000.",
    )
    parser.add_argument(
        "--resolution-kernel-sigma-width",
        type=float,
        default=4.0,
        help="Gaussian convolution half-width in sigma units. Default: 4.",
    )
    return parser.parse_args()


def clean_fits_text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)


def load_provenance(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_provenance(rows: list[dict[str, str]], chimera_id: str) -> dict[str, str]:
    matches = [row for row in rows if row["chimera_id"] == chimera_id]
    if not matches:
        raise ValueError(f"Could not find chimera_id={chimera_id!r}")
    return matches[0]


def find_cosmos_row(path: Path, object_id: int) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        matches = np.flatnonzero(np.asarray(data["NUMBER"]) == object_id)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one COSMOS2015 row for NUMBER={object_id}, found {len(matches)}"
            )
        row = data[matches[0]]
        return {name: row[name] for name in data.names}


def parse_zcosmos_readme_mappings(directory: Path) -> dict[int, list[Path]]:
    mapping: dict[int, list[Path]] = {}
    for readme in directory.glob("readme*.txt"):
        text = readme.read_text(errors="ignore")
        for archive_name, original_name in re.findall(
            r"-\s+(ADP[^\s]+\.fits)\s+(zCOSMOS[^\s]+\.fits)", text
        ):
            match = re.search(r"_(\d{6,9})_", original_name)
            if not match:
                continue
            object_id = int(match.group(1))
            archive_path = directory / archive_name.replace(":", "_")
            if archive_path.exists():
                mapping.setdefault(object_id, []).append(archive_path)
    return mapping


def build_galaxy_spectrum_index(directories: list[Path]) -> tuple[dict[int, list[Path]], list[str]]:
    index: dict[int, list[Path]] = {}
    diagnostics: list[str] = []

    for directory in directories:
        if not directory.exists():
            diagnostics.append(f"missing optional directory: {directory}")
            continue

        for object_id, paths in parse_zcosmos_readme_mappings(directory).items():
            index.setdefault(object_id, []).extend(paths)

        for path in sorted(directory.glob("*.fits")):
            if path.name.endswith(":Zone.Identifier"):
                continue
            try:
                with fits.open(path, memmap=True) as hdul:
                    headers = [hdul[0].header]
                    if len(hdul) > 1:
                        headers.append(hdul[1].header)
                    ids: list[int] = []
                    for header in headers:
                        for key in ("OBJECT", "ORIGFILE", "TITLE"):
                            if key in header:
                                ids.extend(
                                    int(value)
                                    for value in re.findall(
                                        r"\b\d{6,9}\b", clean_fits_text(header[key])
                                    )
                                )
                    for object_id in ids:
                        index.setdefault(object_id, []).append(path)
            except Exception as exc:  # pragma: no cover - diagnostic path
                diagnostics.append(f"could not inspect {path}: {exc}")

    deduped: dict[int, list[Path]] = {}
    for object_id, paths in index.items():
        seen: list[Path] = []
        for path in paths:
            if path not in seen:
                seen.append(path)
        deduped[object_id] = seen
    return deduped, diagnostics


def find_galaxy_spectra(directories: list[Path], cosmos_id: int) -> tuple[list[Path], list[str]]:
    """Find spectra for one COSMOS ID without building a full archive index."""
    diagnostics: list[str] = []
    matches: list[Path] = []

    for directory in directories:
        if not directory.exists():
            diagnostics.append(f"missing optional directory: {directory}")
            continue

        readme_matches = parse_zcosmos_readme_mappings(directory).get(cosmos_id, [])
        for path in readme_matches:
            if path not in matches:
                matches.append(path)
        if matches:
            return matches, diagnostics

        for path in sorted(directory.glob("*.fits")):
            if path.name.endswith(":Zone.Identifier"):
                continue
            try:
                with fits.open(path, memmap=True) as hdul:
                    headers = [hdul[0].header]
                    if len(hdul) > 1:
                        headers.append(hdul[1].header)
                    found = False
                    for header in headers:
                        for key in ("OBJECT", "ORIGFILE", "TITLE"):
                            if key not in header:
                                continue
                            ids = {
                                int(value)
                                for value in re.findall(
                                    r"\b\d{6,9}\b", clean_fits_text(header[key])
                                )
                            }
                            if cosmos_id in ids:
                                found = True
                                break
                        if found:
                            break
                    if found and path not in matches:
                        matches.append(path)
                        return matches, diagnostics
            except Exception as exc:  # pragma: no cover - diagnostic path
                diagnostics.append(f"could not inspect {path}: {exc}")

    return matches, diagnostics


def find_dr7q_spectrum(directory: Path, plate: int, mjd: int, fiber: int) -> Path:
    candidates = [
        directory / f"spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
        directory / f"spec-{plate}-{mjd}-{fiber}.fits",
        directory / f"spec-{plate:04d}-{mjd}-{fiber:03d}.fits",
    ]
    for path in candidates:
        if path.exists():
            return path

    globbed = sorted(directory.glob(f"spec-{plate:04d}-{mjd}-*.fits"))
    globbed += sorted(directory.glob(f"spec-{plate}-{mjd}-*.fits"))
    fiber_tokens = {f"-{fiber:04d}.fits", f"-{fiber:03d}.fits", f"-{fiber}.fits"}
    for path in globbed:
        if any(path.name.endswith(token) for token in fiber_tokens):
            return path
    raise FileNotFoundError(f"Could not find DR7Q spectrum for {plate}-{mjd}-{fiber}")


def finite_sorted_spectrum(wavelength, flux_density, error=None):
    wavelength = np.asarray(wavelength, dtype=float)
    flux_density = np.asarray(flux_density, dtype=float)
    if error is None:
        error = np.full_like(flux_density, np.nan, dtype=float)
    else:
        error = np.asarray(error, dtype=float)

    valid = np.isfinite(wavelength) & np.isfinite(flux_density) & (wavelength > 0)
    wavelength = wavelength[valid]
    flux_density = flux_density[valid]
    error = error[valid]
    order = np.argsort(wavelength)
    return wavelength[order], flux_density[order], error[order]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def best_galaxy_redshift(row: dict[str, object], fallback: float) -> tuple[float, str]:
    for key in GALAXY_REDSHIFT_COLUMNS:
        if key not in row:
            continue
        z = finite_float(row.get(key))
        if np.isfinite(z) and z > -1.0:
            return z, key
    return fallback, "chimera_redshift"


def load_zcosmos_spectrum(path: Path, chimera_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        row = hdul[1].data[0]
        names = set(hdul[1].data.names)
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX_REDUCED"], dtype=float)
        err_column = next((name for name in ZCOSMOS_ERROR_COLUMNS if name in names), None)
        err = np.asarray(row[err_column], dtype=float) if err_column is not None else None
        header = dict(hdul[0].header)
        ext_header = dict(hdul[1].header)
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "galaxy",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
        "flux_density_err_column": err_column or "",
        "source_redshift": chimera_redshift,
        "unit": "erg cm^-2 s^-1 Angstrom^-1",
        "header": header,
        "ext_header": ext_header,
    }


def load_sdss_dr7q_spectrum(path: Path, dr7q_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        coadd = hdul[1].data
        names = set(coadd.names)
        if "loglam" in names:
            wave = 10.0 ** np.asarray(coadd["loglam"], dtype=float)
            wavelength_note = "converted SDSS loglam to Angstrom"
        elif "lambda" in names:
            wave = np.asarray(coadd["lambda"], dtype=float)
            wavelength_note = "read SDSS wavelength column directly"
        else:
            raise KeyError(f"No wavelength/loglam column found in {path}")

        flux = np.asarray(coadd["flux"], dtype=float) * 1e-17
        if "ivar" in names:
            ivar = np.asarray(coadd["ivar"], dtype=float)
            err = np.full_like(flux, np.nan, dtype=float)
            good_ivar = ivar > 0
            err[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar]) * 1e-17
        else:
            err = None
        header = dict(hdul[0].header)
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "qso",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
        "source_redshift": dr7q_redshift,
        "unit": "erg cm^-2 s^-1 Angstrom^-1",
        "wavelength_note": wavelength_note,
        "header": header,
    }


def shift_flux_density_to_redshift(spec: dict[str, object], target_redshift: float) -> dict[str, object]:
    z_source = float(spec["source_redshift"])
    wave_obs = np.asarray(spec["wavelength_obs_angstrom"], dtype=float)
    flux_obs = np.asarray(spec["flux_density_obs"], dtype=float)
    err_obs = np.asarray(spec["flux_density_err_obs"], dtype=float)

    wave_rest = wave_obs / (1.0 + z_source)
    flux_rest = flux_obs * (1.0 + z_source)
    err_rest = err_obs * (1.0 + z_source)

    wave_target = wave_rest * (1.0 + target_redshift)
    flux_target = flux_rest / (1.0 + target_redshift)
    err_target = err_rest / (1.0 + target_redshift)

    out = dict(spec)
    out.update(
        {
            "wavelength_rest_angstrom": wave_rest,
            "flux_density_rest": flux_rest,
            "flux_density_err_rest": err_rest,
            "wavelength_target_obs_angstrom": wave_target,
            "flux_density_target_obs": flux_target,
            "flux_density_err_target_obs": err_target,
            "target_redshift": target_redshift,
        }
    )
    return out


def convolve_to_lower_resolution(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    source_r: float,
    target_r: float,
    *,
    nsigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    if (
        not np.isfinite(source_r)
        or not np.isfinite(target_r)
        or source_r <= 0.0
        or target_r <= 0.0
        or source_r <= target_r
    ):
        return flux, err, False

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_source = wave / source_r * fwhm_to_sigma
    sigma_target = wave / target_r * fwhm_to_sigma
    sigma_kernel = np.sqrt(np.maximum(sigma_target**2 - sigma_source**2, 0.0))
    if not np.any(np.isfinite(sigma_kernel) & (sigma_kernel > 0.0)):
        return flux, err, False

    out_flux = np.full_like(flux, np.nan, dtype=float)
    out_err = np.full_like(err, np.nan, dtype=float)
    finite_wave = np.isfinite(wave)
    for i, center in enumerate(wave):
        sigma = sigma_kernel[i]
        if not np.isfinite(center) or not np.isfinite(sigma) or sigma <= 0.0:
            out_flux[i] = flux[i]
            out_err[i] = err[i]
            continue
        window = finite_wave & (np.abs(wave - center) <= abs(float(nsigma)) * sigma)
        valid_flux = window & np.isfinite(flux)
        if np.count_nonzero(valid_flux) < 2:
            out_flux[i] = flux[i]
            out_err[i] = err[i]
            continue
        weights = np.exp(-0.5 * ((wave[valid_flux] - center) / sigma) ** 2)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            out_flux[i] = flux[i]
            out_err[i] = err[i]
            continue
        norm_weights = weights / weight_sum
        out_flux[i] = float(np.sum(norm_weights * flux[valid_flux]))
        err_values = err[valid_flux]
        valid_err = np.isfinite(err_values) & (err_values > 0.0)
        if np.any(valid_err):
            out_err[i] = float(np.sqrt(np.sum((norm_weights[valid_err] * err_values[valid_err]) ** 2)))
    return out_flux, out_err, True


def match_spectral_resolution(
    galaxy_spec: dict[str, object],
    qso_spec: dict[str, object],
    *,
    galaxy_r: float,
    qso_r: float,
    enabled: bool,
    nsigma: float,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    galaxy = dict(galaxy_spec)
    qso = dict(qso_spec)
    target_r = min(float(galaxy_r), float(qso_r))
    metadata: dict[str, object] = {
        "resolution_match_enabled": bool(enabled),
        "resolution_match_method": "Gaussian LSF degradation in observed frame",
        "galaxy_resolving_power_assumed": float(galaxy_r),
        "qso_resolving_power_assumed": float(qso_r),
        "target_resolving_power": target_r,
        "resolution_kernel_sigma_width": float(nsigma),
        "galaxy_resolution_convolved": False,
        "qso_resolution_convolved": False,
        "resolution_match_note": "Only degrades higher-resolution spectra; never deconvolves lower-resolution spectra.",
    }
    if not enabled:
        metadata["target_resolving_power"] = ""
        metadata["resolution_match_method"] = "none"
        return galaxy, qso, metadata

    for spec, source_r, key in (
        (galaxy, float(galaxy_r), "galaxy_resolution_convolved"),
        (qso, float(qso_r), "qso_resolution_convolved"),
    ):
        flux, err, changed = convolve_to_lower_resolution(
            np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float),
            np.asarray(spec["flux_density_target_obs"], dtype=float),
            np.asarray(spec["flux_density_err_target_obs"], dtype=float),
            source_r,
            target_r,
            nsigma=nsigma,
        )
        spec["flux_density_target_obs"] = flux
        spec["flux_density_err_target_obs"] = err
        metadata[key] = changed
    return galaxy, qso, metadata


def ccm89_a_lambda_over_av(wavelength_angstrom, rv=3.1):
    wave_micron = np.asarray(wavelength_angstrom, dtype=float) / 1e4
    x = 1.0 / wave_micron
    a = np.zeros_like(x, dtype=float)
    b = np.zeros_like(x, dtype=float)

    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    y = x[opt] - 1.82
    a[opt] = (
        1
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[opt] = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    uv = (x >= 3.3) & (x <= 8.0)
    fa = np.zeros_like(x[uv])
    fb = np.zeros_like(x[uv])
    high = x[uv] >= 5.9
    fa[high] = -0.04473 * (x[uv][high] - 5.9) ** 2 - 0.009779 * (
        x[uv][high] - 5.9
    ) ** 3
    fb[high] = 0.2130 * (x[uv][high] - 5.9) ** 2 + 0.1207 * (
        x[uv][high] - 5.9
    ) ** 3
    a[uv] = 1.752 - 0.316 * x[uv] - 0.104 / ((x[uv] - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * x[uv] + 1.206 / ((x[uv] - 4.62) ** 2 + 0.263) + fb

    outside = ~(ir | opt | uv)
    if np.any(outside):
        warnings.warn("Some wavelengths are outside the CCM89 range; extinction set to NaN.")
        a[outside] = np.nan
        b[outside] = np.nan
    return a + b / rv


def apply_foreground_extinction(wavelength_angstrom, flux_density, ebv, rv=3.1, apply=True):
    if not apply:
        return np.asarray(flux_density, dtype=float), np.zeros_like(np.asarray(flux_density))
    a_lambda = ccm89_a_lambda_over_av(wavelength_angstrom, rv=rv) * rv * ebv
    attenuation = 10.0 ** (-0.4 * a_lambda)
    return np.asarray(flux_density, dtype=float) * attenuation, a_lambda


def make_common_grid(specs: list[dict[str, object]]) -> np.ndarray:
    starts = [np.nanmin(spec["wavelength_target_obs_angstrom"]) for spec in specs]
    stops = [np.nanmax(spec["wavelength_target_obs_angstrom"]) for spec in specs]
    start = max(starts)
    stop = min(stops)
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("The galaxy and QSO spectra do not overlap in target observed frame.")

    step = max(
        np.nanmedian(np.diff(np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float)))
        for spec in specs
    )
    n = int(np.floor((stop - start) / step)) + 1
    return start + step * np.arange(n)


def galaxy_observed_grid(galaxy_spec: dict[str, object], qso_spec: dict[str, object]) -> np.ndarray:
    """Use the native galaxy observed-frame grid, clipped to QSO overlap."""
    galaxy_wave = np.asarray(galaxy_spec["wavelength_target_obs_angstrom"], dtype=float)
    qso_wave = np.asarray(qso_spec["wavelength_target_obs_angstrom"], dtype=float)
    valid = np.isfinite(galaxy_wave) & (galaxy_wave > 0.0)
    if not np.any(valid):
        raise ValueError("The galaxy spectrum has no valid wavelength pixels.")
    qso_valid = np.isfinite(qso_wave) & (qso_wave > 0.0)
    if not np.any(qso_valid):
        raise ValueError("The QSO spectrum has no valid target-frame wavelength pixels.")
    start = float(np.nanmin(qso_wave[qso_valid]))
    stop = float(np.nanmax(qso_wave[qso_valid]))
    overlap = valid & (galaxy_wave >= start) & (galaxy_wave <= stop)
    if np.count_nonzero(overlap) < 2:
        raise ValueError("The galaxy and QSO spectra do not overlap on the galaxy observed grid.")
    return galaxy_wave[overlap]


def interp_flux(spec: dict[str, object], grid: np.ndarray, flux_key="flux_density_extincted"):
    wave = np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float)
    flux = np.asarray(spec[flux_key], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux)
    return np.interp(grid, wave[valid], flux[valid], left=np.nan, right=np.nan)


def interp_positive_error(spec: dict[str, object], grid: np.ndarray, err_key="flux_density_err_extincted"):
    wave = np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float)
    err = np.asarray(spec[err_key], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(err) & (err > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, wave[valid], err[valid], left=np.nan, right=np.nan)


def pixel_edges_from_centers(wavelength: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    if len(wavelength) < 2:
        raise ValueError("Need at least two wavelength centers to define pixel edges.")
    edges = np.empty(len(wavelength) + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return edges


def flux_conserving_rebin(
    source_wave: np.ndarray,
    source_flux_density: np.ndarray,
    target_wave: np.ndarray,
    source_err_density: np.ndarray | None = None,
    *,
    min_coverage: float = 0.999,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    source_wave = np.asarray(source_wave, dtype=float)
    source_flux_density = np.asarray(source_flux_density, dtype=float)
    target_wave = np.asarray(target_wave, dtype=float)
    if source_err_density is not None:
        source_err_density = np.asarray(source_err_density, dtype=float)

    source_valid = np.isfinite(source_wave) & np.isfinite(source_flux_density)
    source_wave = source_wave[source_valid]
    source_flux_density = source_flux_density[source_valid]
    if source_err_density is not None:
        source_err_density = source_err_density[source_valid]

    target_valid = np.isfinite(target_wave)
    out_flux = np.full_like(target_wave, np.nan, dtype=float)
    out_err = np.full_like(target_wave, np.nan, dtype=float) if source_err_density is not None else None
    coverage = np.zeros_like(target_wave, dtype=float)
    if np.count_nonzero(source_valid) < 2 or np.count_nonzero(target_valid) < 2:
        return out_flux, out_err, coverage

    source_edges = pixel_edges_from_centers(source_wave)
    target_edges = pixel_edges_from_centers(target_wave[target_valid])
    target_indices = np.flatnonzero(target_valid)
    source_left = source_edges[:-1]
    source_right = source_edges[1:]

    for local_i, global_i in enumerate(target_indices):
        left = target_edges[local_i]
        right = target_edges[local_i + 1]
        width = right - left
        if not np.isfinite(width) or width <= 0.0:
            continue
        overlap = np.maximum(0.0, np.minimum(source_right, right) - np.maximum(source_left, left))
        used = overlap > 0.0
        covered = float(np.sum(overlap[used]))
        coverage[global_i] = covered / width
        if coverage[global_i] < min_coverage:
            continue
        out_flux[global_i] = float(np.sum(source_flux_density[used] * overlap[used]) / width)
        if out_err is not None and source_err_density is not None:
            err_used = used & np.isfinite(source_err_density) & (source_err_density > 0.0)
            if np.any(err_used):
                out_err[global_i] = float(np.sqrt(np.sum((source_err_density[err_used] * overlap[err_used]) ** 2)) / width)
    return out_flux, out_err, coverage


def resample_flux_density(
    spec: dict[str, object],
    grid: np.ndarray,
    flux_key="flux_density_extincted",
    err_key="flux_density_err_extincted",
    *,
    method: str = "flux-conserving",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "interp":
        flux = interp_flux(spec, grid, flux_key=flux_key)
        err = interp_positive_error(spec, grid, err_key=err_key)
        coverage = np.where(np.isfinite(flux), 1.0, 0.0)
        return flux, err, coverage
    if method != "flux-conserving":
        raise ValueError(f"Unknown resampling method: {method}")
    return flux_conserving_rebin(
        np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float),
        np.asarray(spec[flux_key], dtype=float),
        grid,
        np.asarray(spec[err_key], dtype=float),
    )


def apply_error_floor(error, flux, fraction):
    error = np.asarray(error, dtype=float)
    flux = np.asarray(flux, dtype=float)
    floor = np.maximum(abs(float(fraction)) * np.abs(flux), 1.0e-300)
    return np.where(np.isfinite(error) & (error > 0.0), np.maximum(error, floor), floor)


def bin_widths(wavelength: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    if len(wavelength) < 2:
        return np.ones_like(wavelength)
    return np.diff(pixel_edges_from_centers(wavelength))


def save_plots(output_dir: Path, safe_id: str, chimera_id: str, arrays: dict[str, np.ndarray], qso_weight: float):
    import matplotlib.pyplot as plt

    common_wave = arrays["common_wave"]
    rest_wave = arrays.get("rest_wave")

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(common_wave, arrays["galaxy_bin_flux"], linewidth=1.0, label="COSMOS/zCOSMOS galaxy")
    ax.plot(common_wave, arrays["weighted_qso_bin_flux"], linewidth=1.0, label=f"DR7Q QSO x {qso_weight:g}")
    ax.plot(common_wave, arrays["composite_bin_flux"], linewidth=1.4, color="black", label="Composite")
    ax.set_xlabel("Observed wavelength at Chimera redshift (Angstrom)")
    ax.set_ylabel(r"Bin flux (erg cm$^{-2}$ s$^{-1}$)")
    ax.set_title(f"{chimera_id} observed-frame bin flux")
    ax.legend()
    path = output_dir / f"{safe_id}_composite_bin_flux.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(common_wave, arrays["galaxy_flux_density_1e17"], linewidth=1.0, label="COSMOS/zCOSMOS galaxy")
    ax.plot(
        common_wave,
        arrays["weighted_qso_flux_density_1e17"],
        linewidth=1.0,
        label=f"DR7Q QSO x {qso_weight:g}",
    )
    ax.plot(common_wave, arrays["composite_flux_density_1e17"], linewidth=1.4, color="black", label="Composite")
    ax.set_xlabel("Observed wavelength at Chimera redshift (Angstrom)")
    ax.set_ylabel(r"$F_\lambda$ (" + FLUX_DENSITY_DISPLAY_UNIT + ")")
    ax.set_title(chimera_id)
    ax.legend()
    path = output_dir / f"{safe_id}_composite_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    if rest_wave is None:
        return

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(rest_wave, arrays["galaxy_rest_bin_flux"], linewidth=1.0, label="Galaxy")
    ax.plot(rest_wave, arrays["weighted_qso_rest_bin_flux"], linewidth=1.0, label="Weighted QSO")
    ax.plot(rest_wave, arrays["composite_rest_bin_flux"], linewidth=1.4, color="black", label="Composite")
    ax.set_xlabel("Rest wavelength (Angstrom)")
    ax.set_ylabel(r"Rest-frame bin flux (erg cm$^{-2}$ s$^{-1}$)")
    ax.set_title(f"{chimera_id} rest-frame bin flux")
    ax.legend()
    path = output_dir / f"{safe_id}_composite_bin_flux_rest_frame.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(rest_wave, arrays["galaxy_rest_flux_density_1e17"], linewidth=1.0, label="Galaxy")
    ax.plot(rest_wave, arrays["weighted_qso_rest_flux_density_1e17"], linewidth=1.0, label="Weighted QSO")
    ax.plot(rest_wave, arrays["composite_rest_flux_density_1e17"], linewidth=1.4, color="black", label="Composite")
    ax.set_xlabel("Rest wavelength (Angstrom)")
    ax.set_ylabel(r"Rest-frame $F_\lambda$ (" + FLUX_DENSITY_DISPLAY_UNIT + ")")
    ax.set_title(f"{chimera_id} rest-frame composite")
    ax.legend()
    path = output_dir / f"{safe_id}_composite_spectrum_rest_frame.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).resolve()
    output_dir = args.output_dir or WORKFLOW_ROOT / "outputs" / "18_chimera_composite_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    (output_dir / "mplconfig").mkdir(parents=True, exist_ok=True)

    provenance = select_provenance(
        load_provenance(project_root / "chimera_provenance.csv"), args.chimera_id
    )
    chimera_id = provenance["chimera_id"]
    cosmos_id = int(provenance["cosmos_id"])
    dr7q_plate = int(provenance["dr7q_plate"])
    dr7q_mjd = int(provenance["dr7q_mjd"])
    dr7q_fiber = int(provenance["dr7q_fiber"])
    chimera_redshift = float(provenance["chimera_redshift"])
    target_redshift, target_redshift_source = best_galaxy_redshift(provenance, chimera_redshift)
    dr7q_redshift = float(provenance["dr7q_redshift"])
    qso_weight = float(provenance["chimera_qso_weight"])
    safe_id = safe_filename(chimera_id)

    cosmos_row = find_cosmos_row(data_dir / "COSMOS2015_Laigle+_v1.1.fits", cosmos_id)
    cosmos_ebv = float(cosmos_row.get("EBV", 0.0))

    galaxy_dirs = [
        data_dir / "zCOSMOS_data",
        data_dir / "zCOSMOS selected",
        data_dir / "cesam_vudz",
        data_dir / "cesam_vuds",
    ]
    galaxy_paths, diagnostics = find_galaxy_spectra(galaxy_dirs, cosmos_id)
    if not galaxy_paths:
        raise FileNotFoundError(f"Could not find galaxy spectrum for COSMOS ID {cosmos_id}")
    galaxy_spectrum_path = galaxy_paths[0]
    dr7q_spectrum_path = find_dr7q_spectrum(
        data_dir / "dr7q_spectra", dr7q_plate, dr7q_mjd, dr7q_fiber
    )

    galaxy_spec = load_zcosmos_spectrum(galaxy_spectrum_path, target_redshift)
    qso_spec = load_sdss_dr7q_spectrum(dr7q_spectrum_path, dr7q_redshift)
    galaxy_shifted = shift_flux_density_to_redshift(galaxy_spec, target_redshift)
    qso_shifted = shift_flux_density_to_redshift(qso_spec, target_redshift)
    galaxy_shifted, qso_shifted, resolution_metadata = match_spectral_resolution(
        galaxy_shifted,
        qso_shifted,
        galaxy_r=args.galaxy_resolving_power,
        qso_r=args.qso_resolving_power,
        enabled=not args.no_resolution_match,
        nsigma=args.resolution_kernel_sigma_width,
    )

    apply_extinction = not args.no_extinction
    for spec in [galaxy_shifted, qso_shifted]:
        attenuated, a_lambda = apply_foreground_extinction(
            spec["wavelength_target_obs_angstrom"],
            spec["flux_density_target_obs"],
            cosmos_ebv,
            apply=apply_extinction,
        )
        spec["flux_density_extincted"] = attenuated
        spec["a_lambda_cosmos_mag"] = a_lambda
        err_attenuated, _ = apply_foreground_extinction(
            spec["wavelength_target_obs_angstrom"],
            spec["flux_density_err_target_obs"],
            cosmos_ebv,
            apply=apply_extinction,
        )
        spec["flux_density_err_extincted"] = err_attenuated

    common_wave = galaxy_observed_grid(galaxy_shifted, qso_shifted)
    galaxy_grid_flux, galaxy_grid_err, galaxy_coverage = resample_flux_density(
        galaxy_shifted,
        common_wave,
        method=args.resampling_method,
    )
    qso_grid_flux, qso_grid_err, qso_coverage = resample_flux_density(
        qso_shifted,
        common_wave,
        method=args.resampling_method,
    )
    weighted_qso_grid_flux = qso_weight * qso_grid_flux
    composite_flux_density = galaxy_grid_flux + weighted_qso_grid_flux
    weighted_qso_grid_err = abs(qso_weight) * qso_grid_err
    composite_flux_density_err_raw = np.sqrt(galaxy_grid_err**2 + weighted_qso_grid_err**2)
    composite_flux_density_err = apply_error_floor(
        composite_flux_density_err_raw,
        composite_flux_density,
        args.error_floor_fraction,
    )

    delta_lambda = bin_widths(common_wave)
    galaxy_bin_flux = galaxy_grid_flux * delta_lambda
    weighted_qso_bin_flux = weighted_qso_grid_flux * delta_lambda
    composite_bin_flux = composite_flux_density * delta_lambda
    composite_bin_flux_err = composite_flux_density_err * delta_lambda

    metadata = {
        "chimera_id": chimera_id,
        "cosmos_id": cosmos_id,
        "dr7q_spectrum_id": provenance["dr7q_spectrum_id"],
        "chimera_redshift": chimera_redshift,
        "target_redshift": target_redshift,
        "target_redshift_source": target_redshift_source,
        "galaxy_spectroscopic_redshift": target_redshift,
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "cosmos_ebv": cosmos_ebv,
        "galaxy_spectrum_path": str(galaxy_spectrum_path.relative_to(project_root)),
        "qso_spectrum_path": str(dr7q_spectrum_path.relative_to(project_root)),
        "galaxy_error_column": galaxy_spec.get("flux_density_err_column", ""),
        "qso_error_source": "ivar" if np.any(np.isfinite(qso_spec["flux_density_err_obs"])) else "",
        "extinction_curve": "CCM89 Rv=3.1" if apply_extinction else "none",
        "error_floor_fraction": args.error_floor_fraction,
        "error_propagation": "sqrt(zCOSMOS_error^2 + (chimera_qso_weight * SDSS_DR7Q_error)^2), then error floor",
        "wavelength_grid": "native zCOSMOS galaxy observed-frame grid clipped to QSO overlap",
        "resampling_method": args.resampling_method,
        "resampling_note": "Flux-conserving mode rebins flux density through pixel-edge overlap integrals after resolution matching.",
        "galaxy_rebin_min_coverage": float(np.nanmin(galaxy_coverage[np.isfinite(composite_flux_density)])) if np.any(np.isfinite(composite_flux_density)) else np.nan,
        "qso_rebin_min_coverage": float(np.nanmin(qso_coverage[np.isfinite(composite_flux_density)])) if np.any(np.isfinite(composite_flux_density)) else np.nan,
    }
    metadata.update(resolution_metadata)

    composite_table = Table(
        {
            "wavelength_obs_angstrom": common_wave,
            "delta_lambda_angstrom": delta_lambda,
            "galaxy_flux_density_erg_cm2_s_A": galaxy_grid_flux,
            "qso_flux_density_erg_cm2_s_A": qso_grid_flux,
            "weighted_qso_flux_density_erg_cm2_s_A": weighted_qso_grid_flux,
            "composite_flux_density_erg_cm2_s_A": composite_flux_density,
            "galaxy_flux_density_err_erg_cm2_s_A": galaxy_grid_err,
            "qso_flux_density_err_erg_cm2_s_A": qso_grid_err,
            "weighted_qso_flux_density_err_erg_cm2_s_A": weighted_qso_grid_err,
            "composite_flux_density_err_raw_erg_cm2_s_A": composite_flux_density_err_raw,
            "composite_flux_density_err_erg_cm2_s_A": composite_flux_density_err,
            "galaxy_rebin_coverage": galaxy_coverage,
            "qso_rebin_coverage": qso_coverage,
            "galaxy_flux_density_1e-17_erg_cm2_s_A": galaxy_grid_flux * FLUX_DENSITY_SCALE,
            "qso_flux_density_1e-17_erg_cm2_s_A": qso_grid_flux * FLUX_DENSITY_SCALE,
            "weighted_qso_flux_density_1e-17_erg_cm2_s_A": weighted_qso_grid_flux
            * FLUX_DENSITY_SCALE,
            "composite_flux_density_1e-17_erg_cm2_s_A": composite_flux_density
            * FLUX_DENSITY_SCALE,
            "composite_flux_density_err_1e-17_erg_cm2_s_A": composite_flux_density_err
            * FLUX_DENSITY_SCALE,
            "galaxy_bin_flux_erg_cm2_s": galaxy_bin_flux,
            "weighted_qso_bin_flux_erg_cm2_s": weighted_qso_bin_flux,
            "composite_bin_flux_erg_cm2_s": composite_bin_flux,
            "composite_bin_flux_err_erg_cm2_s": composite_bin_flux_err,
        }
    )
    composite_table.meta.update(metadata)
    observed_ecsv = output_dir / f"{safe_id}_composite_spectrum.ecsv"
    composite_table.write(observed_ecsv, format="ascii.ecsv", overwrite=True)

    arrays = {
        "common_wave": common_wave,
        "galaxy_bin_flux": galaxy_bin_flux,
        "weighted_qso_bin_flux": weighted_qso_bin_flux,
        "composite_bin_flux": composite_bin_flux,
        "galaxy_flux_density_1e17": galaxy_grid_flux * FLUX_DENSITY_SCALE,
        "weighted_qso_flux_density_1e17": weighted_qso_grid_flux * FLUX_DENSITY_SCALE,
        "composite_flux_density_1e17": composite_flux_density * FLUX_DENSITY_SCALE,
        "composite_flux_density_err_1e17": composite_flux_density_err * FLUX_DENSITY_SCALE,
    }

    rest_ecsv = None
    if not args.no_rest:
        rest_wave = common_wave / (1.0 + target_redshift)
        rest_delta_lambda = delta_lambda / (1.0 + target_redshift)
        galaxy_rest_flux = galaxy_grid_flux * (1.0 + target_redshift)
        weighted_qso_rest_flux = weighted_qso_grid_flux * (1.0 + target_redshift)
        composite_rest_flux = composite_flux_density * (1.0 + target_redshift)
        composite_rest_flux_err = composite_flux_density_err * (1.0 + target_redshift)
        galaxy_rest_bin_flux = galaxy_rest_flux * rest_delta_lambda
        weighted_qso_rest_bin_flux = weighted_qso_rest_flux * rest_delta_lambda
        composite_rest_bin_flux = composite_rest_flux * rest_delta_lambda
        composite_rest_bin_flux_err = composite_rest_flux_err * rest_delta_lambda

        rest_table = Table(
            {
                "wavelength_rest_angstrom": rest_wave,
                "delta_lambda_rest_angstrom": rest_delta_lambda,
                "galaxy_flux_density_rest_erg_cm2_s_A": galaxy_rest_flux,
                "weighted_qso_flux_density_rest_erg_cm2_s_A": weighted_qso_rest_flux,
                "composite_flux_density_rest_erg_cm2_s_A": composite_rest_flux,
                "composite_flux_density_err_rest_erg_cm2_s_A": composite_rest_flux_err,
                "galaxy_flux_density_rest_1e-17_erg_cm2_s_A": galaxy_rest_flux
                * FLUX_DENSITY_SCALE,
                "weighted_qso_flux_density_rest_1e-17_erg_cm2_s_A": weighted_qso_rest_flux
                * FLUX_DENSITY_SCALE,
                "composite_flux_density_rest_1e-17_erg_cm2_s_A": composite_rest_flux
                * FLUX_DENSITY_SCALE,
                "composite_flux_density_err_rest_1e-17_erg_cm2_s_A": composite_rest_flux_err
                * FLUX_DENSITY_SCALE,
                "galaxy_bin_flux_rest_erg_cm2_s": galaxy_rest_bin_flux,
                "weighted_qso_bin_flux_rest_erg_cm2_s": weighted_qso_rest_bin_flux,
                "composite_bin_flux_rest_erg_cm2_s": composite_rest_bin_flux,
                "composite_bin_flux_err_rest_erg_cm2_s": composite_rest_bin_flux_err,
            }
        )
        rest_table.meta.update(metadata)
        rest_ecsv = output_dir / f"{safe_id}_composite_spectrum_rest_frame.ecsv"
        rest_table.write(rest_ecsv, format="ascii.ecsv", overwrite=True)

        arrays.update(
            {
                "rest_wave": rest_wave,
                "galaxy_rest_bin_flux": galaxy_rest_bin_flux,
                "weighted_qso_rest_bin_flux": weighted_qso_rest_bin_flux,
                "composite_rest_bin_flux": composite_rest_bin_flux,
                "galaxy_rest_flux_density_1e17": galaxy_rest_flux * FLUX_DENSITY_SCALE,
                "weighted_qso_rest_flux_density_1e17": weighted_qso_rest_flux
                * FLUX_DENSITY_SCALE,
                "composite_rest_flux_density_1e17": composite_rest_flux
                * FLUX_DENSITY_SCALE,
            }
        )

    if not args.no_plots:
        save_plots(output_dir, safe_id, chimera_id, arrays, qso_weight)

    print(f"Chimera ID: {chimera_id}")
    print(f"Galaxy spectrum: {galaxy_spectrum_path}")
    print(f"DR7Q spectrum: {dr7q_spectrum_path}")
    for diagnostic in diagnostics:
        if diagnostic.startswith("missing optional directory"):
            print(diagnostic)
    print(f"Wrote observed table: {observed_ecsv}")
    if rest_ecsv is not None:
        print(f"Wrote rest-frame table: {rest_ecsv}")
    if not args.no_plots:
        print(f"Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()
