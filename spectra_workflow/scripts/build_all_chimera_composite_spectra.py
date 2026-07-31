#!/usr/bin/env python
"""Build publication-ready Chimera composite spectra in batch.

The script follows the same workflow as ``chimera_composite_spectra.py``:
find the zCOSMOS/CESAM galaxy spectrum and the matching SDSS DR7Q spectrum,
shift the QSO spectrum to the best galaxy spectroscopic redshift, match the
higher-resolution component down to the lower resolving power, apply the
Chimera QSO weight, convert the combined F_lambda spectrum to mJy, and write
an ECSV table with the columns expected by publication workflow:

    wave_obs, flux_mjy, flux_err_mjy, mask

Rows without both required spectra are skipped and recorded in a failure table.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import traceback
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


C_ANG_PER_S = 2.99792458e18
DR7Q_RE = re.compile(r"spec-(\d+)-(\d+)-(\d+)\.fits$")
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
        description="Create publication-ready composite spectra for all available Chimera IDs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root containing chimera_provenance.csv. Default: inferred repo root.",
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
        "--provenance",
        type=Path,
        default=None,
        help="Path to chimera_provenance.csv. Default: project-root/chimera_provenance.csv.",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=None,
        help=(
            "Optional fit_manifest.csv used to restrict spectra to the active fitting sample. "
            "Default: project-root/fit_manifest.csv if it exists."
        ),
    )
    parser.add_argument(
        "--ignore-fit-manifest",
        action="store_true",
        help="Inspect all provenance rows instead of restricting to fit_manifest.csv.",
    )
    parser.add_argument(
        "--zcosmos-matches",
        type=Path,
        default=None,
        help=(
            "Optional chimera_zcosmos_alpha_delta_matches.csv. If present, this "
            "explicit ID_COSMOS -> zcosmos_file map is used before FITS-header matching."
        ),
    )
    parser.add_argument(
        "--qso-spectrum-overrides",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping Chimera IDs or requested DR7Q keys to replacement QSO "
            "spectrum files. Expected columns: spectrum_path plus either chimera_id or "
            "dr7q_requested_key/plate/mjd/fiber."
        ),
    )
    parser.add_argument(
        "--source-match-audit",
        type=Path,
        default=None,
        help=(
            "Optional spectrum_source_match_audit.csv from audit_spectrum_source_matches.py. "
            "When supplied, the builder uses the audited galaxy/QSO spectrum paths, including "
            "coordinate-fallback galaxy matches."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "spectra_workflow/outputs/all_chimera_spectra."
        ),
    )
    parser.add_argument(
        "--chimera-id",
        action="append",
        default=None,
        help="Process only this Chimera ID. May be supplied more than once.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First provenance row to inspect.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum provenance rows to inspect.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing publication-ready spectra instead of skipping them.",
    )
    parser.add_argument(
        "--no-extinction",
        action="store_true",
        help="Skip COSMOS foreground attenuation. Useful if COSMOS2015 is unavailable.",
    )
    parser.add_argument(
        "--error-floor-fraction",
        type=float,
        default=0.10,
        help="Fractional error floor for spectra with missing/non-positive errors. Default: 0.10.",
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
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=50,
        help="Minimum valid pixels required in the publication-ready spectrum. Default: 50.",
    )
    parser.add_argument(
        "--write-full-table",
        action="store_true",
        help="Also write a component/full composite ECSV table for each success.",
    )
    parser.add_argument(
        "--write-rest-table",
        action="store_true",
        help="Also write a rest-frame component ECSV table for each success.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each skipped/created row.")
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


def load_fit_manifest_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["object_id"] for row in csv.DictReader(handle) if row.get("object_id")}


def load_qso_spectrum_overrides(path: Path | None, project_root: Path) -> tuple[dict[str, Path], dict[tuple[int, int, int], Path]]:
    by_chimera_id: dict[str, Path] = {}
    by_key: dict[tuple[int, int, int], Path] = {}
    if path is None:
        return by_chimera_id, by_key
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QSO spectrum override table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_path = row.get("spectrum_path") or row.get("qso_spectrum_path") or row.get("replacement_spectrum_path")
            if not raw_path:
                continue
            spectrum_path = Path(raw_path).expanduser()
            if not spectrum_path.is_absolute():
                candidates = [path.parent / spectrum_path, project_root / spectrum_path]
                spectrum_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            spectrum_path = spectrum_path.resolve()
            chimera_id = row.get("chimera_id") or row.get("object_id")
            if chimera_id:
                by_chimera_id[str(chimera_id)] = spectrum_path
            key_text = row.get("dr7q_requested_key") or row.get("requested_key") or row.get("dr7q_spectrum_id")
            if key_text:
                match = re.search(r"(\d+)[-_:](\d+)[-_:](\d+)", key_text)
                if match:
                    by_key[tuple(int(match.group(i)) for i in range(1, 4))] = spectrum_path
            key_cols = (row.get("dr7q_plate") or row.get("plate"), row.get("dr7q_mjd") or row.get("mjd"), row.get("dr7q_fiber") or row.get("fiber"))
            if all(key_cols):
                by_key[(int(key_cols[0]), int(key_cols[1]), int(key_cols[2]))] = spectrum_path
    return by_chimera_id, by_key


def resolve_existing_path(raw_path: str, base_dir: Path, project_root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, project_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_source_match_audit(path: Path | None, project_root: Path) -> dict[str, dict[str, Path]]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source-match audit table not found: {path}")
    out: dict[str, dict[str, Path]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            chimera_id = row.get("chimera_id", "")
            if not chimera_id:
                continue
            payload: dict[str, Path] = {}
            galaxy_path = resolve_existing_path(row.get("galaxy_spectrum_path", ""), path.parent, project_root)
            qso_path = resolve_existing_path(row.get("dr7q_spectrum_path", ""), path.parent, project_root)
            if galaxy_path is not None:
                payload["galaxy"] = galaxy_path
            if qso_path is not None:
                payload["qso"] = qso_path
            if payload:
                out[chimera_id] = payload
    return out


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
                    for header in headers:
                        for key in ("OBJECT", "ORIGFILE", "TITLE"):
                            if key not in header:
                                continue
                            for value in re.findall(r"\b\d{6,9}\b", clean_fits_text(header[key])):
                                index.setdefault(int(value), []).append(path)
            except Exception as exc:  # pragma: no cover - diagnostics only
                diagnostics.append(f"could not inspect {path}: {exc}")

    deduped: dict[int, list[Path]] = {}
    for object_id, paths in index.items():
        seen: list[Path] = []
        for path in paths:
            if path not in seen:
                seen.append(path)
        deduped[object_id] = seen
    return deduped, diagnostics


def find_existing_zcosmos_file(filename: str, directories: list[Path]) -> Path | None:
    filename = filename.strip()
    if not filename:
        return None
    variants = {filename, filename.replace(":", "_")}
    for directory in directories:
        for variant in variants:
            path = directory / variant
            if path.exists():
                return path
    return None


def find_zcosmos_matches_path(
    explicit_path: Path | None,
    project_root: Path,
    data_dir: Path,
) -> Path | None:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "config" / "chimera_zcosmos_alpha_delta_matches.csv",
            project_root / "chimera_zcosmos_alpha_delta_matches.csv",
            project_root / "spectra_workflow" / "config" / "chimera_zcosmos_alpha_delta_matches.csv",
            data_dir / "chimera_zcosmos_alpha_delta_matches.csv",
            data_dir.parent / "chimera_zcosmos_alpha_delta_matches.csv",
            data_dir.parent / "grahspj" / "chimera_zcosmos_alpha_delta_matches.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def build_zcosmos_match_index(path: Path, directories: list[Path]) -> tuple[dict[int, list[Path]], list[str]]:
    index: dict[int, list[Path]] = {}
    diagnostics: list[str] = []
    rows = load_provenance(path)
    for row in rows:
        try:
            cosmos_id = int(row["ID_COSMOS"])
        except (KeyError, TypeError, ValueError):
            diagnostics.append(f"could not read ID_COSMOS in {path}")
            continue
        spectrum_path = find_existing_zcosmos_file(row.get("zcosmos_file", ""), directories)
        if spectrum_path is None:
            diagnostics.append(
                f"missing matched zCOSMOS file for ID_COSMOS={cosmos_id}: {row.get('zcosmos_file', '')}"
            )
            continue
        index.setdefault(cosmos_id, []).append(spectrum_path)

    deduped: dict[int, list[Path]] = {}
    for cosmos_id, paths in index.items():
        seen: list[Path] = []
        for path_item in paths:
            if path_item not in seen:
                seen.append(path_item)
        deduped[cosmos_id] = seen
    return deduped, diagnostics


def merge_galaxy_indices(
    primary: dict[int, list[Path]],
    fallback: dict[int, list[Path]],
) -> dict[int, list[Path]]:
    merged: dict[int, list[Path]] = {key: list(paths) for key, paths in primary.items()}
    for key, paths in fallback.items():
        dest = merged.setdefault(key, [])
        for path in paths:
            if path not in dest:
                dest.append(path)
    return merged


def build_dr7q_spectrum_index(directory: Path) -> dict[tuple[int, int, int], Path]:
    index: dict[tuple[int, int, int], Path] = {}
    if not directory.exists():
        return index
    for path in sorted(directory.glob("spec-*.fits")):
        match = DR7Q_RE.match(path.name)
        if not match:
            continue
        plate, mjd, fiber = (int(match.group(i)) for i in range(1, 4))
        index[(plate, mjd, fiber)] = path
    return index


def load_cosmos_ebv(path: Path) -> dict[int, float]:
    ebv_by_id: dict[int, float] = {}
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)
        if "NUMBER" not in names or "EBV" not in names:
            raise KeyError(f"{path} must contain NUMBER and EBV columns")
        for number, ebv in zip(data["NUMBER"], data["EBV"]):
            ebv_by_id[int(number)] = float(ebv)
    return ebv_by_id


def finite_sorted_spectrum(wavelength, flux_density, error=None):
    wavelength = np.asarray(wavelength, dtype=float)
    flux_density = np.asarray(flux_density, dtype=float)
    if error is None:
        error = np.full_like(flux_density, np.nan, dtype=float)
    else:
        error = np.asarray(error, dtype=float)

    valid = np.isfinite(wavelength) & np.isfinite(flux_density) & (wavelength > 0.0)
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


def load_zcosmos_spectrum(path: Path, chimera_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        row = hdul[1].data[0]
        names = set(hdul[1].data.names)
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX_REDUCED"], dtype=float)
        err_column = next((name for name in ZCOSMOS_ERROR_COLUMNS if name in names), None)
        err = np.asarray(row[err_column], dtype=float) if err_column is not None else None
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "galaxy",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
        "flux_density_err_column": err_column or "",
        "source_redshift": chimera_redshift,
    }


def load_sdss_dr7q_spectrum(path: Path, dr7q_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        coadd = hdul[1].data
        names = set(coadd.names)
        if "loglam" in names:
            wave = 10.0 ** np.asarray(coadd["loglam"], dtype=float)
        elif "lambda" in names:
            wave = np.asarray(coadd["lambda"], dtype=float)
        else:
            raise KeyError(f"No wavelength/loglam column found in {path}")
        flux = np.asarray(coadd["flux"], dtype=float) * 1e-17
        if "ivar" in names:
            ivar = np.asarray(coadd["ivar"], dtype=float)
            err = np.full_like(flux, np.nan, dtype=float)
            good_ivar = ivar > 0.0
            err[good_ivar] = 1.0 / np.sqrt(ivar[good_ivar]) * 1e-17
        else:
            err = None
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "qso",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
        "source_redshift": dr7q_redshift,
    }


def shift_flux_density_to_redshift(spec: dict[str, object], target_redshift: float) -> dict[str, object]:
    z_source = float(spec["source_redshift"])
    wave_obs = np.asarray(spec["wavelength_obs_angstrom"], dtype=float)
    flux_obs = np.asarray(spec["flux_density_obs"], dtype=float)
    err_obs = np.asarray(spec["flux_density_err_obs"], dtype=float)

    wave_rest = wave_obs / (1.0 + z_source)
    flux_rest = flux_obs * (1.0 + z_source)
    err_rest = err_obs * (1.0 + z_source)

    out = dict(spec)
    out.update(
        {
            "wavelength_rest_angstrom": wave_rest,
            "flux_density_rest": flux_rest,
            "flux_density_err_rest": err_rest,
            "wavelength_target_obs_angstrom": wave_rest * (1.0 + target_redshift),
            "flux_density_target_obs": flux_rest / (1.0 + target_redshift),
            "flux_density_err_target_obs": err_rest / (1.0 + target_redshift),
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


def flambda_cgs_to_mjy(wave_angstrom, flux_cgs):
    wave_angstrom = np.asarray(wave_angstrom, dtype=float)
    flux_cgs = np.asarray(flux_cgs, dtype=float)
    f_nu_cgs = flux_cgs * wave_angstrom**2 / C_ANG_PER_S
    return f_nu_cgs / 1.0e-26


def apply_error_floor(error, flux, fraction):
    error = np.asarray(error, dtype=float)
    flux = np.asarray(flux, dtype=float)
    floor = np.maximum(abs(float(fraction)) * np.abs(flux), 1.0e-300)
    return np.where(np.isfinite(error) & (error > 0.0), np.maximum(error, floor), floor)


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
    if n < 2:
        raise ValueError("The common wavelength grid has fewer than two pixels.")
    return start + step * np.arange(n)


def interp_flux(spec: dict[str, object], grid: np.ndarray, flux_key="flux_density_extincted"):
    wave = np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float)
    flux = np.asarray(spec[flux_key], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(valid) < 2:
        return np.full_like(grid, np.nan, dtype=float)
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


def bin_widths(wavelength: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    if len(wavelength) < 2:
        return np.ones_like(wavelength)
    return np.diff(pixel_edges_from_centers(wavelength))


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def process_row(
    row: dict[str, str],
    row_index: int,
    args: argparse.Namespace,
    project_root: Path,
    output_dir: Path,
    galaxy_index: dict[int, list[Path]],
    dr7q_index: dict[tuple[int, int, int], Path],
    source_match_paths: dict[str, dict[str, Path]],
    qso_overrides_by_chimera_id: dict[str, Path],
    qso_overrides_by_key: dict[tuple[int, int, int], Path],
    ebv_by_id: dict[int, float],
) -> dict[str, object]:
    chimera_id = row["chimera_id"]
    safe_id = safe_filename(chimera_id)
    spectrum_dir = output_dir / "spectra"
    full_dir = output_dir / "full_tables"
    rest_dir = output_dir / "rest_tables"
    spectrum_path = spectrum_dir / f"{safe_id}_spectrum.ecsv"

    if spectrum_path.exists() and not args.overwrite:
        table = Table.read(spectrum_path, format="ascii.ecsv")
        mask = np.asarray(table["mask"], dtype=bool)
        wave = np.asarray(table["wave_obs"], dtype=float)
        return {
            "status": "success",
            "action": "skipped_existing",
            "row_index": row_index,
            "chimera_id": chimera_id,
            "cosmos_id": row["cosmos_id"],
            "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
            "chimera_redshift": row["chimera_redshift"],
            "dr7q_redshift": row["dr7q_redshift"],
            "chimera_qso_weight": row["chimera_qso_weight"],
            "spectrum_path": str(spectrum_path),
            "n_pixels": int(len(table)),
            "n_valid_pixels": int(np.count_nonzero(mask)),
            "wave_min": float(np.nanmin(wave[mask])) if np.any(mask) else np.nan,
            "wave_max": float(np.nanmax(wave[mask])) if np.any(mask) else np.nan,
            "galaxy_spectrum_path": table.meta.get("galaxy_spectrum_path", ""),
            "qso_spectrum_path": table.meta.get("qso_spectrum_path", ""),
        }

    cosmos_id = int(row["cosmos_id"])
    dr7q_key = (int(row["dr7q_plate"]), int(row["dr7q_mjd"]), int(row["dr7q_fiber"]))
    audited_paths = source_match_paths.get(chimera_id, {})
    galaxy_paths = [audited_paths["galaxy"]] if "galaxy" in audited_paths else galaxy_index.get(cosmos_id, [])
    if not galaxy_paths:
        raise FileNotFoundError(f"missing galaxy spectrum for COSMOS ID {cosmos_id}")
    dr7q_spectrum_path = (
        qso_overrides_by_chimera_id.get(chimera_id)
        or qso_overrides_by_key.get(dr7q_key)
        or audited_paths.get("qso")
        or dr7q_index.get(dr7q_key)
    )
    if dr7q_spectrum_path is None:
        raise FileNotFoundError(f"missing DR7Q spectrum for plate/mjd/fiber {dr7q_key}")

    galaxy_spectrum_path = galaxy_paths[0]
    chimera_redshift = float(row["chimera_redshift"])
    target_redshift, target_redshift_source = best_galaxy_redshift(row, chimera_redshift)
    dr7q_redshift = float(row["dr7q_redshift"])
    qso_weight = float(row["chimera_qso_weight"])
    cosmos_ebv = ebv_by_id.get(cosmos_id, 0.0)

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
    for spec in (galaxy_shifted, qso_shifted):
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
    galaxy_flux, galaxy_err, galaxy_coverage = resample_flux_density(
        galaxy_shifted,
        common_wave,
        method=args.resampling_method,
    )
    qso_flux, qso_err, qso_coverage = resample_flux_density(
        qso_shifted,
        common_wave,
        method=args.resampling_method,
    )
    weighted_qso_flux = qso_weight * qso_flux
    composite_flux = galaxy_flux + weighted_qso_flux

    weighted_qso_err = abs(qso_weight) * qso_err
    composite_err_raw = np.sqrt(galaxy_err**2 + weighted_qso_err**2)
    composite_err = apply_error_floor(composite_err_raw, composite_flux, args.error_floor_fraction)

    composite_mjy = flambda_cgs_to_mjy(common_wave, composite_flux)
    composite_err_mjy = flambda_cgs_to_mjy(common_wave, composite_err)
    mask = (
        np.isfinite(common_wave)
        & np.isfinite(composite_mjy)
        & np.isfinite(composite_err_mjy)
        & (composite_err_mjy > 0.0)
        & (common_wave > 0.0)
    )
    n_valid = int(np.count_nonzero(mask))
    if n_valid < args.min_valid_pixels:
        raise ValueError(f"only {n_valid} valid pixels; need at least {args.min_valid_pixels}")

    metadata = {
        "chimera_id": chimera_id,
        "cosmos_id": cosmos_id,
        "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
        "chimera_redshift": chimera_redshift,
        "target_redshift": target_redshift,
        "target_redshift_source": target_redshift_source,
        "galaxy_spectroscopic_redshift": target_redshift,
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "cosmos_ebv": cosmos_ebv,
        "galaxy_spectrum_path": maybe_relative(galaxy_spectrum_path, project_root),
        "qso_spectrum_path": maybe_relative(dr7q_spectrum_path, project_root),
        "galaxy_error_column": galaxy_spec.get("flux_density_err_column", ""),
        "qso_error_source": "ivar" if np.any(np.isfinite(qso_spec["flux_density_err_obs"])) else "",
        "extinction_curve": "CCM89 Rv=3.1" if apply_extinction else "none",
        "format": "grahspj publication workflow SpectroscopyData input",
        "flux_unit": "mJy",
        "wave_unit": "Angstrom",
        "instrument": "ChimeraComposite",
        "aperture_diameter_arcsec": np.nan,
        "error_floor_fraction": args.error_floor_fraction,
        "error_propagation": "sqrt(zCOSMOS_error^2 + (chimera_qso_weight * SDSS_DR7Q_error)^2), then error floor",
        "wavelength_grid": "native zCOSMOS galaxy observed-frame grid clipped to QSO overlap",
        "resampling_method": args.resampling_method,
        "resampling_note": "Flux-conserving mode rebins flux density through pixel-edge overlap integrals after resolution matching.",
        "galaxy_rebin_min_coverage": float(np.nanmin(galaxy_coverage[mask])) if np.any(mask) else np.nan,
        "qso_rebin_min_coverage": float(np.nanmin(qso_coverage[mask])) if np.any(mask) else np.nan,
    }
    metadata.update(resolution_metadata)

    spectrum_dir.mkdir(parents=True, exist_ok=True)
    spectrum_table = Table(
        {
            "wave_obs": common_wave,
            "flux_mjy": composite_mjy,
            "flux_err_mjy": composite_err_mjy,
            "mask": mask,
        }
    )
    spectrum_table.meta.update(metadata)
    spectrum_table.write(spectrum_path, format="ascii.ecsv", overwrite=True)

    if args.write_full_table or args.write_rest_table:
        delta_lambda = bin_widths(common_wave)

    if args.write_full_table:
        full_dir.mkdir(parents=True, exist_ok=True)
        full_table = Table(
            {
                "wavelength_obs_angstrom": common_wave,
                "delta_lambda_angstrom": delta_lambda,
                "galaxy_flux_density_erg_cm2_s_A": galaxy_flux,
                "qso_flux_density_erg_cm2_s_A": qso_flux,
                "weighted_qso_flux_density_erg_cm2_s_A": weighted_qso_flux,
                "composite_flux_density_erg_cm2_s_A": composite_flux,
                "galaxy_flux_density_err_erg_cm2_s_A": galaxy_err,
                "qso_flux_density_err_erg_cm2_s_A": qso_err,
                "weighted_qso_flux_density_err_erg_cm2_s_A": weighted_qso_err,
                "composite_flux_density_err_raw_erg_cm2_s_A": composite_err_raw,
                "composite_flux_density_err_erg_cm2_s_A": composite_err,
                "galaxy_rebin_coverage": galaxy_coverage,
                "qso_rebin_coverage": qso_coverage,
                "composite_flux_mjy": composite_mjy,
                "composite_flux_err_mjy": composite_err_mjy,
                "spectrum_mask": mask,
                "composite_bin_flux_erg_cm2_s": composite_flux * delta_lambda,
            }
        )
        full_table.meta.update(metadata)
        full_table.write(
            full_dir / f"{safe_id}_composite_spectrum.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )

    if args.write_rest_table:
        rest_dir.mkdir(parents=True, exist_ok=True)
        rest_wave = common_wave / (1.0 + target_redshift)
        rest_delta_lambda = delta_lambda / (1.0 + target_redshift)
        galaxy_rest_flux = galaxy_flux * (1.0 + target_redshift)
        weighted_qso_rest_flux = weighted_qso_flux * (1.0 + target_redshift)
        composite_rest_flux = composite_flux * (1.0 + target_redshift)
        rest_table = Table(
            {
                "wavelength_rest_angstrom": rest_wave,
                "delta_lambda_rest_angstrom": rest_delta_lambda,
                "galaxy_flux_density_rest_erg_cm2_s_A": galaxy_rest_flux,
                "weighted_qso_flux_density_rest_erg_cm2_s_A": weighted_qso_rest_flux,
                "composite_flux_density_rest_erg_cm2_s_A": composite_rest_flux,
                "composite_bin_flux_rest_erg_cm2_s": composite_rest_flux * rest_delta_lambda,
            }
        )
        rest_table.meta.update(metadata)
        rest_table.write(
            rest_dir / f"{safe_id}_composite_spectrum_rest_frame.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )

    return {
        "status": "success",
        "action": "created",
        "row_index": row_index,
        "chimera_id": chimera_id,
        "cosmos_id": cosmos_id,
        "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
        "chimera_redshift": chimera_redshift,
        "target_redshift": target_redshift,
        "target_redshift_source": target_redshift_source,
        "galaxy_spectroscopic_redshift": target_redshift,
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "spectrum_path": str(spectrum_path),
        "n_pixels": int(len(common_wave)),
        "n_valid_pixels": n_valid,
        "wave_min": float(np.nanmin(common_wave[mask])),
        "wave_max": float(np.nanmax(common_wave[mask])),
        "galaxy_spectrum_path": str(galaxy_spectrum_path),
        "qso_spectrum_path": str(dr7q_spectrum_path),
        "resolution_match_enabled": resolution_metadata["resolution_match_enabled"],
        "target_resolving_power": resolution_metadata["target_resolving_power"],
        "galaxy_resolution_convolved": resolution_metadata["galaxy_resolution_convolved"],
        "qso_resolution_convolved": resolution_metadata["qso_resolution_convolved"],
        "resampling_method": args.resampling_method,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).resolve()
    provenance_path = (args.provenance or project_root / "chimera_provenance.csv").resolve()
    fit_manifest_path = (args.fit_manifest or project_root / "fit_manifest.csv").resolve()
    output_dir = (
        args.output_dir
        or WORKFLOW_ROOT / "outputs" / "all_chimera_spectra"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    (output_dir / "mplconfig").mkdir(parents=True, exist_ok=True)

    rows = load_provenance(provenance_path)
    selected_ids = set(args.chimera_id or [])
    fit_manifest_ids: set[str] = set()
    if not args.ignore_fit_manifest and fit_manifest_path.is_file():
        fit_manifest_ids = load_fit_manifest_ids(fit_manifest_path)
    indexed_rows = list(enumerate(rows))
    if selected_ids:
        indexed_rows = [(i, row) for i, row in indexed_rows if row.get("chimera_id") in selected_ids]
    if fit_manifest_ids:
        indexed_rows = [(i, row) for i, row in indexed_rows if row.get("chimera_id") in fit_manifest_ids]
    indexed_rows = indexed_rows[args.start_index :]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]

    galaxy_dirs = [
        data_dir / "zCOSMOS_data",
        data_dir / "zCOSMOS selected",
        data_dir / "cesam_vudz",
        data_dir / "cesam_vuds",
    ]
    print(f"Project root: {project_root}")
    print(f"Data dir: {data_dir}")
    print(f"Provenance: {provenance_path}")
    if fit_manifest_ids:
        print(f"Restricted to fit manifest: {fit_manifest_path} ({len(fit_manifest_ids)} IDs)")
    else:
        print("Fit manifest restriction: off")
    print(f"Output dir: {output_dir}")
    print(f"Rows selected for inspection: {len(indexed_rows)}")
    print("Indexing galaxy spectra...")
    zcosmos_matches_path = find_zcosmos_matches_path(args.zcosmos_matches, project_root, data_dir)
    if zcosmos_matches_path is not None:
        zcosmos_index, zcosmos_diagnostics = build_zcosmos_match_index(
            zcosmos_matches_path,
            galaxy_dirs,
        )
        print(f"Indexed explicit zCOSMOS matches for {len(zcosmos_index)} COSMOS IDs")
        print(f"Using zCOSMOS match table: {zcosmos_matches_path}")
    else:
        zcosmos_index = {}
        zcosmos_diagnostics = []
        print("No chimera_zcosmos_alpha_delta_matches.csv found; using FITS-header matching only")
    fallback_index, diagnostics = build_galaxy_spectrum_index(galaxy_dirs)
    galaxy_index = merge_galaxy_indices(zcosmos_index, fallback_index)
    print(f"Indexed galaxy spectra for {len(galaxy_index)} COSMOS IDs")
    for diagnostic in diagnostics:
        if diagnostic.startswith("missing optional directory"):
            print(diagnostic)
    missing_matched_files = [
        item for item in zcosmos_diagnostics if item.startswith("missing matched zCOSMOS file")
    ]
    if missing_matched_files:
        print(f"Missing files referenced by zCOSMOS match table: {len(missing_matched_files)}")

    print("Indexing DR7Q spectra...")
    dr7q_index = build_dr7q_spectrum_index(data_dir / "dr7q_spectra")
    print(f"Indexed {len(dr7q_index)} DR7Q spectra")
    source_match_paths = load_source_match_audit(args.source_match_audit, project_root)
    if args.source_match_audit:
        print(f"Loaded source-match audit paths for {len(source_match_paths)} Chimera IDs")
    qso_overrides_by_chimera_id, qso_overrides_by_key = load_qso_spectrum_overrides(args.qso_spectrum_overrides, project_root)
    if args.qso_spectrum_overrides:
        print(
            "Loaded QSO spectrum overrides: "
            f"{len(qso_overrides_by_chimera_id)} by Chimera ID, {len(qso_overrides_by_key)} by requested key"
        )

    if args.no_extinction:
        ebv_by_id: dict[int, float] = {}
    else:
        cosmos_path = data_dir / "COSMOS2015_Laigle+_v1.1.fits"
        print(f"Reading COSMOS EBV: {cosmos_path}")
        ebv_by_id = load_cosmos_ebv(cosmos_path)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for count, (row_index, row) in enumerate(indexed_rows, start=1):
        chimera_id = row.get("chimera_id", "")
        try:
            payload = process_row(
                row,
                row_index,
                args,
                project_root,
                output_dir,
                galaxy_index,
                dr7q_index,
                source_match_paths,
                qso_overrides_by_chimera_id,
                qso_overrides_by_key,
                ebv_by_id,
            )
            successes.append(payload)
            if args.verbose:
                print(f"[{count}/{len(indexed_rows)}] ok {chimera_id} ({payload['action']})")
        except Exception as exc:
            failure = {
                "status": "failed",
                "row_index": row_index,
                "chimera_id": chimera_id,
                "cosmos_id": row.get("cosmos_id", ""),
                "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            if args.verbose:
                print(f"[{count}/{len(indexed_rows)}] skip {chimera_id}: {failure['reason']}")

        if count % 500 == 0:
            print(f"Processed {count}/{len(indexed_rows)}; successes={len(successes)} failures={len(failures)}")

    manifest_path = output_dir / "chimera_spectra_manifest.csv"
    failures_path = output_dir / "chimera_spectra_failures.csv"
    write_csv(manifest_path, successes)
    write_csv(failures_path, failures)

    print(f"Done. successes={len(successes)} failures={len(failures)}")
    print(f"Wrote manifest: {manifest_path}")
    if failures:
        print(f"Wrote failures: {failures_path}")
    print(f"Composite spectra are under: {output_dir / 'spectra'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
