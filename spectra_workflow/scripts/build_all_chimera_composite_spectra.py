#!/usr/bin/env python
"""Build publication-ready Chimera composite spectra in batch.

The script follows the same workflow as ``chimera_composite_spectra.py``:
find the zCOSMOS/CESAM galaxy spectrum and the matching SDSS DR7Q spectrum,
shift both to the Chimera redshift, apply the Chimera QSO weight, convert the
combined F_lambda spectrum to mJy, and write an ECSV table with the columns
expected by publication workflow:

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


def load_zcosmos_spectrum(path: Path, chimera_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        row = hdul[1].data[0]
        names = set(hdul[1].data.names)
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX_REDUCED"], dtype=float)
        err = np.asarray(row["ERR"], dtype=float) if "ERR" in names else None
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "galaxy",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
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


def bin_widths(wavelength: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    edges = np.empty(len(wavelength) + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return np.diff(edges)


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
    dr7q_redshift = float(row["dr7q_redshift"])
    qso_weight = float(row["chimera_qso_weight"])
    cosmos_ebv = ebv_by_id.get(cosmos_id, 0.0)

    galaxy_spec = load_zcosmos_spectrum(galaxy_spectrum_path, chimera_redshift)
    qso_spec = load_sdss_dr7q_spectrum(dr7q_spectrum_path, dr7q_redshift)
    galaxy_shifted = shift_flux_density_to_redshift(galaxy_spec, chimera_redshift)
    qso_shifted = shift_flux_density_to_redshift(qso_spec, chimera_redshift)

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

    common_wave = make_common_grid([galaxy_shifted, qso_shifted])
    galaxy_flux = interp_flux(galaxy_shifted, common_wave)
    qso_flux = interp_flux(qso_shifted, common_wave)
    weighted_qso_flux = qso_weight * qso_flux
    composite_flux = galaxy_flux + weighted_qso_flux

    galaxy_err = interp_positive_error(galaxy_shifted, common_wave)
    qso_err = interp_positive_error(qso_shifted, common_wave)
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
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "cosmos_ebv": cosmos_ebv,
        "galaxy_spectrum_path": maybe_relative(galaxy_spectrum_path, project_root),
        "qso_spectrum_path": maybe_relative(dr7q_spectrum_path, project_root),
        "extinction_curve": "CCM89 Rv=3.1" if apply_extinction else "none",
        "format": "grahspj publication workflow SpectroscopyData input",
        "flux_unit": "mJy",
        "wave_unit": "Angstrom",
        "instrument": "ChimeraComposite",
        "aperture_diameter_arcsec": np.nan,
        "error_floor_fraction": args.error_floor_fraction,
    }

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
        rest_wave = common_wave / (1.0 + chimera_redshift)
        rest_delta_lambda = delta_lambda / (1.0 + chimera_redshift)
        galaxy_rest_flux = galaxy_flux * (1.0 + chimera_redshift)
        weighted_qso_rest_flux = weighted_qso_flux * (1.0 + chimera_redshift)
        composite_rest_flux = composite_flux * (1.0 + chimera_redshift)
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
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "spectrum_path": str(spectrum_path),
        "n_pixels": int(len(common_wave)),
        "n_valid_pixels": n_valid,
        "wave_min": float(np.nanmin(common_wave[mask])),
        "wave_max": float(np.nanmax(common_wave[mask])),
        "galaxy_spectrum_path": str(galaxy_spectrum_path),
        "qso_spectrum_path": str(dr7q_spectrum_path),
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
