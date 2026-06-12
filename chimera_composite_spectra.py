#!/usr/bin/env python
"""Build a Chimera composite spectrum from local galaxy and DR7Q spectra.

This script is the command-line version of notebook 18. It reads
``chimera_provenance.csv``, finds the matching zCOSMOS/CESAM-style galaxy
spectrum and DR7Q spectrum, standardizes wavelength/flux-density units, shifts
the spectra to the Chimera redshift, optionally applies COSMOS foreground
extinction, and writes observed-frame/rest-frame tables and plots.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Chimera galaxy + weighted-QSO composite spectrum."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository/project root. Defaults to the parent of tools/.",
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
        help="Output directory. Defaults to notebook_outputs/18_chimera_composite_spectra.",
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


def load_zcosmos_spectrum(path: Path, chimera_redshift: float) -> dict[str, object]:
    with fits.open(path, memmap=True) as hdul:
        row = hdul[1].data[0]
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX_REDUCED"], dtype=float)
        err = np.asarray(row["ERR"], dtype=float) if "ERR" in hdul[1].data.names else None
        header = dict(hdul[0].header)
        ext_header = dict(hdul[1].header)
    wave, flux, err = finite_sorted_spectrum(wave, flux, err)
    return {
        "label": "galaxy",
        "path": path,
        "wavelength_obs_angstrom": wave,
        "flux_density_obs": flux,
        "flux_density_err_obs": err,
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


def interp_flux(spec: dict[str, object], grid: np.ndarray, flux_key="flux_density_extincted"):
    wave = np.asarray(spec["wavelength_target_obs_angstrom"], dtype=float)
    flux = np.asarray(spec[flux_key], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux)
    return np.interp(grid, wave[valid], flux[valid], left=np.nan, right=np.nan)


def bin_widths(wavelength: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    edges = np.empty(len(wavelength) + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return np.diff(edges)


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
    data_dir = project_root / "data"
    output_dir = args.output_dir or project_root / "notebook_outputs" / "18_chimera_composite_spectra"
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

    galaxy_spec = load_zcosmos_spectrum(galaxy_spectrum_path, chimera_redshift)
    qso_spec = load_sdss_dr7q_spectrum(dr7q_spectrum_path, dr7q_redshift)
    galaxy_shifted = shift_flux_density_to_redshift(galaxy_spec, chimera_redshift)
    qso_shifted = shift_flux_density_to_redshift(qso_spec, chimera_redshift)

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

    common_wave = make_common_grid([galaxy_shifted, qso_shifted])
    galaxy_grid_flux = interp_flux(galaxy_shifted, common_wave)
    qso_grid_flux = interp_flux(qso_shifted, common_wave)
    weighted_qso_grid_flux = qso_weight * qso_grid_flux
    composite_flux_density = galaxy_grid_flux + weighted_qso_grid_flux

    delta_lambda = bin_widths(common_wave)
    galaxy_bin_flux = galaxy_grid_flux * delta_lambda
    weighted_qso_bin_flux = weighted_qso_grid_flux * delta_lambda
    composite_bin_flux = composite_flux_density * delta_lambda

    metadata = {
        "chimera_id": chimera_id,
        "cosmos_id": cosmos_id,
        "dr7q_spectrum_id": provenance["dr7q_spectrum_id"],
        "chimera_redshift": chimera_redshift,
        "dr7q_redshift": dr7q_redshift,
        "chimera_qso_weight": qso_weight,
        "cosmos_ebv": cosmos_ebv,
        "galaxy_spectrum_path": str(galaxy_spectrum_path.relative_to(project_root)),
        "qso_spectrum_path": str(dr7q_spectrum_path.relative_to(project_root)),
        "extinction_curve": "CCM89 Rv=3.1" if apply_extinction else "none",
    }

    composite_table = Table(
        {
            "wavelength_obs_angstrom": common_wave,
            "delta_lambda_angstrom": delta_lambda,
            "galaxy_flux_density_erg_cm2_s_A": galaxy_grid_flux,
            "qso_flux_density_erg_cm2_s_A": qso_grid_flux,
            "weighted_qso_flux_density_erg_cm2_s_A": weighted_qso_grid_flux,
            "composite_flux_density_erg_cm2_s_A": composite_flux_density,
            "galaxy_flux_density_1e-17_erg_cm2_s_A": galaxy_grid_flux * FLUX_DENSITY_SCALE,
            "qso_flux_density_1e-17_erg_cm2_s_A": qso_grid_flux * FLUX_DENSITY_SCALE,
            "weighted_qso_flux_density_1e-17_erg_cm2_s_A": weighted_qso_grid_flux
            * FLUX_DENSITY_SCALE,
            "composite_flux_density_1e-17_erg_cm2_s_A": composite_flux_density
            * FLUX_DENSITY_SCALE,
            "galaxy_bin_flux_erg_cm2_s": galaxy_bin_flux,
            "weighted_qso_bin_flux_erg_cm2_s": weighted_qso_bin_flux,
            "composite_bin_flux_erg_cm2_s": composite_bin_flux,
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
    }

    rest_ecsv = None
    if not args.no_rest:
        rest_wave = common_wave / (1.0 + chimera_redshift)
        rest_delta_lambda = delta_lambda / (1.0 + chimera_redshift)
        galaxy_rest_flux = galaxy_grid_flux * (1.0 + chimera_redshift)
        weighted_qso_rest_flux = weighted_qso_grid_flux * (1.0 + chimera_redshift)
        composite_rest_flux = composite_flux_density * (1.0 + chimera_redshift)
        galaxy_rest_bin_flux = galaxy_rest_flux * rest_delta_lambda
        weighted_qso_rest_bin_flux = weighted_qso_rest_flux * rest_delta_lambda
        composite_rest_bin_flux = composite_rest_flux * rest_delta_lambda

        rest_table = Table(
            {
                "wavelength_rest_angstrom": rest_wave,
                "delta_lambda_rest_angstrom": rest_delta_lambda,
                "galaxy_flux_density_rest_erg_cm2_s_A": galaxy_rest_flux,
                "weighted_qso_flux_density_rest_erg_cm2_s_A": weighted_qso_rest_flux,
                "composite_flux_density_rest_erg_cm2_s_A": composite_rest_flux,
                "galaxy_flux_density_rest_1e-17_erg_cm2_s_A": galaxy_rest_flux
                * FLUX_DENSITY_SCALE,
                "weighted_qso_flux_density_rest_1e-17_erg_cm2_s_A": weighted_qso_rest_flux
                * FLUX_DENSITY_SCALE,
                "composite_flux_density_rest_1e-17_erg_cm2_s_A": composite_rest_flux
                * FLUX_DENSITY_SCALE,
                "galaxy_bin_flux_rest_erg_cm2_s": galaxy_rest_bin_flux,
                "weighted_qso_bin_flux_rest_erg_cm2_s": weighted_qso_rest_bin_flux,
                "composite_bin_flux_rest_erg_cm2_s": composite_rest_bin_flux,
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
