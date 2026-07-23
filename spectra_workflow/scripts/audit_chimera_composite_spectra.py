#!/usr/bin/env python
"""Audit Chimera composite spectra against benchmark photometry.

The goal is to test whether the notebook-18 composite-spectrum architecture is
compatible with the Chimera benchmark photometry.  The script writes:

* per-band composite spectrum vs manifest photometry ratios
* per-object scale/slope summaries
* low-QSO-weight and high-QSO-weight diagnostic subsets
* optional galaxy/QSO component diagnostics reconstructed from the source spectra

The band comparison uses local spectral flux near the filter effective
wavelength.  It is a deliberately simple diagnostic; if the result looks bad,
the next step is full filter-curve synthetic photometry.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


C_MS = 299792458.0

FILTER_WAVELENGTHS_A = {
    "u_sdss": 3543.0,
    "r_sdss": 6231.0,
    "i_sdss": 7625.0,
    "z_sdss": 9134.0,
    "J_2mass": 12350.0,
    "H_2mass": 16620.0,
    "Ks_2mass": 21590.0,
    "spitzer.irac.I1": 35634.0,
    "spitzer.irac.I2": 45110.0,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=None)
    parser.add_argument(
        "--chimera-fits",
        type=Path,
        default=None,
        help="Optional Chimera FITS table used to provide photometry rows absent from fit_manifest.csv.",
    )
    parser.add_argument("--spectra-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-window-a", type=float, default=150.0)
    parser.add_argument("--low-qso-weight", type=float, default=1.0e-3)
    parser.add_argument("--high-qso-weight", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--component-audit",
        action="store_true",
        help="Rebuild notebook-18 galaxy and weighted-QSO components from source spectra.",
    )
    parser.add_argument(
        "--include-nonpositive",
        action="store_true",
        help="Allow non-positive spectrum pixels in local band estimates.",
    )
    return parser.parse_args()


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


def load_fit_rows(path: Path) -> dict[str, dict[str, str]]:
    return {str(row["object_id"]): row for row in read_csv_rows(path)}


def scalar_to_text(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value)


def load_chimera_rows(path: Path) -> dict[str, dict[str, str]]:
    table = Table.read(path)
    out: dict[str, dict[str, str]] = {}
    for row in table:
        object_id = scalar_to_text(row["id"])
        payload = {name: scalar_to_text(row[name]) for name in table.colnames}
        payload["object_id"] = object_id
        if "ID_COSMOS" in payload:
            payload["COSMOS_ID0"] = payload["ID_COSMOS"]
        out[object_id] = payload
    return out


def load_photometry_rows(fit_manifest: Path, chimera_fits: Path | None) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if chimera_fits is not None:
        rows.update(load_chimera_rows(chimera_fits))
    rows.update(load_fit_rows(fit_manifest))
    return rows


def resolve_spectrum_path(spectra_manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(spectra_manifest.parent / path)
    candidates.append(spectra_manifest.parent / "spectra" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "; ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve spectrum path {raw_path!r}; searched {searched}")


def flux_lambda_to_mjy(wavelength_a: np.ndarray | float, flux_lambda: np.ndarray | float) -> np.ndarray | float:
    wave = np.asarray(wavelength_a, dtype=float)
    flux = np.asarray(flux_lambda, dtype=float)
    out = flux * 1.0e7 * (wave * 1.0e-10) ** 2 / C_MS / 1.0e-29
    if np.ndim(out) == 0:
        return float(out)
    return out


def local_spectral_flux(
    wave: np.ndarray,
    flux: np.ndarray,
    mask: np.ndarray,
    wavelength: float,
    *,
    window_a: float,
    include_nonpositive: bool,
) -> tuple[float, int, str]:
    valid = np.isfinite(wave) & np.isfinite(flux) & mask
    if not include_nonpositive:
        valid &= flux > 0.0
    if np.count_nonzero(valid) < 2:
        return float("nan"), 0, "no_valid_pixels"
    wave_valid = wave[valid]
    flux_valid = flux[valid]
    if not (float(np.nanmin(wave_valid)) <= wavelength <= float(np.nanmax(wave_valid))):
        return float("nan"), 0, "outside_spectrum"

    nearby = np.abs(wave_valid - wavelength) <= window_a
    if np.count_nonzero(nearby) >= 3:
        return float(np.nanmedian(flux_valid[nearby])), int(np.count_nonzero(nearby)), "local_median"
    return float(np.interp(wavelength, wave_valid, flux_valid)), 0, "interpolated"


def ratio_payload(spectrum_flux: float, phot_flux: float) -> dict[str, float]:
    if not (math.isfinite(spectrum_flux) and math.isfinite(phot_flux) and phot_flux > 0.0 and spectrum_flux > 0.0):
        return {
            "ratio_spectrum_over_manifest": float("nan"),
            "log10_ratio": float("nan"),
            "delta_mag_spectrum_minus_manifest": float("nan"),
        }
    ratio = spectrum_flux / phot_flux
    return {
        "ratio_spectrum_over_manifest": float(ratio),
        "log10_ratio": float(np.log10(ratio)),
        "delta_mag_spectrum_minus_manifest": float(-2.5 * np.log10(ratio)),
    }


def audit_composite_row(
    row: dict[str, str],
    fit_row: dict[str, str],
    spectra_manifest: Path,
    *,
    window_a: float,
    include_nonpositive: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], Table, Path]:
    spectrum_path = resolve_spectrum_path(spectra_manifest, row["spectrum_path"])
    table = Table.read(spectrum_path, format="ascii.ecsv")
    wave = np.asarray(table["wave_obs"], dtype=float)
    flux = np.asarray(table["flux_mjy"], dtype=float)
    mask = np.asarray(table["mask"], dtype=bool)

    qso_weight = parse_float(row.get("chimera_qso_weight", fit_row.get("chimera_QSO_weight")))
    base = {
        "object_id": fit_row["object_id"],
        "fit_index": fit_row.get("fit_index", ""),
        "COSMOS_ID0": fit_row.get("COSMOS_ID0", row.get("cosmos_id", "")),
        "dr7q_spectrum_id": row.get("dr7q_spectrum_id", fit_row.get("dr7q_spectrum_id", "")),
        "chimera_QSO_weight": qso_weight,
        "luminosity_bin": fit_row.get("luminosity_bin", ""),
        "redshift": fit_row.get("redshift", row.get("chimera_redshift", "")),
        "spectrum_path": str(spectrum_path),
        "spectrum_wave_min": float(np.nanmin(wave[mask])) if np.any(mask) else float("nan"),
        "spectrum_wave_max": float(np.nanmax(wave[mask])) if np.any(mask) else float("nan"),
    }

    band_rows: list[dict[str, Any]] = []
    for band, wavelength in FILTER_WAVELENGTHS_A.items():
        phot = parse_float(fit_row.get(band))
        phot_err = parse_float(fit_row.get(f"{band}_err"))
        spec_flux, n_local, method = local_spectral_flux(
            wave,
            flux,
            mask,
            wavelength,
            window_a=window_a,
            include_nonpositive=include_nonpositive,
        )
        payload = {
            **base,
            "band": band,
            "band_wavelength_a": wavelength,
            "manifest_flux_mjy": phot,
            "manifest_flux_err_mjy": phot_err,
            "spectrum_flux_mjy": spec_flux,
            "n_local_pixels": n_local,
            "estimate_method": method,
        }
        payload.update(ratio_payload(spec_flux, phot))
        band_rows.append(payload)

    summary = summarize_object(base, band_rows)
    return band_rows, summary, table, spectrum_path


def summarize_object(base: dict[str, Any], band_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in band_rows
        if math.isfinite(parse_float(row.get("log10_ratio"))) and row.get("estimate_method") != "outside_spectrum"
    ]
    out = {
        **base,
        "n_overlap_bands": len(valid),
        "median_ratio": float("nan"),
        "min_ratio": float("nan"),
        "max_ratio": float("nan"),
        "scatter_log10_ratio": float("nan"),
        "slope_log10_ratio_vs_log10_wave": float("nan"),
        "architecture_flag": "no_overlap",
    }
    if not valid:
        return out
    ratios = np.asarray([parse_float(row["ratio_spectrum_over_manifest"]) for row in valid], dtype=float)
    logs = np.asarray([parse_float(row["log10_ratio"]) for row in valid], dtype=float)
    waves = np.asarray([parse_float(row["band_wavelength_a"]) for row in valid], dtype=float)
    out["median_ratio"] = float(10.0 ** np.nanmedian(logs))
    out["min_ratio"] = float(np.nanmin(ratios))
    out["max_ratio"] = float(np.nanmax(ratios))
    out["scatter_log10_ratio"] = float(np.nanstd(logs))
    if len(valid) >= 2:
        out["slope_log10_ratio_vs_log10_wave"] = float(np.polyfit(np.log10(waves), logs, 1)[0])

    median = out["median_ratio"]
    scatter = out["scatter_log10_ratio"]
    if 0.8 <= median <= 1.25 and scatter <= 0.1:
        out["architecture_flag"] = "consistent"
    elif 0.5 <= median <= 2.0:
        out["architecture_flag"] = "scale_or_aperture_offset"
    else:
        out["architecture_flag"] = "large_mismatch"
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


def audit_components(
    module: Any,
    table: Table,
    fit_row: dict[str, str],
    base_row: dict[str, Any],
    *,
    window_a: float,
    include_nonpositive: bool,
) -> list[dict[str, Any]]:
    meta = table.meta
    galaxy_path = Path(str(meta.get("galaxy_spectrum_path", ""))).expanduser()
    qso_path = Path(str(meta.get("qso_spectrum_path", ""))).expanduser()
    if not galaxy_path.is_file() or not qso_path.is_file():
        raise FileNotFoundError(f"component source paths are missing: {galaxy_path}; {qso_path}")

    chimera_redshift = float(meta["chimera_redshift"])
    dr7q_redshift = float(meta["dr7q_redshift"])
    qso_weight = float(meta["chimera_qso_weight"])
    cosmos_ebv = float(meta.get("cosmos_ebv", 0.0))
    apply_extinction = str(meta.get("extinction_curve", "none")).lower() != "none"

    galaxy = module.shift_flux_density_to_redshift(
        module.load_zcosmos_spectrum(galaxy_path, chimera_redshift),
        chimera_redshift,
    )
    qso = module.shift_flux_density_to_redshift(
        module.load_sdss_dr7q_spectrum(qso_path, dr7q_redshift),
        chimera_redshift,
    )
    for spec in (galaxy, qso):
        attenuated, _ = module.apply_foreground_extinction(
            spec["wavelength_target_obs_angstrom"],
            spec["flux_density_target_obs"],
            cosmos_ebv,
            apply=apply_extinction,
        )
        spec["flux_density_extincted"] = attenuated

    rows: list[dict[str, Any]] = []
    for band, wavelength in FILTER_WAVELENGTHS_A.items():
        phot = parse_float(fit_row.get(band))
        galaxy_flux_lambda, galaxy_n, galaxy_method = local_spectral_flux(
            np.asarray(galaxy["wavelength_target_obs_angstrom"], dtype=float),
            np.asarray(galaxy["flux_density_extincted"], dtype=float),
            np.ones_like(np.asarray(galaxy["wavelength_target_obs_angstrom"], dtype=float), dtype=bool),
            wavelength,
            window_a=window_a,
            include_nonpositive=include_nonpositive,
        )
        qso_flux_lambda, qso_n, qso_method = local_spectral_flux(
            np.asarray(qso["wavelength_target_obs_angstrom"], dtype=float),
            np.asarray(qso["flux_density_extincted"], dtype=float),
            np.ones_like(np.asarray(qso["wavelength_target_obs_angstrom"], dtype=float), dtype=bool),
            wavelength,
            window_a=window_a,
            include_nonpositive=include_nonpositive,
        )
        galaxy_mjy = flux_lambda_to_mjy(wavelength, galaxy_flux_lambda) if math.isfinite(galaxy_flux_lambda) else float("nan")
        qso_mjy = flux_lambda_to_mjy(wavelength, qso_flux_lambda) if math.isfinite(qso_flux_lambda) else float("nan")
        weighted_qso_mjy = qso_weight * qso_mjy if math.isfinite(qso_mjy) else float("nan")
        component_sum = galaxy_mjy + weighted_qso_mjy if math.isfinite(galaxy_mjy) and math.isfinite(weighted_qso_mjy) else float("nan")
        qso_fraction = weighted_qso_mjy / component_sum if math.isfinite(component_sum) and component_sum > 0 else float("nan")
        payload = {
            **base_row,
            "band": band,
            "band_wavelength_a": wavelength,
            "manifest_flux_mjy": phot,
            "galaxy_component_mjy": galaxy_mjy,
            "qso_component_unweighted_mjy": qso_mjy,
            "weighted_qso_component_mjy": weighted_qso_mjy,
            "component_sum_mjy": component_sum,
            "weighted_qso_fraction": qso_fraction,
            "galaxy_estimate_method": galaxy_method,
            "qso_estimate_method": qso_method,
            "galaxy_n_local_pixels": galaxy_n,
            "qso_n_local_pixels": qso_n,
        }
        payload.update(ratio_payload(component_sum, phot))
        rows.append(payload)
    return rows


def summary_stats(rows: list[dict[str, Any]], *, low_qso_weight: float, high_qso_weight: float) -> list[dict[str, Any]]:
    groups = {
        "all": rows,
        "low_qso": [row for row in rows if parse_float(row.get("chimera_QSO_weight")) <= low_qso_weight],
        "high_qso": [row for row in rows if parse_float(row.get("chimera_QSO_weight")) >= high_qso_weight],
    }
    out: list[dict[str, Any]] = []
    for name, group_rows in groups.items():
        med = np.asarray([parse_float(row.get("median_ratio")) for row in group_rows], dtype=float)
        slopes = np.asarray([parse_float(row.get("slope_log10_ratio_vs_log10_wave")) for row in group_rows], dtype=float)
        med = med[np.isfinite(med)]
        slopes = slopes[np.isfinite(slopes)]
        out.append(
            {
                "group": name,
                "n_objects": len(group_rows),
                "n_with_overlap": int(med.size),
                "median_of_median_ratios": float(np.nanmedian(med)) if med.size else float("nan"),
                "p16_median_ratio": float(np.nanpercentile(med, 16.0)) if med.size else float("nan"),
                "p84_median_ratio": float(np.nanpercentile(med, 84.0)) if med.size else float("nan"),
                "median_abs_log10_slope": float(np.nanmedian(np.abs(slopes))) if slopes.size else float("nan"),
                "n_large_mismatch": sum(row.get("architecture_flag") == "large_mismatch" for row in group_rows),
                "n_scale_or_aperture_offset": sum(row.get("architecture_flag") == "scale_or_aperture_offset" for row in group_rows),
                "n_consistent": sum(row.get("architecture_flag") == "consistent" for row in group_rows),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    fit_manifest = (args.fit_manifest or project_root / "fit_manifest.csv").expanduser().resolve()
    chimera_fits = args.chimera_fits.expanduser().resolve() if args.chimera_fits is not None else None
    spectra_manifest = (
        args.spectra_manifest
        or WORKFLOW_ROOT
        / "outputs"
        / "all_chimera_spectra"
        / "chimera_spectra_manifest.csv"
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or WORKFLOW_ROOT / "outputs" / "chimera_composite_spectra_audit"
    ).expanduser().resolve()

    print(f"Project root: {project_root}")
    print(f"Fit manifest: {fit_manifest}")
    if chimera_fits is not None:
        print(f"Chimera FITS photometry: {chimera_fits}")
    print(f"Spectra manifest: {spectra_manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Component audit: {args.component_audit}")

    fit_rows = load_photometry_rows(fit_manifest, chimera_fits)
    spectra_rows = [row for row in read_csv_rows(spectra_manifest) if row.get("status", "success") == "success"]
    if args.max_rows is not None:
        spectra_rows = spectra_rows[: args.max_rows]

    component_module = load_component_module(project_root) if args.component_audit else None
    band_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, row in enumerate(spectra_rows, start=1):
        object_id = str(row.get("chimera_id", "")).strip()
        try:
            fit_row = fit_rows.get(object_id)
            if fit_row is None:
                raise KeyError(f"no fit_manifest row for {object_id!r}")
            one_band_rows, summary, table, _ = audit_composite_row(
                row,
                fit_row,
                spectra_manifest,
                window_a=args.local_window_a,
                include_nonpositive=args.include_nonpositive,
            )
            band_rows.extend(one_band_rows)
            object_rows.append(summary)
            if component_module is not None:
                component_rows.extend(
                    audit_components(
                        component_module,
                        table,
                        fit_row,
                        summary,
                        window_a=args.local_window_a,
                        include_nonpositive=args.include_nonpositive,
                    )
                )
        except Exception as exc:
            failures.append(
                {
                    "object_id": object_id,
                    "row_index": row.get("row_index", ""),
                    "spectrum_path": row.get("spectrum_path", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(spectra_rows)}; failures={len(failures)}")

    low_qso_band_rows = [
        row for row in band_rows if parse_float(row.get("chimera_QSO_weight")) <= args.low_qso_weight
    ]
    high_qso_band_rows = [
        row for row in band_rows if parse_float(row.get("chimera_QSO_weight")) >= args.high_qso_weight
    ]
    high_qso_component_rows = [
        row for row in component_rows if parse_float(row.get("chimera_QSO_weight")) >= args.high_qso_weight
    ]

    write_csv(output_dir / "composite_band_audit.csv", band_rows)
    write_csv(output_dir / "object_summary.csv", object_rows)
    write_csv(output_dir / "low_qso_galaxy_dominated_audit.csv", low_qso_band_rows)
    write_csv(output_dir / "high_qso_composite_audit.csv", high_qso_band_rows)
    write_csv(
        output_dir / "summary_stats.csv",
        summary_stats(
            object_rows,
            low_qso_weight=args.low_qso_weight,
            high_qso_weight=args.high_qso_weight,
        ),
    )
    write_csv(output_dir / "failures.csv", failures)
    if component_rows:
        write_csv(output_dir / "component_band_audit.csv", component_rows)
        write_csv(output_dir / "high_qso_qso_scaling_audit.csv", high_qso_component_rows)

    print(f"Objects audited: {len(object_rows)}")
    print(f"Band rows: {len(band_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
