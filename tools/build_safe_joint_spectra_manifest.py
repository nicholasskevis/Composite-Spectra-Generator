#!/usr/bin/env python
"""Build a safe notebook-6 spectra manifest for joint photometry+spectroscopy fits.

This script audits the generated Chimera composite spectra against the Chimera
benchmark photometry before allowing them into the joint-fit manifest.  It also
writes a clean copy of each accepted spectrum with a stricter mask.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


OVERLAP_FILTER_WAVELENGTHS_A = {
    "r_sdss": 6231.0,
    "i_sdss": 7625.0,
    "z_sdss": 9134.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate notebook-6 Chimera composite spectra against benchmark "
            "photometry and write a safe spectra manifest."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fit-manifest", type=Path, default=None)
    parser.add_argument("--input-spectra-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-scale", type=float, default=0.2)
    parser.add_argument("--max-scale", type=float, default=5.0)
    parser.add_argument("--min-overlap-bands", type=int, default=1)
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--max-negative-fraction", type=float, default=0.20)
    parser.add_argument(
        "--keep-nonpositive-flux",
        action="store_true",
        help="Do not remove non-positive flux pixels from the output mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing safe spectrum files.",
    )
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


def load_fit_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {str(row["object_id"]): row for row in rows}


def resolve_spectrum_path(input_manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(input_manifest.parent / path)
    candidates.append(input_manifest.parent / "spectra" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "; ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve spectrum path {raw_path!r}; searched {searched}")


def stricter_mask(table: Table, *, keep_nonpositive_flux: bool) -> np.ndarray:
    wave = np.asarray(table["wave_obs"], dtype=float)
    flux = np.asarray(table["flux_mjy"], dtype=float)
    err = np.asarray(table["flux_err_mjy"], dtype=float)
    mask = np.asarray(table["mask"], dtype=bool)
    out = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0.0) & mask
    if not keep_nonpositive_flux:
        out &= flux > 0.0
    return out


def scale_diagnostics(table: Table, fit_row: dict[str, str], mask: np.ndarray) -> dict[str, Any]:
    wave = np.asarray(table["wave_obs"], dtype=float)
    flux = np.asarray(table["flux_mjy"], dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux) & mask
    if np.count_nonzero(valid) < 2:
        return {"n_overlap_bands": 0, "median_spec_over_phot": float("nan")}

    wave_valid = wave[valid]
    flux_valid = flux[valid]
    ratios: list[float] = []
    payload: dict[str, Any] = {}
    for filter_name, wavelength in OVERLAP_FILTER_WAVELENGTHS_A.items():
        phot = parse_float(fit_row.get(filter_name))
        err = parse_float(fit_row.get(f"{filter_name}_err"))
        if not (math.isfinite(phot) and phot > 0.0):
            continue
        if not (float(np.nanmin(wave_valid)) <= wavelength <= float(np.nanmax(wave_valid))):
            continue
        spec = float(np.interp(wavelength, wave_valid, flux_valid))
        if not (math.isfinite(spec) and spec > 0.0):
            continue
        ratio = spec / phot
        ratios.append(ratio)
        payload[f"{filter_name}_spectrum_mjy"] = spec
        payload[f"{filter_name}_photometry_mjy"] = phot
        payload[f"{filter_name}_photometry_err_mjy"] = err
        payload[f"{filter_name}_spec_over_phot"] = ratio

    payload["n_overlap_bands"] = len(ratios)
    if ratios:
        payload["median_spec_over_phot"] = float(10.0 ** np.median(np.log10(ratios)))
        payload["min_spec_over_phot"] = float(np.nanmin(ratios))
        payload["max_spec_over_phot"] = float(np.nanmax(ratios))
    else:
        payload["median_spec_over_phot"] = float("nan")
        payload["min_spec_over_phot"] = float("nan")
        payload["max_spec_over_phot"] = float("nan")
    return payload


def write_safe_spectrum(src: Path, dst: Path, table: Table, mask: np.ndarray, diagnostics: dict[str, Any]) -> None:
    safe_table = table.copy(copy_data=True)
    safe_table["mask"] = mask
    safe_table.meta.update(
        {
            "safe_manifest_source": str(src),
            "safe_mask_requires_positive_flux": bool(np.all(np.asarray(safe_table["flux_mjy"])[mask] > 0.0)),
            "safe_median_spec_over_phot": diagnostics.get("median_spec_over_phot", np.nan),
            "safe_n_overlap_bands": diagnostics.get("n_overlap_bands", 0),
        }
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    safe_table.write(dst, format="ascii.ecsv", overwrite=True)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    fit_manifest = (args.fit_manifest or project_root / "fit_manifest.csv").expanduser().resolve()
    input_manifest = (
        args.input_spectra_manifest
        or project_root
        / "notebook_outputs"
        / "all_chimera_notebook6_spectra"
        / "chimera_notebook6_spectra_manifest.csv"
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or project_root / "notebook_outputs" / "safe_chimera_notebook6_spectra"
    ).expanduser().resolve()

    print(f"Project root: {project_root}")
    print(f"Fit manifest: {fit_manifest}")
    print(f"Input spectra manifest: {input_manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Allowed median spectrum/photometry scale: {args.min_scale:g}-{args.max_scale:g}")

    fit_rows = load_fit_rows(fit_manifest)
    input_rows = read_csv_rows(input_manifest)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(input_rows, start=1):
        chimera_id = str(row.get("chimera_id", "")).strip()
        base_payload: dict[str, Any] = {
            "chimera_id": chimera_id,
            "row_index": row.get("row_index", ""),
            "cosmos_id": row.get("cosmos_id", ""),
            "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
            "chimera_redshift": row.get("chimera_redshift", ""),
            "dr7q_redshift": row.get("dr7q_redshift", ""),
            "chimera_qso_weight": row.get("chimera_qso_weight", ""),
        }
        try:
            if row.get("status") != "success":
                raise ValueError(f"input row has status={row.get('status')!r}")
            fit_row = fit_rows.get(chimera_id)
            if fit_row is None:
                raise KeyError(f"no fit_manifest row for chimera_id={chimera_id!r}")

            src = resolve_spectrum_path(input_manifest, row["spectrum_path"])
            table = Table.read(src, format="ascii.ecsv")
            required = {"wave_obs", "flux_mjy", "flux_err_mjy", "mask"}
            missing = sorted(required.difference(table.colnames))
            if missing:
                raise ValueError(f"missing required columns {missing}")

            original_mask = np.asarray(table["mask"], dtype=bool)
            original_flux = np.asarray(table["flux_mjy"], dtype=float)
            original_valid = np.isfinite(original_flux) & original_mask
            negative_fraction = (
                float(np.count_nonzero(original_flux[original_valid] <= 0.0) / np.count_nonzero(original_valid))
                if np.count_nonzero(original_valid)
                else float("nan")
            )
            mask = stricter_mask(table, keep_nonpositive_flux=args.keep_nonpositive_flux)
            n_valid = int(np.count_nonzero(mask))
            if n_valid < args.min_valid_pixels:
                raise ValueError(f"only {n_valid} valid pixels after strict masking")
            if math.isfinite(negative_fraction) and negative_fraction > args.max_negative_fraction:
                raise ValueError(
                    f"negative/zero original pixel fraction {negative_fraction:.3f} exceeds "
                    f"{args.max_negative_fraction:.3f}"
                )

            diagnostics = scale_diagnostics(table, fit_row, mask)
            if int(diagnostics["n_overlap_bands"]) < args.min_overlap_bands:
                raise ValueError(
                    f"only {diagnostics['n_overlap_bands']} overlapping photometry bands; "
                    f"need {args.min_overlap_bands}"
                )
            scale = float(diagnostics["median_spec_over_phot"])
            if not (args.min_scale <= scale <= args.max_scale):
                raise ValueError(f"median spectrum/photometry scale {scale:.6g} is outside allowed range")

            safe_name = f"{safe_filename(chimera_id)}_safe_notebook6_spectrum.ecsv"
            dst = output_dir / "spectra" / safe_name
            if args.overwrite or not dst.exists():
                write_safe_spectrum(src, dst, table, mask, diagnostics)

            payload = {
                "status": "success",
                "action": "accepted",
                **base_payload,
                "spectrum_path": str(dst),
                "source_spectrum_path": str(src),
                "n_pixels": len(table),
                "n_valid_pixels": n_valid,
                "original_negative_fraction": negative_fraction,
                **diagnostics,
            }
            accepted.append(payload)
        except Exception as exc:
            rejected.append(
                {
                    "status": "rejected",
                    **base_payload,
                    "source_spectrum_path": row.get("spectrum_path", ""),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(input_rows)}; accepted={len(accepted)} rejected={len(rejected)}")

    manifest_path = output_dir / "safe_chimera_notebook6_spectra_manifest.csv"
    rejected_path = output_dir / "safe_chimera_notebook6_spectra_rejected.csv"
    write_csv(manifest_path, accepted)
    write_csv(rejected_path, rejected)
    print(f"Done. accepted={len(accepted)} rejected={len(rejected)}")
    print(f"Wrote safe manifest: {manifest_path}")
    print(f"Wrote rejected table: {rejected_path}")
    print(f"Safe spectra are under: {output_dir / 'spectra'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
