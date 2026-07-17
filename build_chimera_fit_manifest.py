#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


CHIMERA_FILTER_NAMES = (
    "u_sdss",
    "r_sdss",
    "i_sdss",
    "z_sdss",
    "J_2mass",
    "H_2mass",
    "Ks_2mass",
    "spitzer.irac.I1",
    "spitzer.irac.I2",
)

CHIMERA_FILTER_COLUMN_MAP = {
    "spitzer.irac.I1": "IRAC1",
    "spitzer.irac.I2": "IRAC2",
}

LUMINOSITY_BINS = (
    ("L < 42", None, 42.0),
    ("42 < L < 43", 42.0, 43.0),
    ("43 < L < 44", 43.0, 44.0),
    ("44 < L < 45", 44.0, 45.0),
)

BASE_FIELDS = [
    "fit_index",
    "object_id",
    "COSMOS_ID0",
    "ID_COSMOS",
    "redshift",
    "chimera_QSO_weight",
    "resample_weight",
    "log_stellar_mass_truth",
    "SFR_BEST_GAL",
    "SFR_MED_GAL",
    "SFR_MED_MIN68_GAL",
    "SFR_MED_MAX68_GAL",
    "SSFR_BEST_GAL",
    "SSFR_MED_GAL",
    "SSFR_MED_MIN68_GAL",
    "SSFR_MED_MAX68_GAL",
    "logLbol_QSO",
    "logLbol_chimera",
    "logL5100_QSO",
    "e_logL5100_QSO",
    "logL5100_chimera",
    "logL3000_QSO",
    "e_logL3000_QSO",
    "logL3000_chimera",
    "logL1350_QSO",
    "e_logL1350_QSO",
    "logL1350_chimera",
    "luminosity_bin",
]

PROVENANCE_FIELDS = [
    "dr7q_name",
    "dr7q_plate",
    "dr7q_mjd",
    "dr7q_fiber",
    "dr7q_spectrum_id",
    "dr7q_redshift",
]


def _read_fits_by_id(path: Path, columns: list[str]) -> dict[str, dict[str, Any]]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        available_columns = set(data.names)
        out: dict[str, dict[str, Any]] = {}
        for index in range(len(data)):
            row_id = str(data["id"][index])
            row: dict[str, Any] = {}
            for col in columns:
                if col not in available_columns:
                    row[col] = np.nan
                    continue
                value = data[col][index]
                if np.ndim(value) == 0 and hasattr(value, "item"):
                    value = value.item()
                row[col] = value
            out[row_id] = row
        return out


def _load_provenance(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {row["chimera_id"]: row for row in csv.DictReader(fh)}


def _load_joined_rows(data_dir: Path, provenance_path: Path | None) -> list[dict[str, Any]]:
    phot_path = data_dir / "chimeras-grahsp.fits"
    truth_path = data_dir / "chimeras-fullinfo.fits"
    if not phot_path.is_file():
        raise RuntimeError(f"Chimera photometry FITS not found: {phot_path}")
    if not truth_path.is_file():
        raise RuntimeError(f"Chimera full-info FITS not found: {truth_path}")

    phot_filter_cols = [CHIMERA_FILTER_COLUMN_MAP.get(name, name) for name in CHIMERA_FILTER_NAMES]
    phot_cols = [
        "id",
        "ID_COSMOS",
        "redshift",
        "chimera_QSO_weight",
        "resample_weight",
        *phot_filter_cols,
        *[f"{name}_err" for name in phot_filter_cols],
    ]
    truth_cols = [
        "id",
        "MASS_MED_GAL",
        "SFR_BEST_GAL",
        "SFR_MED_GAL",
        "SFR_MED_MIN68_GAL",
        "SFR_MED_MAX68_GAL",
        "SSFR_BEST_GAL",
        "SSFR_MED_GAL",
        "SSFR_MED_MIN68_GAL",
        "SSFR_MED_MAX68_GAL",
        "resample_weight",
        "chimera_QSO_weight",
        "ID_COSMOS",
        "redshift",
        "logLbol_QSO",
        "logL5100_QSO",
        "e_logL5100_QSO",
        "logL3000_QSO",
        "e_logL3000_QSO",
        "logL1350_QSO",
        "e_logL1350_QSO",
        *[f"{column}_GAL" for column in phot_filter_cols],
        *[f"{column}_err_GAL" for column in phot_filter_cols],
    ]
    phot = _read_fits_by_id(phot_path, phot_cols)
    truth = _read_fits_by_id(truth_path, truth_cols)
    provenance = _load_provenance(provenance_path)

    rows = []
    for row_id in sorted(set(phot).intersection(truth)):
        prow = phot[row_id]
        trow = truth[row_id]
        row = {
            "id": row_id,
            "ID_COSMOS": str(prow["ID_COSMOS"]),
            "redshift": float(prow["redshift"]),
            "chimera_QSO_weight": float(prow["chimera_QSO_weight"]),
            "resample_weight": float(trow["resample_weight"]),
            "log_stellar_mass_truth": float(trow["MASS_MED_GAL"]),
            "logLbol_QSO": float(trow["logLbol_QSO"]),
        }
        for field in (
            "SFR_BEST_GAL",
            "SFR_MED_GAL",
            "SFR_MED_MIN68_GAL",
            "SFR_MED_MAX68_GAL",
            "SSFR_BEST_GAL",
            "SSFR_MED_GAL",
            "SSFR_MED_MIN68_GAL",
            "SSFR_MED_MAX68_GAL",
            "logL5100_QSO",
            "e_logL5100_QSO",
            "logL3000_QSO",
            "e_logL3000_QSO",
            "logL1350_QSO",
            "e_logL1350_QSO",
        ):
            row[field] = float(trow[field])
        for name in CHIMERA_FILTER_NAMES:
            column = CHIMERA_FILTER_COLUMN_MAP.get(name, name)
            row[name] = float(prow[column])
            row[f"{name}_err"] = float(prow[f"{column}_err"])
            row[f"{name}_gal"] = float(trow[f"{column}_GAL"])
            row[f"{name}_gal_err"] = float(trow[f"{column}_err_GAL"])
        if row_id in provenance:
            row.update({field: provenance[row_id].get(field, "") for field in PROVENANCE_FIELDS})
        rows.append(row)
    return rows


def _select_rows(rows: list[dict[str, Any]], include_over_luminous: bool) -> list[dict[str, Any]]:
    rows_with_lbol = []
    for row in rows:
        qso_weight = float(row["chimera_QSO_weight"])
        log_lbol_qso = float(row["logLbol_QSO"])
        if not np.isfinite(log_lbol_qso):
            continue
        if not np.isfinite(qso_weight) or qso_weight <= 0.0:
            continue
        enriched = dict(row)
        enriched["logLbol_chimera"] = float(log_lbol_qso + np.log10(qso_weight))
        for qso_field in ("logL5100_QSO", "logL3000_QSO", "logL1350_QSO"):
            qso_lum = float(row.get(qso_field, np.nan))
            chimera_field = qso_field.replace("_QSO", "_chimera")
            enriched[chimera_field] = float(qso_lum + np.log10(qso_weight)) if np.isfinite(qso_lum) else np.nan
        rows_with_lbol.append(enriched)

    selected = []
    for bin_label, lower, upper in LUMINOSITY_BINS:
        selected.extend(_rows_in_bin(rows_with_lbol, bin_label, lower, upper))
    if include_over_luminous:
        selected.extend(_rows_in_bin(rows_with_lbol, "L >= 45", 45.0, None))

    for fit_index, row in enumerate(selected):
        row["fit_index"] = fit_index
    return selected


def _rows_in_bin(
    rows: list[dict[str, Any]],
    bin_label: str,
    lower: float | None,
    upper: float | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        log_lbol = float(row["logLbol_chimera"])
        if not np.isfinite(log_lbol):
            continue
        lower_ok = True if lower is None else log_lbol >= lower
        upper_ok = True if upper is None else log_lbol < upper
        if not (lower_ok and upper_ok):
            continue
        out = {
            "object_id": str(row["id"]),
            "COSMOS_ID0": str(row["ID_COSMOS"]),
            "ID_COSMOS": str(row["ID_COSMOS"]),
            "redshift": float(row["redshift"]),
            "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
            "resample_weight": float(row["resample_weight"]),
            "log_stellar_mass_truth": float(row["log_stellar_mass_truth"]),
            "SFR_BEST_GAL": float(row["SFR_BEST_GAL"]),
            "SFR_MED_GAL": float(row["SFR_MED_GAL"]),
            "SFR_MED_MIN68_GAL": float(row["SFR_MED_MIN68_GAL"]),
            "SFR_MED_MAX68_GAL": float(row["SFR_MED_MAX68_GAL"]),
            "SSFR_BEST_GAL": float(row["SSFR_BEST_GAL"]),
            "SSFR_MED_GAL": float(row["SSFR_MED_GAL"]),
            "SSFR_MED_MIN68_GAL": float(row["SSFR_MED_MIN68_GAL"]),
            "SSFR_MED_MAX68_GAL": float(row["SSFR_MED_MAX68_GAL"]),
            "logLbol_QSO": float(row["logLbol_QSO"]),
            "logLbol_chimera": float(row["logLbol_chimera"]),
            "logL5100_QSO": float(row["logL5100_QSO"]),
            "e_logL5100_QSO": float(row["e_logL5100_QSO"]),
            "logL5100_chimera": float(row["logL5100_chimera"]),
            "logL3000_QSO": float(row["logL3000_QSO"]),
            "e_logL3000_QSO": float(row["e_logL3000_QSO"]),
            "logL3000_chimera": float(row["logL3000_chimera"]),
            "logL1350_QSO": float(row["logL1350_QSO"]),
            "e_logL1350_QSO": float(row["e_logL1350_QSO"]),
            "logL1350_chimera": float(row["logL1350_chimera"]),
            "luminosity_bin": bin_label,
        }
        for name in CHIMERA_FILTER_NAMES:
            out[name] = float(row[name])
            out[f"{name}_err"] = float(row[f"{name}_err"])
            out[f"{name}_gal"] = float(row[f"{name}_gal"])
            out[f"{name}_gal_err"] = float(row[f"{name}_gal_err"])
        for field in PROVENANCE_FIELDS:
            if field in row:
                out[field] = row[field]
        selected.append(out)
    return selected


def _fieldnames(include_provenance: bool) -> list[str]:
    phot_fields = []
    for name in CHIMERA_FILTER_NAMES:
        phot_fields.extend([name, f"{name}_err"])
    galaxy_phot_fields = []
    for name in CHIMERA_FILTER_NAMES:
        galaxy_phot_fields.extend([f"{name}_gal", f"{name}_gal_err"])
    fields = [*BASE_FIELDS]
    if include_provenance:
        fields.extend(PROVENANCE_FIELDS)
    fields.extend(phot_fields)
    fields.extend(galaxy_phot_fields)
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build fit_manifest.csv for Chimera JAXSEDfit/GRAHSPJ runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../grahspj/data/chimeras-2023-10-11"))
    parser.add_argument("--provenance", type=Path, default=Path("chimera_provenance.csv"))
    parser.add_argument("--output", type=Path, default=Path("fit_manifest.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("fit_manifest.summary.json"))
    parser.add_argument("--expected-count", type=int, default=13558)
    parser.add_argument("--include-over-luminous", action="store_true", help="Include rows with logLbol_chimera >= 45.")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    provenance = args.provenance.expanduser().resolve() if args.provenance is not None else None
    output = args.output.expanduser().resolve()
    summary_output = args.summary_output.expanduser().resolve()

    rows = _select_rows(_load_joined_rows(data_dir, provenance), args.include_over_luminous)
    if args.expected_count is not None and args.expected_count > 0 and len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} selected rows, found {len(rows)}.")

    include_provenance = bool(provenance and provenance.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames(include_provenance), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    bin_counts = Counter(row["luminosity_bin"] for row in rows)
    cosmos_counts = Counter(row["COSMOS_ID0"] for row in rows)
    summary = {
        "manifest": str(output),
        "data_dir": str(data_dir),
        "provenance": str(provenance) if provenance else None,
        "n_fit_rows": len(rows),
        "n_unique_COSMOS_ID0": len(cosmos_counts),
        "n_repeated_COSMOS_ID0": sum(1 for count in cosmos_counts.values() if count > 1),
        "bin_counts": dict(bin_counts),
        "first_fit_index": rows[0]["fit_index"] if rows else None,
        "last_fit_index": rows[-1]["fit_index"] if rows else None,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
