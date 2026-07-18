#!/usr/bin/env python3
"""Build a JAXSEDFit manifest directly from COSMOS-only photometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


FILTER_MAP = (
    ("u_sdss", "CFHT_u", "CFHT_u_err"),
    ("r_sdss", "SUBARU_r", "SUBARU_r_err"),
    ("i_sdss", "SUBARU_i", "SUBARU_i_err"),
    ("z_sdss", "SUBARU_z", "SUBARU_z_err"),
    ("J_2mass", "WFCAM_J", "WFCAM_J_err"),
    ("H_2mass", "CFHT_H", "CFHT_H_err"),
    ("Ks_2mass", "CFHT_K", "CFHT_K_err"),
    ("spitzer.irac.I1", "IRAC1", "IRAC1_err"),
    ("spitzer.irac.I2", "IRAC2", "IRAC2_err"),
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
    "dr7q_name",
    "dr7q_ra",
    "dr7q_dec",
    "dr7q_plate",
    "dr7q_mjd",
    "dr7q_fiberid",
    "dr7q_spectrum_id",
    "dr7q_redshift",
]
FILTER_FIELDS = [name for filt, _, _ in FILTER_MAP for name in (filt, f"{filt}_err")]
FIELDNAMES = BASE_FIELDS + FILTER_FIELDS


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _column(table: Table, name: str) -> np.ndarray:
    if name not in table.colnames:
        raise KeyError(f"Missing required column {name!r}")
    return np.asarray(table[name])


def _valid_flux_error(flux: float, err: float, *, allow_negative_flux: bool) -> bool:
    if not (math.isfinite(flux) and math.isfinite(err)):
        return False
    if err <= 0:
        return False
    if not allow_negative_flux and flux <= 0:
        return False
    return True


def _row_value(row: Any, name: str) -> float:
    return _finite_float(row[name]) if name in row.colnames else math.nan


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    table = Table.read(args.cosmos_fits)
    for _, flux_col, err_col in FILTER_MAP:
        _column(table, flux_col)
        _column(table, err_col)
    for required in ("id", "redshift", args.truth_mass_column):
        _column(table, required)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    n_written = 0
    skipped_by_reason: dict[str, int] = {}

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in table:
            cosmos_id = str(row["id"]).strip()
            redshift = _finite_float(row["redshift"])
            truth_mass = _finite_float(row[args.truth_mass_column])
            if not cosmos_id or not math.isfinite(redshift) or not math.isfinite(truth_mass):
                n_skipped += 1
                skipped_by_reason["missing_id_redshift_or_truth"] = skipped_by_reason.get("missing_id_redshift_or_truth", 0) + 1
                continue

            manifest_row: dict[str, Any] = {
                "fit_index": n_written,
                "object_id": f"COSMOS{cosmos_id}",
                "COSMOS_ID0": cosmos_id,
                "ID_COSMOS": cosmos_id,
                "redshift": redshift,
                "chimera_QSO_weight": 0.0,
                "resample_weight": 1.0,
                "log_stellar_mass_truth": truth_mass,
                "SFR_BEST_GAL": _row_value(row, "SFR_BEST"),
                "SFR_MED_GAL": _row_value(row, "SFR_MED"),
                "SFR_MED_MIN68_GAL": _row_value(row, "SFR_MED_MIN68"),
                "SFR_MED_MAX68_GAL": _row_value(row, "SFR_MED_MAX68"),
                "SSFR_BEST_GAL": _row_value(row, "SSFR_BEST"),
                "SSFR_MED_GAL": _row_value(row, "SSFR_MED"),
                "SSFR_MED_MIN68_GAL": _row_value(row, "SSFR_MED_MIN68"),
                "SSFR_MED_MAX68_GAL": _row_value(row, "SSFR_MED_MAX68"),
                "logLbol_QSO": 0.0,
                "logLbol_chimera": 0.0,
                "logL5100_QSO": 0.0,
                "e_logL5100_QSO": 0.0,
                "logL5100_chimera": 0.0,
                "logL3000_QSO": 0.0,
                "e_logL3000_QSO": 0.0,
                "logL3000_chimera": 0.0,
                "logL1350_QSO": 0.0,
                "e_logL1350_QSO": 0.0,
                "logL1350_chimera": 0.0,
                "luminosity_bin": "COSMOS only",
                "dr7q_name": "",
                "dr7q_ra": "",
                "dr7q_dec": "",
                "dr7q_plate": "",
                "dr7q_mjd": "",
                "dr7q_fiberid": "",
                "dr7q_spectrum_id": "",
                "dr7q_redshift": "",
            }

            valid_filters = 0
            for filter_name, flux_col, err_col in FILTER_MAP:
                flux = _finite_float(row[flux_col])
                err = _finite_float(row[err_col])
                if _valid_flux_error(flux, err, allow_negative_flux=args.allow_negative_flux):
                    valid_filters += 1
                else:
                    flux = math.nan
                    err = math.nan
                manifest_row[filter_name] = flux
                manifest_row[f"{filter_name}_err"] = err

            if valid_filters < args.min_valid_filters:
                n_skipped += 1
                skipped_by_reason["too_few_valid_filters"] = skipped_by_reason.get("too_few_valid_filters", 0) + 1
                continue

            writer.writerow(manifest_row)
            n_written += 1

    summary = {
        "cosmos_fits": str(args.cosmos_fits),
        "output": str(output),
        "input_rows": len(table),
        "written_rows": n_written,
        "skipped_rows": n_skipped,
        "skipped_by_reason": skipped_by_reason,
        "truth_mass_column": args.truth_mass_column,
        "min_valid_filters": args.min_valid_filters,
        "allow_negative_flux": args.allow_negative_flux,
        "filter_map": [
            {"manifest_filter": filt, "flux_column": flux_col, "error_column": err_col}
            for filt, flux_col, err_col in FILTER_MAP
        ],
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-fits", type=Path, required=True, help="Filtered COSMOS FITS file to convert.")
    parser.add_argument("--output", type=Path, default=Path("fit_manifest_cosmos_only.csv"), help="Output manifest CSV.")
    parser.add_argument("--truth-mass-column", default="MASS_MED", help="COSMOS column to use as log stellar-mass truth.")
    parser.add_argument("--min-valid-filters", type=int, default=len(FILTER_MAP), help="Minimum required valid filters per object.")
    parser.add_argument(
        "--allow-negative-flux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep finite negative fluxes when their errors are positive.",
    )
    args = parser.parse_args()

    summary = build_manifest(args)
    print(f"Input COSMOS rows: {summary['input_rows']}")
    print(f"Wrote manifest rows: {summary['written_rows']}")
    print(f"Skipped rows: {summary['skipped_rows']}")
    print(f"Manifest: {summary['output']}")
    print(f"Summary: {summary['output']}.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
