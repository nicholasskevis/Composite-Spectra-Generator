from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "/home/ns2385/project_pi_pn38/ns2385/grahspj_loglbol_mass_retrieval/july7_1948_chimera_grahspj"
)

PREFERRED_SUCCESS_FIELDS = [
    "fit_index",
    "object_id",
    "COSMOS_ID0",
    "ID_COSMOS",
    "backend",
    "sampler",
    "luminosity_bin",
    "redshift",
    "logLbol_QSO",
    "logLbol_chimera",
    "log_agn_bol_luminosity_fit_erg_s16",
    "log_agn_bol_luminosity_fit_erg_s",
    "log_agn_bol_luminosity_fit_erg_s84",
    "log_agn_bol_luminosity_fit16",
    "log_agn_bol_luminosity_fit",
    "log_agn_bol_luminosity_fit84",
    "logL5100_QSO",
    "logL5100_chimera",
    "log_agn_lambda_l5100_fit_erg_s16",
    "log_agn_lambda_l5100_fit_erg_s",
    "log_agn_lambda_l5100_fit_erg_s84",
    "log_agn_lambda_l5100_fit16",
    "log_agn_lambda_l5100_fit",
    "log_agn_lambda_l5100_fit84",
    "log_agn_l5100_fit_erg_s16",
    "log_agn_l5100_fit_erg_s",
    "log_agn_l5100_fit_erg_s84",
    "fracAGN_5100_fit16",
    "fracAGN_5100_fit",
    "fracAGN_5100_fit84",
    "chimera_QSO_weight",
    "resample_weight",
    "log_stellar_mass_truth",
    "recovered_logm",
    "logm16",
    "logm84",
    "residual_log_ratio",
    "sfr_sample_key",
    "log_sfr16",
    "log_sfr",
    "log_sfr84",
    "sfr16",
    "sfr",
    "sfr84",
    "SFR_MED_GAL",
    "SFR_MED_MIN68_GAL",
    "SFR_MED_MAX68_GAL",
    "sfr_truth",
    "log_sfr_truth",
    "sfr_current_fit16",
    "sfr_current_fit",
    "sfr_current_fit84",
    "log_sfr_current_fit16",
    "log_sfr_current_fit",
    "log_sfr_current_fit84",
    "sfr_100myr_fit16",
    "sfr_100myr_fit",
    "sfr_100myr_fit84",
    "log_sfr_100myr_fit16",
    "log_sfr_100myr_fit",
    "log_sfr_100myr_fit84",
    "fit_method",
    "optax_steps",
    "optax_lr",
    "nuts_warmup",
    "nuts_samples",
    "nuts_chains",
    "target_accept_prob",
    "sed_pdf_path",
    "corner_pdf_path",
    "trace_pdf_path",
    "posterior_samples_path",
    "posterior_samples_format",
    "posterior_samples_saved_keys",
    "posterior_samples_skipped_keys",
]

PREFERRED_FAILURE_FIELDS = [
    "fit_index",
    "object_id",
    "COSMOS_ID0",
    "ID_COSMOS",
    "backend",
    "luminosity_bin",
    "redshift",
    "logLbol_QSO",
    "logLbol_chimera",
    "chimera_QSO_weight",
    "log_stellar_mass_truth",
    "error",
]


def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
    fit_index = row.get("fit_index", row.get("zero_based_index", row.get("array_index", -1)))
    try:
        fit_index_int = int(fit_index)
    except (TypeError, ValueError):
        fit_index_int = -1
    return fit_index_int, str(row.get("object_id", ""))


def _normalize_row(row: dict[str, Any], path: Path) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("source_json", str(path))
    if "fit_index" not in out:
        out["fit_index"] = out.get("zero_based_index", out.get("array_index"))
    if "COSMOS_ID0" not in out and "ID_COSMOS" in out:
        out["COSMOS_ID0"] = out["ID_COSMOS"]
    if "ID_COSMOS" not in out and "COSMOS_ID0" in out:
        out["ID_COSMOS"] = out["COSMOS_ID0"]
    if "log_stellar_mass_truth" not in out and "truth_logm" in out:
        out["log_stellar_mass_truth"] = out["truth_logm"]
    if "backend" not in out:
        out["backend"] = "jaxsedfit"
    return out


def _load_jsons(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            rows.append(_normalize_row(json.load(fh), path))
    rows.sort(key=_sort_key)
    return rows


def _fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = set(preferred)
    extras = sorted({key for row in rows for key in row if key not in seen and not isinstance(row.get(key), (dict, list))})
    return preferred + extras


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fields(rows, preferred_fields)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, *, output_dir: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    attempted = len(results) + len(failures)
    payload = {
        "output_dir": str(output_dir),
        "results_count": len(results),
        "failures_count": len(failures),
        "attempted_count": attempted,
        "success_fraction": None if attempted == 0 else len(results) / attempted,
        "min_success_fit_index": None if not results else _sort_key(results[0])[0],
        "max_success_fit_index": None if not results else _sort_key(results[-1])[0],
        "min_failure_fit_index": None if not failures else _sort_key(failures[0])[0],
        "max_failure_fit_index": None if not failures else _sort_key(failures[-1])[0],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge grahspj/JAXSEDFit per-task logLbol JSON results from an HPC Slurm run."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Run directory containing results/ and failures/ subdirectories.",
    )
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--failures-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    results = _load_jsons(output_dir / "results")
    failures = _load_jsons(output_dir / "failures")

    results_csv = args.results_csv or output_dir / "chimera_mass_retrieval_by_logLbol_grahspj.csv"
    failures_csv = args.failures_csv or output_dir / "chimera_mass_retrieval_failures_by_logLbol_grahspj.csv"
    summary_json = args.summary_json or output_dir / "chimera_mass_retrieval_merge_summary_grahspj.json"

    _write_csv(results_csv.expanduser().resolve(), results, PREFERRED_SUCCESS_FIELDS)
    _write_csv(failures_csv.expanduser().resolve(), failures, PREFERRED_FAILURE_FIELDS)
    _write_summary(summary_json.expanduser().resolve(), output_dir=output_dir, results=results, failures=failures)

    print(f"Read run directory: {output_dir}")
    print(f"Wrote {len(results)} successful rows to {results_csv}")
    print(f"Wrote {len(failures)} failure rows to {failures_csv}")
    print(f"Wrote summary to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
