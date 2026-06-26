#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("hpc_outputs") / "matplotlib_cache"))

import matplotlib.pyplot as plt


SUMMARY_FIELDS = [
    "fit_index",
    "object_id",
    "COSMOS_ID0",
    "ID_COSMOS",
    "luminosity_bin",
    "redshift",
    "logLbol_QSO",
    "logLbol_chimera",
    "chimera_QSO_weight",
    "resample_weight",
    "log_stellar_mass_truth",
    "recovered_logm",
    "logm16",
    "logm84",
    "residual_log_ratio",
    "sfr_sample_key",
    "log_sfr",
    "log_sfr16",
    "log_sfr84",
    "sfr",
    "sfr16",
    "sfr84",
    "sampler",
    "backend",
    "sed_pdf_path",
    "corner_pdf_path",
    "trace_pdf_path",
]

FAILURE_FIELDS = [
    "fit_index",
    "object_id",
    "COSMOS_ID0",
    "ID_COSMOS",
    "luminosity_bin",
    "redshift",
    "logLbol_QSO",
    "logLbol_chimera",
    "chimera_QSO_weight",
    "error",
]


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _load_jsons(directory: Path) -> list[dict[str, Any]]:
    rows = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            row = json.load(fh)
        if "fit_index" not in row:
            row["fit_index"] = row.get("zero_based_index", row.get("array_index"))
        if "COSMOS_ID0" not in row and "ID_COSMOS" in row:
            row["COSMOS_ID0"] = row["ID_COSMOS"]
        if "log_stellar_mass_truth" not in row and "truth_logm" in row:
            row["log_stellar_mass_truth"] = row["truth_logm"]
        rows.append(row)
    rows.sort(key=lambda row: int(row.get("fit_index", row.get("zero_based_index", row.get("array_index", 0)))))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scatter(ax, rows: list[dict[str, Any]], x_key: str, y_key: str, *, xlabel: str, ylabel: str) -> None:
    x = [_float_or_nan(row.get(x_key)) for row in rows]
    y = [_float_or_nan(row.get(y_key)) for row in rows]
    pairs = [(xx, yy) for xx, yy in zip(x, y) if math.isfinite(xx) and math.isfinite(yy)]
    if not pairs:
        ax.text(0.5, 0.5, f"No finite {ylabel} values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    xx, yy = zip(*pairs)
    ax.scatter(xx, yy, s=10, alpha=0.65, linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Chimera JAXSEDfit per-AGN properties")

    _scatter(axes[0], rows, "fit_index", "logLbol_chimera", xlabel="fit index", ylabel="log AGN luminosity")
    _apply_limits(axes[0], rows, "fit_index", "logLbol_chimera")

    _scatter(axes[1], rows, "fit_index", "recovered_logm", xlabel="fit index", ylabel="log stellar mass")
    _apply_limits(axes[1], rows, "fit_index", "recovered_logm")

    _scatter(axes[2], rows, "fit_index", "log_sfr", xlabel="fit index", ylabel="log SFR")
    _apply_limits(axes[2], rows, "fit_index", "log_sfr")

    fig.savefig(output, dpi=200)
    plt.close(fig)


def _apply_limits(ax, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    xlim = _axis_limits(rows, x_key)
    ylim = _axis_limits(rows, y_key)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _axis_limits(rows: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = [_float_or_nan(row.get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        pad = max(0.5, abs(lo) * 0.05)
    else:
        pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge Chimera fit JSONs and plot AGN luminosity, stellar mass, and star-formation diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory containing results/ and failures/.")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--failures-csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    results = _load_jsons(output_dir / "results")
    failures = _load_jsons(output_dir / "failures")

    summary_csv = args.summary_csv or output_dir / "chimera_jaxsedfit_properties.csv"
    failures_csv = args.failures_csv or output_dir / "chimera_jaxsedfit_failures.csv"
    plot_path = args.plot or output_dir / "chimera_jaxsedfit_properties.png"

    _write_csv(summary_csv, results, SUMMARY_FIELDS)
    _write_csv(failures_csv, failures, FAILURE_FIELDS)
    _plot_summary(results, plot_path)

    with_sfr = sum(1 for row in results if row.get("sfr_sample_key"))
    print(f"Wrote {len(results)} successful rows to {summary_csv}")
    print(f"Wrote {len(failures)} failure rows to {failures_csv}")
    print(f"Wrote plot to {plot_path}")
    print(f"SFR sample key present for {with_sfr}/{len(results)} successful fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
