#!/usr/bin/env python
"""Plot spectroscopic completeness for Chimera COSMOS hosts.

Run this after ``search_sdss_desi_galaxy_spectra.py`` finishes.  If
``search_cosmos_spectroscopic_catalogs.py`` has also finished, the plot also
includes a broader catalog-level completeness estimate from every external
spectroscopic-redshift catalog queried there.

By default, a candidate only counts if its redshift is consistent with the
Chimera host redshift.  Use ``--include-redshift-inconsistent`` for a looser
"coordinate-only candidate" census.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Default: project-root/chimera_provenance.csv.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help=(
            "Candidate table from search_sdss_desi_galaxy_spectra.py. "
            "Default: spectra_workflow/outputs/sdss_desi_galaxy_spectra/galaxy_spectrum_candidates.csv."
        ),
    )
    parser.add_argument(
        "--zcosmos-audit",
        type=Path,
        default=None,
        help=(
            "Source-match audit table used to count existing zCOSMOS galaxy spectra. "
            "Default: spectra_workflow/outputs/source_match_audit/spectrum_source_match_audit.csv."
        ),
    )
    parser.add_argument(
        "--external-candidates",
        type=Path,
        default=None,
        help=(
            "Candidate table from search_cosmos_spectroscopic_catalogs.py. "
            "Default: spectra_workflow/outputs/external_spectroscopic_catalogs/"
            "external_spectroscopic_catalog_candidates.csv. If missing, external catalogs count as zero."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/sdss_desi_galaxy_spectra/completeness.",
    )
    parser.add_argument(
        "--survey-label",
        default="zCOSMOS + SDSS + DESI DR2 + external catalogs",
        help="Label used in plot titles. Default: zCOSMOS + SDSS + DESI DR2 + external catalogs.",
    )
    parser.add_argument(
        "--include-redshift-inconsistent",
        action="store_true",
        help="Count coordinate matches even when the match redshift disagrees with the Chimera host redshift.",
    )
    parser.add_argument(
        "--redshift-bins",
        default="0,0.4,0.8,1.2,1.6,2.0,3.0,6.0",
        help="Comma-separated redshift bin edges for the completeness-by-redshift panel.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return str(value).strip()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def host_id(row: dict[str, str]) -> str:
    return clean_text(row.get("cosmos_id") or row.get("ID_COSMOS") or row.get("COSMOS_ID0"))


def host_redshift(row: dict[str, str]) -> float:
    for key in ("galaxy_spectroscopic_redshift", "host_redshift", "chimera_redshift", "redshift"):
        z = finite_float(row.get(key))
        if math.isfinite(z):
            return z
    return float("nan")


def load_denominator(provenance_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(provenance_path)
    hosts: dict[str, dict[str, Any]] = {}
    for row in rows:
        cosmos_id = host_id(row)
        if not cosmos_id:
            continue
        entry = hosts.setdefault(
            cosmos_id,
            {
                "cosmos_id": cosmos_id,
                "n_chimeras": 0,
                "host_redshift": host_redshift(row),
                "example_chimera_id": clean_text(row.get("chimera_id", "")),
            },
        )
        entry["n_chimeras"] += 1
        if not math.isfinite(entry["host_redshift"]):
            entry["host_redshift"] = host_redshift(row)
    return hosts


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_detected_services(
    candidates_path: Path,
    *,
    require_redshift_consistent: bool,
) -> dict[str, set[str]]:
    detected: dict[str, set[str]] = defaultdict(set)
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Missing candidate table: {candidates_path}")
    for row in read_csv(candidates_path):
        cosmos_id = host_id(row)
        service = clean_text(row.get("service", "")).lower()
        if not cosmos_id or service not in {"sdss", "desi"}:
            continue
        if require_redshift_consistent and not truthy(row.get("redshift_consistent")):
            continue
        detected[cosmos_id].add(service)
    return detected


def load_external_catalog_services(
    candidates_path: Path,
    *,
    require_redshift_consistent: bool,
) -> dict[str, set[str]]:
    detected: dict[str, set[str]] = defaultdict(set)
    if not candidates_path.is_file():
        return detected
    for row in read_csv(candidates_path):
        cosmos_id = host_id(row)
        if not cosmos_id:
            continue
        if require_redshift_consistent and not truthy(row.get("redshift_consistent")):
            continue
        label = clean_text(row.get("catalog_label") or row.get("service") or "external_catalog")
        detected[cosmos_id].add(f"external:{label}")
    return detected


def load_zcosmos_services(audit_path: Path | None) -> dict[str, set[str]]:
    detected: dict[str, set[str]] = defaultdict(set)
    if audit_path is None or not audit_path.is_file():
        return detected
    for row in read_csv(audit_path):
        cosmos_id = host_id(row)
        if not cosmos_id:
            continue
        n_paths = finite_float(row.get("n_combined_galaxy_paths"))
        has_path = bool(clean_text(row.get("galaxy_spectrum_path", "")))
        if has_path or (math.isfinite(n_paths) and n_paths > 0):
            detected[cosmos_id].add("zcosmos")
    return detected


def merge_service_maps(*maps: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for service_map in maps:
        for cosmos_id, services in service_map.items():
            merged[cosmos_id].update(services)
    return merged


def service_category(services: set[str]) -> str:
    has_zcosmos = "zcosmos" in services
    has_sdss_desi = bool({"sdss", "desi"}.intersection(services))
    if has_zcosmos and has_sdss_desi:
        return "zCOSMOS + SDSS/DESI"
    if has_zcosmos:
        return "zCOSMOS only"
    if has_sdss_desi:
        return "SDSS/DESI only"
    return "No spectrum"


def percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def finite_max(values: list[np.ndarray], fallback: float = 5.0) -> float:
    finite_values = np.concatenate([np.ravel(value[np.isfinite(value)]) for value in values])
    if finite_values.size == 0:
        return fallback
    return max(fallback, float(np.max(finite_values)))


def has_any(services: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(service.startswith(prefix) for service in services for prefix in prefixes)


def detected_counts(
    hosts: dict[str, dict[str, Any]],
    detected: dict[str, set[str]],
    *,
    prefixes: tuple[str, ...],
) -> tuple[int, int]:
    n_hosts = 0
    n_chimeras = 0
    for cosmos_id, host in hosts.items():
        if has_any(detected.get(cosmos_id, set()), prefixes):
            n_hosts += 1
            n_chimeras += int(host["n_chimeras"])
    return n_hosts, n_chimeras


def build_summary_rows(
    hosts: dict[str, dict[str, Any]],
    detected_all: dict[str, set[str]],
    detected_zcosmos: dict[str, set[str]],
    detected_sdss_desi: dict[str, set[str]],
    detected_external: dict[str, set[str]],
    z_edges: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_hosts = len(hosts)
    total_chimeras = sum(int(row["n_chimeras"]) for row in hosts.values())
    core_detected = merge_service_maps(detected_zcosmos, detected_sdss_desi)
    rows_to_count = [
        ("zCOSMOS", detected_zcosmos, ("zcosmos",)),
        ("SDSS/DESI", detected_sdss_desi, ("sdss", "desi")),
        ("zCOSMOS + SDSS/DESI", core_detected, ("zcosmos", "sdss", "desi")),
        ("Full queried catalogs", detected_all, ("zcosmos", "sdss", "desi", "external:")),
    ]
    summary_rows = []
    for category, detected_map, prefixes in rows_to_count:
        n_hosts_found, n_chimeras_found = detected_counts(hosts, detected_map, prefixes=prefixes)
        summary_rows.append(
            {
                "category": category,
                "n_total_unique_cosmos_hosts": total_hosts,
                "n_unique_cosmos_hosts": n_hosts_found,
                "pct_unique_cosmos_hosts": percent(n_hosts_found, total_hosts),
                "n_total_chimera_rows": total_chimeras,
                "n_chimera_rows": n_chimeras_found,
                "pct_chimera_rows": percent(n_chimeras_found, total_chimeras),
            }
        )

    z_rows = []
    detected_core = merge_service_maps(detected_zcosmos, detected_sdss_desi)
    for lo, hi in zip(z_edges[:-1], z_edges[1:]):
        in_bin = [
            (cosmos_id, host)
            for cosmos_id, host in hosts.items()
            if math.isfinite(float(host["host_redshift"])) and lo <= float(host["host_redshift"]) < hi
        ]
        n_hosts = len(in_bin)
        n_chimeras = sum(int(host["n_chimeras"]) for _, host in in_bin)
        n_hosts_zcosmos = sum(1 for cosmos_id, _ in in_bin if detected_zcosmos.get(cosmos_id))
        n_hosts_sdss_desi = sum(1 for cosmos_id, _ in in_bin if detected_sdss_desi.get(cosmos_id))
        n_hosts_core = sum(1 for cosmos_id, _ in in_bin if detected_core.get(cosmos_id))
        n_hosts_external = sum(1 for cosmos_id, _ in in_bin if detected_external.get(cosmos_id))
        n_hosts_any = sum(1 for cosmos_id, _ in in_bin if detected_all.get(cosmos_id))
        n_chimeras_zcosmos = sum(int(host["n_chimeras"]) for cosmos_id, host in in_bin if detected_zcosmos.get(cosmos_id))
        n_chimeras_sdss_desi = sum(
            int(host["n_chimeras"]) for cosmos_id, host in in_bin if detected_sdss_desi.get(cosmos_id)
        )
        n_chimeras_core = sum(int(host["n_chimeras"]) for cosmos_id, host in in_bin if detected_core.get(cosmos_id))
        n_chimeras_external = sum(int(host["n_chimeras"]) for cosmos_id, host in in_bin if detected_external.get(cosmos_id))
        n_chimeras_any = sum(int(host["n_chimeras"]) for cosmos_id, host in in_bin if detected_all.get(cosmos_id))
        z_rows.append(
            {
                "z_min": lo,
                "z_max": hi,
                "n_unique_cosmos_hosts": n_hosts,
                "n_unique_cosmos_hosts_with_zcosmos": n_hosts_zcosmos,
                "pct_unique_cosmos_hosts_with_zcosmos": percent(n_hosts_zcosmos, n_hosts),
                "n_unique_cosmos_hosts_with_sdss_desi": n_hosts_sdss_desi,
                "pct_unique_cosmos_hosts_with_sdss_desi": percent(n_hosts_sdss_desi, n_hosts),
                "n_unique_cosmos_hosts_with_zcosmos_sdss_desi": n_hosts_core,
                "pct_unique_cosmos_hosts_with_zcosmos_sdss_desi": percent(n_hosts_core, n_hosts),
                "n_unique_cosmos_hosts_with_external_catalog": n_hosts_external,
                "pct_unique_cosmos_hosts_with_external_catalog": percent(n_hosts_external, n_hosts),
                "n_unique_cosmos_hosts_with_any_spectrum": n_hosts_any,
                "pct_unique_cosmos_hosts_with_any_spectrum": percent(n_hosts_any, n_hosts),
                "n_chimera_rows": n_chimeras,
                "n_chimera_rows_with_zcosmos": n_chimeras_zcosmos,
                "pct_chimera_rows_with_zcosmos": percent(n_chimeras_zcosmos, n_chimeras),
                "n_chimera_rows_with_sdss_desi": n_chimeras_sdss_desi,
                "pct_chimera_rows_with_sdss_desi": percent(n_chimeras_sdss_desi, n_chimeras),
                "n_chimera_rows_with_zcosmos_sdss_desi": n_chimeras_core,
                "pct_chimera_rows_with_zcosmos_sdss_desi": percent(n_chimeras_core, n_chimeras),
                "n_chimera_rows_with_external_catalog": n_chimeras_external,
                "pct_chimera_rows_with_external_catalog": percent(n_chimeras_external, n_chimeras),
                "n_chimera_rows_with_any_spectrum": n_chimeras_any,
                "pct_chimera_rows_with_any_spectrum": percent(n_chimeras_any, n_chimeras),
            }
        )
    return summary_rows, z_rows


def plot_completeness(
    summary_rows: list[dict[str, Any]],
    z_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    survey_label: str,
    require_redshift_consistent: bool,
    dpi: int,
) -> None:
    categories = [row["category"] for row in summary_rows]
    unique_pct = np.array([float(row["pct_unique_cosmos_hosts"]) for row in summary_rows])
    chimera_pct = np.array([float(row["pct_chimera_rows"]) for row in summary_rows])
    unique_counts = [int(row["n_unique_cosmos_hosts"]) for row in summary_rows]
    chimera_counts = [int(row["n_chimera_rows"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    x = np.arange(len(categories))
    width = 0.38
    ax = axes[0]
    ax.bar(x - width / 2, unique_pct, width, label="unique COSMOS hosts", color="#377eb8")
    ax.bar(x + width / 2, chimera_pct, width, label="Chimera rows", color="#4daf4a")
    for xpos, pct, count in zip(x - width / 2, unique_pct, unique_counts):
        ax.text(xpos, pct + 1.0, f"{count}", ha="center", va="bottom", fontsize=8)
    for xpos, pct, count in zip(x + width / 2, chimera_pct, chimera_counts):
        ax.text(xpos, pct + 1.0, f"{count}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=18, ha="right")
    ax.set_ylabel("Fraction of sample [%]")
    ax.set_ylim(0, finite_max([unique_pct, chimera_pct]) * 1.25)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    centers = np.array([(float(row["z_min"]) + float(row["z_max"])) / 2.0 for row in z_rows])
    widths = np.array([float(row["z_max"]) - float(row["z_min"]) for row in z_rows])
    core_z = np.array([float(row["pct_unique_cosmos_hosts_with_zcosmos_sdss_desi"]) for row in z_rows])
    external_z = np.array([float(row["pct_unique_cosmos_hosts_with_external_catalog"]) for row in z_rows])
    any_z = np.array([float(row["pct_unique_cosmos_hosts_with_any_spectrum"]) for row in z_rows])
    ax.step(centers, core_z, where="mid", color="#377eb8", label="zCOSMOS + SDSS/DESI")
    ax.plot(centers, core_z, "o", color="#377eb8")
    ax.step(centers, external_z, where="mid", color="#984ea3", label="external catalogs")
    ax.plot(centers, external_z, "s", color="#984ea3")
    ax.step(centers, any_z, where="mid", color="#4daf4a", label="full")
    ax.plot(centers, any_z, "^", color="#4daf4a")
    for idx, (center, width_value, row) in enumerate(zip(centers, widths, z_rows)):
        ax.axvspan(center - width_value / 2, center + width_value / 2, color="0.5", alpha=0.035)
        label_y = max(core_z[idx], external_z[idx], any_z[idx])
        if not np.isfinite(label_y):
            label_y = 0.0
        ax.text(
            center,
            label_y + 1.0,
            f"{int(row['n_unique_cosmos_hosts_with_any_spectrum'])}/{int(row['n_unique_cosmos_hosts'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("COSMOS host redshift")
    ax.set_ylabel("Completeness [%]")
    ax.set_ylim(0, finite_max([core_z, external_z, any_z]) * 1.25)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    mode = "redshift-consistent matches" if require_redshift_consistent else "coordinate matches"
    n_total = int(summary_rows[0]["n_total_unique_cosmos_hosts"]) if summary_rows else 0
    n_found = int(summary_rows[-1]["n_unique_cosmos_hosts"])
    fig.suptitle(f"{survey_label}: full completeness {n_found}/{n_total} unique COSMOS hosts ({mode})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_edges(text: str) -> np.ndarray:
    values = np.array([float(item.strip()) for item in text.split(",") if item.strip()], dtype=float)
    if values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("--redshift-bins must contain increasing comma-separated edges")
    return values


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    provenance_path = (args.provenance or project_root / "chimera_provenance.csv").expanduser().resolve()
    candidates_path = (
        args.candidates
        or project_root
        / "spectra_workflow"
        / "outputs"
        / "sdss_desi_galaxy_spectra"
        / "galaxy_spectrum_candidates.csv"
    ).expanduser().resolve()
    zcosmos_audit_path = (
        args.zcosmos_audit
        or project_root
        / "spectra_workflow"
        / "outputs"
        / "source_match_audit"
        / "spectrum_source_match_audit.csv"
    ).expanduser().resolve()
    external_candidates_path = (
        args.external_candidates
        or project_root
        / "spectra_workflow"
        / "outputs"
        / "external_spectroscopic_catalogs"
        / "external_spectroscopic_catalog_candidates.csv"
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or project_root / "spectra_workflow" / "outputs" / "sdss_desi_galaxy_spectra" / "completeness"
    ).expanduser().resolve()
    require_redshift_consistent = not args.include_redshift_inconsistent

    hosts = load_denominator(provenance_path)
    detected_sdss_desi = load_detected_services(candidates_path, require_redshift_consistent=require_redshift_consistent)
    detected_zcosmos = load_zcosmos_services(zcosmos_audit_path)
    detected_external = load_external_catalog_services(
        external_candidates_path,
        require_redshift_consistent=require_redshift_consistent,
    )
    detected = merge_service_maps(detected_zcosmos, detected_sdss_desi, detected_external)
    z_edges = parse_edges(args.redshift_bins)
    summary_rows, z_rows = build_summary_rows(
        hosts,
        detected,
        detected_zcosmos,
        detected_sdss_desi,
        detected_external,
        z_edges,
    )

    suffix = "redshift_consistent" if require_redshift_consistent else "coordinate_only"
    write_csv(output_dir / f"spectroscopic_completeness_summary_{suffix}.csv", summary_rows)
    write_csv(output_dir / f"spectroscopic_completeness_by_redshift_{suffix}.csv", z_rows)
    plot_completeness(
        summary_rows,
        z_rows,
        output_dir / f"spectroscopic_completeness_{suffix}.png",
        survey_label=args.survey_label,
        require_redshift_consistent=require_redshift_consistent,
        dpi=args.dpi,
    )
    print(f"Wrote completeness outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
