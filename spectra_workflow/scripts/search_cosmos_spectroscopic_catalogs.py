#!/usr/bin/env python
"""Search external spectroscopic-redshift catalogs for Chimera COSMOS hosts.

This is a discovery/completeness script, not a spectrum downloader.  It searches
catalog services by COSMOS host coordinates and writes candidate spectroscopic
matches.  A match here means "this host appears in a spectroscopic catalog"; it
does not guarantee that a flux-calibrated 1D spectrum is publicly available or
usable for synthetic Chimera spectra.

Supported services:

* VizieR catalogs, via ``astroquery.vizier``.
* NED, via ``astroquery.ipac.ned``.

The denominator is the unique COSMOS host list from ``chimera_provenance.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep this list conservative and editable.  These are discovery catalogs, not
# guaranteed spectrum-file providers.
DEFAULT_VIZIER_CATALOGS = {
    "zCOSMOS_Lilly2007": "J/ApJS/172/70/zcosmos3",
    "COSMOS_Ilbert2013_zspec": "J/ApJS/206/8",
    "C3R2_KMOS_Euclid2020": "J/A+A/642/A192",
}


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
        "--output-dir",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/external_spectroscopic_catalogs.",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=("vizier", "ned"),
        default=("vizier",),
        help="Services to query. Default: vizier. Add ned for NED coordinate searches.",
    )
    parser.add_argument(
        "--vizier-catalog",
        action="append",
        default=None,
        metavar="LABEL=CATALOG_ID",
        help=(
            "Additional or replacement VizieR catalogs. If supplied, only these catalogs are used. "
            "Example: --vizier-catalog zCOSMOS=J/ApJS/172/70/zcosmos3"
        ),
    )
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--redshift-tolerance", type=float, default=0.02)
    parser.add_argument("--max-candidates-per-host-catalog", type=int, default=3)
    parser.add_argument("--vizier-row-limit", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip()


def finite_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if np.ma.is_masked(value):
        return float("nan")
    if isinstance(value, np.ma.MaskedArray):
        if value.size == 0 or bool(np.any(value.mask)):
            return float("nan")
        value = value.item() if value.size == 1 else value
    if clean_text(value).lower() in {"", "--", "nan", "masked"}:
        return float("nan")
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


def load_hosts(provenance_path: Path) -> list[dict[str, Any]]:
    rows = read_csv(provenance_path)
    by_host: dict[str, dict[str, Any]] = {}
    for row in rows:
        cosmos_id = host_id(row)
        if not cosmos_id:
            continue
        entry = by_host.setdefault(
            cosmos_id,
            {
                "cosmos_id": cosmos_id,
                "n_chimeras": 0,
                "representative_chimera_id": clean_text(row.get("chimera_id", "")),
                "host_ra_deg": finite_float(row.get("galaxy_ra_deg")),
                "host_dec_deg": finite_float(row.get("galaxy_dec_deg")),
                "host_redshift": host_redshift(row),
            },
        )
        entry["n_chimeras"] += 1
        for key in ("host_ra_deg", "host_dec_deg"):
            if not math.isfinite(float(entry[key])):
                entry[key] = finite_float(row.get("galaxy_ra_deg" if key == "host_ra_deg" else "galaxy_dec_deg"))
        if not math.isfinite(float(entry["host_redshift"])):
            entry["host_redshift"] = host_redshift(row)
    return [
        row
        for row in by_host.values()
        if math.isfinite(float(row["host_ra_deg"])) and math.isfinite(float(row["host_dec_deg"]))
    ]


def parse_vizier_catalogs(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_VIZIER_CATALOGS)
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--vizier-catalog must be LABEL=CATALOG_ID, got {item!r}")
        label, catalog = item.split("=", 1)
        out[label.strip()] = catalog.strip()
    return out


def table_column_names(row: Any) -> list[str]:
    if hasattr(row, "colnames"):
        return list(row.colnames)
    if hasattr(row, "keys"):
        return list(row.keys())
    return []


def row_value(row: Any, names: Iterable[str]) -> Any:
    available = {name.lower(): name for name in table_column_names(row)}
    for name in names:
        actual = available.get(name.lower())
        if actual is not None:
            try:
                return row[actual]
            except Exception:
                return getattr(row, actual, None)
    return None


def first_existing_column(row: Any, names: Iterable[str]) -> str:
    available = {name.lower(): name for name in table_column_names(row)}
    for name in names:
        actual = available.get(name.lower())
        if actual is not None:
            return str(actual)
    return ""


def match_redshift_delta(match_z: float, host_z: float) -> float:
    if not (math.isfinite(match_z) and math.isfinite(host_z)):
        return float("nan")
    return match_z - host_z


def redshift_flag(match_z: float, host_z: float, tolerance: float) -> str:
    dz = match_redshift_delta(match_z, host_z)
    if not math.isfinite(dz):
        return ""
    return str(abs(dz) <= tolerance)


def candidate_row(
    host: dict[str, Any],
    *,
    service: str,
    catalog_label: str,
    catalog_id: str,
    rank: int,
    match_ra: float,
    match_dec: float,
    match_z: float,
    sep_arcsec: float,
    source_id: str,
    quality: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        **host,
        "service": service,
        "catalog_label": catalog_label,
        "catalog_id": catalog_id,
        "rank": rank,
        "match_ra_deg": match_ra,
        "match_dec_deg": match_dec,
        "separation_arcsec": sep_arcsec,
        "match_redshift": match_z,
        "delta_redshift": match_redshift_delta(match_z, host["host_redshift"]),
        "redshift_consistent": redshift_flag(match_z, host["host_redshift"], extra.get("redshift_tolerance", 0.02) if extra else 0.02),
        "source_id": source_id,
        "quality": quality,
    }
    if extra:
        row.update({key: value for key, value in extra.items() if key != "redshift_tolerance"})
    return row


def query_vizier_catalog(
    host: dict[str, Any],
    *,
    catalog_label: str,
    catalog_id: str,
    radius_arcsec: float,
    max_candidates: int,
    redshift_tolerance: float,
    vizier: Any,
) -> tuple[list[dict[str, Any]], str]:
    coord = SkyCoord(host["host_ra_deg"] * u.deg, host["host_dec_deg"] * u.deg)
    try:
        tables = vizier.query_region(coord, radius=radius_arcsec * u.arcsec, catalog=catalog_id)
    except Exception as exc:
        return [], f"vizier_query_failed: {exc}"
    if not tables:
        return [], "no_vizier_match"

    rows: list[dict[str, Any]] = []
    for table in tables:
        for table_row in table:
            ra = finite_float(row_value(table_row, ("RAJ2000", "RA_ICRS", "RA", "_RAJ2000", "alpha", "ALPHA_J2000")))
            dec = finite_float(row_value(table_row, ("DEJ2000", "DE_ICRS", "DEC", "_DEJ2000", "delta", "DELTA_J2000")))
            if not (math.isfinite(ra) and math.isfinite(dec)):
                continue
            sep = float(coord.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec)
            z = finite_float(row_value(table_row, ("zspec", "z_spec", "z", "Z", "REDSHIFT", "zsp", "z_best", "zpbest")))
            source_id = clean_text(row_value(table_row, ("ID", "Name", "zCOSMOS", "OBJECT_ID", "Seq", "recno", "OBJID")) or "")
            quality = clean_text(row_value(table_row, ("CC", "Q", "Qf", "q_zsp", "zquality", "Flag", "Quality")) or "")
            rows.append(
                candidate_row(
                    host,
                    service="vizier",
                    catalog_label=catalog_label,
                    catalog_id=catalog_id,
                    rank=0,
                    match_ra=ra,
                    match_dec=dec,
                    match_z=z,
                    sep_arcsec=sep,
                    source_id=source_id,
                    quality=quality,
                    extra={
                        "redshift_tolerance": redshift_tolerance,
                        "ra_column": first_existing_column(
                            table_row, ("RAJ2000", "RA_ICRS", "RA", "_RAJ2000", "alpha", "ALPHA_J2000")
                        ),
                        "dec_column": first_existing_column(
                            table_row, ("DEJ2000", "DE_ICRS", "DEC", "_DEJ2000", "delta", "DELTA_J2000")
                        ),
                        "redshift_column": first_existing_column(
                            table_row, ("zspec", "z_spec", "z", "Z", "REDSHIFT", "zsp", "z_best", "zpbest")
                        ),
                    },
                )
            )
    rows.sort(key=lambda row: float(row["separation_arcsec"]))
    for rank, row in enumerate(rows[:max_candidates], start=1):
        row["rank"] = rank
    return rows[:max_candidates], "" if rows else "no_vizier_candidate_with_coordinates"


def query_ned(
    host: dict[str, Any],
    *,
    radius_arcsec: float,
    max_candidates: int,
    redshift_tolerance: float,
    ned: Any,
) -> tuple[list[dict[str, Any]], str]:
    coord = SkyCoord(host["host_ra_deg"] * u.deg, host["host_dec_deg"] * u.deg)
    try:
        table = ned.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as exc:
        return [], f"ned_query_failed: {exc}"
    if table is None or len(table) == 0:
        return [], "no_ned_match"
    rows = []
    for table_row in table:
        ra = finite_float(row_value(table_row, ("RA", "RA(deg)", "RAJ2000")))
        dec = finite_float(row_value(table_row, ("DEC", "DEC(deg)", "DEJ2000")))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        sep = float(coord.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec)
        z = finite_float(row_value(table_row, ("Redshift", "z", "Z")))
        source_id = clean_text(row_value(table_row, ("Object Name", "Object", "Name")) or "")
        rows.append(
            candidate_row(
                host,
                service="ned",
                catalog_label="NED",
                catalog_id="NED",
                rank=0,
                match_ra=ra,
                match_dec=dec,
                match_z=z,
                sep_arcsec=sep,
                source_id=source_id,
                quality="",
                extra={"redshift_tolerance": redshift_tolerance},
            )
        )
    rows.sort(key=lambda row: float(row["separation_arcsec"]))
    for rank, row in enumerate(rows[:max_candidates], start=1):
        row["rank"] = rank
    return rows[:max_candidates], "" if rows else "no_ned_candidate_with_coordinates"


def best_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["cosmos_id"])].append(row)
    out = []
    for cosmos_id, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                row.get("redshift_consistent") != "True",
                float(row.get("separation_arcsec") or float("inf")),
            )
        )
        out.append({**rows[0], "best_match": True})
    return sorted(out, key=lambda row: str(row["cosmos_id"]))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    provenance_path = (args.provenance or project_root / "chimera_provenance.csv").expanduser().resolve()
    output_dir = (
        args.output_dir
        or project_root / "spectra_workflow" / "outputs" / "external_spectroscopic_catalogs"
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    vizier_catalogs = parse_vizier_catalogs(args.vizier_catalog)
    hosts = load_hosts(provenance_path)
    hosts = hosts[args.start_index :]
    if args.limit is not None:
        hosts = hosts[: args.limit]

    vizier = None
    if "vizier" in args.services:
        try:
            from astroquery.vizier import Vizier
        except ImportError as exc:
            raise SystemExit("VizieR querying requires astroquery: python -m pip install astroquery") from exc
        vizier = Vizier(columns=["**"], row_limit=args.vizier_row_limit)

    ned = None
    if "ned" in args.services:
        try:
            from astroquery.ipac.ned import Ned
        except ImportError as exc:
            raise SystemExit("NED querying requires astroquery: python -m pip install astroquery") from exc
        ned = Ned

    print(f"Provenance: {provenance_path}")
    print(f"Hosts to query: {len(hosts)}")
    print(f"Services: {', '.join(args.services)}")
    print(f"Output dir: {output_dir}")

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for idx, host in enumerate(hosts, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"Querying host {idx}/{len(hosts)}; candidates={len(candidates)}")
        if vizier is not None:
            for label, catalog_id in vizier_catalogs.items():
                rows, failure = query_vizier_catalog(
                    host,
                    catalog_label=label,
                    catalog_id=catalog_id,
                    radius_arcsec=args.radius_arcsec,
                    max_candidates=args.max_candidates_per_host_catalog,
                    redshift_tolerance=args.redshift_tolerance,
                    vizier=vizier,
                )
                candidates.extend(rows)
                counts[f"vizier_{label}_candidates"] += len(rows)
                if failure:
                    failures.append({**host, "service": "vizier", "catalog_label": label, "failure_reason": failure})
        if ned is not None:
            rows, failure = query_ned(
                host,
                radius_arcsec=args.radius_arcsec,
                max_candidates=args.max_candidates_per_host_catalog,
                redshift_tolerance=args.redshift_tolerance,
                ned=ned,
            )
            candidates.extend(rows)
            counts["ned_candidates"] += len(rows)
            if failure:
                failures.append({**host, "service": "ned", "catalog_label": "NED", "failure_reason": failure})
        if args.sleep > 0:
            time.sleep(args.sleep)

    best = best_rows(candidates)
    consistent_hosts = {str(row["cosmos_id"]) for row in candidates if row.get("redshift_consistent") == "True"}
    any_hosts = {str(row["cosmos_id"]) for row in candidates}
    summary = {
        "provenance": str(provenance_path),
        "services": list(args.services),
        "vizier_catalogs": vizier_catalogs,
        "radius_arcsec": args.radius_arcsec,
        "redshift_tolerance": args.redshift_tolerance,
        "n_unique_hosts_queried": len(hosts),
        "n_candidates": len(candidates),
        "n_best_matches": len(best),
        "n_hosts_with_any_candidate": len(any_hosts),
        "n_hosts_with_redshift_consistent_candidate": len(consistent_hosts),
        "service_counts": dict(counts),
        "note": "Catalog matches do not guarantee downloadable flux-calibrated spectra.",
    }

    write_csv(output_dir / "external_spectroscopic_catalog_candidates.csv", candidates)
    write_csv(output_dir / "best_external_spectroscopic_catalog_matches.csv", best)
    write_csv(output_dir / "external_spectroscopic_catalog_failures.csv", failures)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
