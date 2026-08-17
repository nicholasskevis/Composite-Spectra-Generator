#!/usr/bin/env python
"""Search SDSS and DESI for spectra of Chimera COSMOS host galaxies.

The root ``chimera_provenance.csv`` identifies Chimera objects and their
COSMOS host IDs, but it may not contain host sky coordinates.  This script
therefore reads the provenance table, joins it to the active Chimera FITS table
for COSMOS host RA/Dec, and queries external spectroscopic services by sky
coordinate.

The output is a set of CSV manifests.  It does not download spectra; it only
answers the census question: which COSMOS hosts appear to have SDSS or DESI
spectra at their coordinates?

Network clients are optional:

* SDSS requires ``astroquery``.
* DESI requires ``sparclclient`` / ``sparcl``.

Install the clients in the active environment before enabling each service.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHIMERA_DIR = Path("/home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11")


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
        "--chimera-fits",
        type=Path,
        default=None,
        help=(
            "Chimera FITS table used to look up COSMOS host coordinates. "
            "Default: /home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11/chimeras-fullinfo.fits "
            "if present, otherwise project-root/data/chimeras-2023-10-11/chimeras-fullinfo.fits."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/sdss_desi_galaxy_spectra.",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=("sdss", "desi"),
        default=("sdss", "desi"),
        help="Services to query. Default: sdss desi.",
    )
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument(
        "--redshift-tolerance",
        type=float,
        default=0.02,
        help="Flag candidate as redshift-consistent when |z_match - z_host| is below this value.",
    )
    parser.add_argument("--max-candidates-per-host", type=int, default=5)
    parser.add_argument(
        "--desi-search-box-arcsec",
        type=float,
        default=None,
        help=(
            "RA/Dec half-width used for the DESI SPARCL box query. "
            "Default: max(3 arcsec, 2 * radius-arcsec). Candidates are then filtered by true separation."
        ),
    )
    parser.add_argument("--desi-find-limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between host queries, useful if a remote service throttles requests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip()


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def finite_float(value: Any) -> float:
    try:
        out = float(scalar(value))
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_int(value: Any) -> int | None:
    try:
        out = int(float(scalar(value)))
    except (TypeError, ValueError):
        return None
    return out


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


def table_value(row: Any, names: Iterable[str]) -> Any:
    colnames = getattr(row, "colnames", None)
    if colnames is None and hasattr(row, "keys"):
        colnames = list(row.keys())
    if colnames is None:
        return None
    available = {str(name).lower(): name for name in colnames}
    for name in names:
        actual = available.get(name.lower())
        if actual is not None:
            try:
                return row[actual]
            except Exception:
                return getattr(row, actual, None)
    return None


def record_value(record: Any, names: Iterable[str]) -> Any:
    if isinstance(record, dict):
        available = {str(key).lower(): key for key in record}
        for name in names:
            actual = available.get(name.lower())
            if actual is not None:
                return record[actual]
        return None
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return None


def find_chimera_fits(project_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        DEFAULT_CHIMERA_DIR / "chimeras-fullinfo.fits",
        project_root / "data" / "chimeras-2023-10-11" / "chimeras-fullinfo.fits",
        project_root / "spectra_workflow" / "data" / "chimeras-2023-10-11" / "chimeras-fullinfo.fits",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find chimeras-fullinfo.fits. Pass --chimera-fits explicitly."
    )


def provenance_host_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        cosmos_id = clean_text(row.get("cosmos_id") or row.get("ID_COSMOS") or row.get("COSMOS_ID0"))
        if cosmos_id:
            counts[cosmos_id] += 1
    return counts


def unique_hosts_from_chimera(
    provenance_rows: list[dict[str, str]],
    chimera_fits: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requested = set(provenance_host_counts(provenance_rows))
    provenance_by_host: dict[str, dict[str, str]] = {}
    for row in provenance_rows:
        cosmos_id = clean_text(row.get("cosmos_id") or row.get("ID_COSMOS") or row.get("COSMOS_ID0"))
        if cosmos_id and cosmos_id not in provenance_by_host:
            provenance_by_host[cosmos_id] = row

    required = {"ID_COSMOS", "ALPHA_J2000_GAL", "DELTA_J2000_GAL", "redshift"}
    out_by_host: dict[str, dict[str, Any]] = {}
    with fits.open(chimera_fits, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)
        missing = sorted(required.difference(names))
        if missing:
            raise KeyError(f"{chimera_fits} is missing required host coordinate columns: {missing}")

        for row in data:
            cosmos_id = clean_text(row["ID_COSMOS"])
            if requested and cosmos_id not in requested:
                continue
            if cosmos_id in out_by_host:
                continue
            ra = finite_float(row["ALPHA_J2000_GAL"])
            dec = finite_float(row["DELTA_J2000_GAL"])
            if not (math.isfinite(ra) and math.isfinite(dec)):
                continue
            prov = provenance_by_host.get(cosmos_id, {})
            host_z = finite_float(row["redshift_GAL"]) if "redshift_GAL" in names else float("nan")
            if not math.isfinite(host_z):
                host_z = finite_float(row["redshift"])
            out_by_host[cosmos_id] = {
                "cosmos_id": cosmos_id,
                "representative_chimera_id": clean_text(row["id"]) if "id" in names else prov.get("chimera_id", ""),
                "n_chimeras_in_provenance": provenance_host_counts(provenance_rows).get(cosmos_id, 0),
                "host_ra_deg": ra,
                "host_dec_deg": dec,
                "host_redshift": host_z,
            }

    missing_rows = [
        {
            "cosmos_id": cosmos_id,
            "n_chimeras_in_provenance": provenance_host_counts(provenance_rows).get(cosmos_id, 0),
            "failure_reason": "cosmos_id_not_found_in_chimera_fits",
            "chimera_fits": str(chimera_fits),
        }
        for cosmos_id in sorted(requested.difference(out_by_host))
    ]
    return list(out_by_host.values()), missing_rows


def redshift_delta(match_z: float, host_z: float) -> float:
    if not (math.isfinite(match_z) and math.isfinite(host_z)):
        return float("nan")
    return match_z - host_z


def redshift_consistent(match_z: float, host_z: float, tolerance: float) -> str:
    dz = redshift_delta(match_z, host_z)
    if not math.isfinite(dz):
        return ""
    return str(abs(dz) <= tolerance)


def query_sdss_host(
    host: dict[str, Any],
    *,
    radius_arcsec: float,
    max_candidates: int,
    redshift_tolerance: float,
    sdss: Any,
) -> tuple[list[dict[str, Any]], str]:
    coord = SkyCoord(host["host_ra_deg"] * u.deg, host["host_dec_deg"] * u.deg)
    try:
        matches = sdss.query_region(coord, radius=radius_arcsec * u.arcsec, spectro=True)
    except Exception as exc:
        return [], f"sdss_query_failed: {exc}"
    if matches is None or len(matches) == 0:
        return [], "no_sdss_match"

    ra = np.array([finite_float(table_value(row, ("ra", "RA"))) for row in matches])
    dec = np.array([finite_float(table_value(row, ("dec", "DEC"))) for row in matches])
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        return [], "sdss_missing_match_coordinates"
    coords = SkyCoord(ra=ra[valid] * u.deg, dec=dec[valid] * u.deg)
    separations = coord.separation(coords).arcsec
    valid_indices = np.flatnonzero(valid)
    order = sorted(range(len(valid_indices)), key=lambda i: float(separations[i]))

    rows: list[dict[str, Any]] = []
    for rank, ordered_idx in enumerate(order[:max_candidates], start=1):
        match_idx = int(valid_indices[ordered_idx])
        sep = float(separations[ordered_idx])
        row = matches[match_idx]
        z = finite_float(table_value(row, ("z", "Z", "redshift", "REDSHIFT")))
        plate = finite_int(table_value(row, ("plate", "PLATE", "plateID", "plateid")))
        mjd = finite_int(table_value(row, ("mjd", "MJD")))
        fiber = finite_int(table_value(row, ("fiberID", "fiberid", "fiber", "FIBER")))
        rows.append(
            {
                **host,
                "service": "sdss",
                "rank": rank,
                "match_ra_deg": finite_float(table_value(row, ("ra", "RA"))),
                "match_dec_deg": finite_float(table_value(row, ("dec", "DEC"))),
                "separation_arcsec": sep,
                "match_redshift": z,
                "match_redshift_error": "",
                "delta_redshift": redshift_delta(z, host["host_redshift"]),
                "redshift_consistent": redshift_consistent(z, host["host_redshift"], redshift_tolerance),
                "spectype": clean_text(table_value(row, ("class", "CLASS", "spectrotype", "SPECTYPE")) or ""),
                "subclass": clean_text(table_value(row, ("subclass", "SUBCLASS")) or ""),
                "data_release": clean_text(table_value(row, ("run2d", "RUN2D")) or ""),
                "spectrum_id": (
                    f"{plate}-{mjd}-{fiber}" if plate is not None and mjd is not None and fiber is not None else ""
                ),
                "plate": plate if plate is not None else "",
                "mjd": mjd if mjd is not None else "",
                "fiber": fiber if fiber is not None else "",
                "targetid": "",
                "sparcl_id": "",
            }
        )
    return rows, ""


def desi_constraints_for_host(host: dict[str, Any], box_arcsec: float) -> dict[str, list[float]]:
    ra = float(host["host_ra_deg"])
    dec = float(host["host_dec_deg"])
    dra = box_arcsec / 3600.0 / max(math.cos(math.radians(dec)), 1.0e-6)
    ddec = box_arcsec / 3600.0
    return {
        "ra": [max(0.0, ra - dra), min(360.0, ra + dra)],
        "dec": [max(-90.0, dec - ddec), min(90.0, dec + ddec)],
    }


def query_desi_host(
    host: dict[str, Any],
    *,
    radius_arcsec: float,
    box_arcsec: float,
    max_candidates: int,
    redshift_tolerance: float,
    client: Any,
    find_limit: int,
) -> tuple[list[dict[str, Any]], str]:
    outfields = [
        "sparcl_id",
        "specid",
        "ra",
        "dec",
        "redshift",
        "redshift_err",
        "spectype",
        "data_release",
    ]
    try:
        found = client.find(
            outfields=outfields,
            constraints=desi_constraints_for_host(host, box_arcsec),
            limit=find_limit,
        )
    except Exception as exc:
        return [], f"desi_query_failed: {exc}"

    records = list(getattr(found, "records", []) or [])
    if not records:
        return [], "no_desi_match"

    coord = SkyCoord(host["host_ra_deg"] * u.deg, host["host_dec_deg"] * u.deg)
    candidates: list[tuple[float, Any]] = []
    for record in records:
        ra = finite_float(record_value(record, ("ra", "RA")))
        dec = finite_float(record_value(record, ("dec", "DEC")))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        sep = float(coord.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec)
        if sep <= radius_arcsec:
            candidates.append((sep, record))
    if not candidates:
        return [], "no_desi_match_within_radius"

    rows: list[dict[str, Any]] = []
    for rank, (sep, record) in enumerate(sorted(candidates, key=lambda item: item[0])[:max_candidates], start=1):
        z = finite_float(record_value(record, ("redshift", "z", "Z")))
        rows.append(
            {
                **host,
                "service": "desi",
                "rank": rank,
                "match_ra_deg": finite_float(record_value(record, ("ra", "RA"))),
                "match_dec_deg": finite_float(record_value(record, ("dec", "DEC"))),
                "separation_arcsec": sep,
                "match_redshift": z,
                "match_redshift_error": finite_float(record_value(record, ("redshift_err", "zerr", "ZERR"))),
                "delta_redshift": redshift_delta(z, host["host_redshift"]),
                "redshift_consistent": redshift_consistent(z, host["host_redshift"], redshift_tolerance),
                "spectype": clean_text(record_value(record, ("spectype", "SPECTYPE")) or ""),
                "subclass": "",
                "data_release": clean_text(record_value(record, ("data_release", "_dr")) or ""),
                "spectrum_id": clean_text(record_value(record, ("specid", "targetid", "TARGETID")) or ""),
                "plate": "",
                "mjd": "",
                "fiber": "",
                "targetid": clean_text(record_value(record, ("targetid", "TARGETID")) or ""),
                "sparcl_id": clean_text(record_value(record, ("sparcl_id",)) or ""),
            }
        )
    return rows, ""


def best_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["cosmos_id"]), []).append(row)

    best: list[dict[str, Any]] = []
    for cosmos_id in sorted(grouped):
        rows = grouped[cosmos_id]
        rows.sort(
            key=lambda row: (
                row.get("redshift_consistent") != "True",
                float(row.get("separation_arcsec") or float("inf")),
                row.get("service") != "desi",
            )
        )
        best.append({**rows[0], "best_match": True})
    return best


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    provenance_path = (args.provenance or project_root / "chimera_provenance.csv").expanduser().resolve()
    chimera_fits = find_chimera_fits(project_root, args.chimera_fits)
    output_dir = (
        args.output_dir
        or project_root / "spectra_workflow" / "outputs" / "sdss_desi_galaxy_spectra"
    ).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Provenance: {provenance_path}")
    print(f"Chimera FITS: {chimera_fits}")
    print(f"Output dir: {output_dir}")
    provenance_rows = read_csv(provenance_path)
    hosts, missing_coordinate_rows = unique_hosts_from_chimera(provenance_rows, chimera_fits)
    if missing_coordinate_rows:
        print(
            f"Warning: {len(missing_coordinate_rows)} provenance COSMOS IDs were not found in {chimera_fits}; "
            "writing missing_host_coordinates.csv and querying the matched hosts."
        )
    hosts = hosts[args.start_index :]
    if args.limit is not None:
        hosts = hosts[: args.limit]
    print(f"Unique COSMOS hosts to query: {len(hosts)}")

    sdss = None
    if "sdss" in args.services:
        try:
            from astroquery.sdss import SDSS
        except ImportError as exc:
            raise SystemExit("SDSS querying requires astroquery: python -m pip install astroquery") from exc
        sdss = SDSS

    desi_client = None
    if "desi" in args.services:
        try:
            from sparcl.client import SparclClient
        except ImportError as exc:
            raise SystemExit("DESI/SPARCL querying requires sparclclient: python -m pip install sparclclient") from exc
        desi_client = SparclClient(announcement=False)

    box_arcsec = args.desi_search_box_arcsec or max(3.0, 2.0 * args.radius_arcsec)
    candidate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    service_counts: Counter[str] = Counter()

    for idx, host in enumerate(hosts, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"Querying host {idx}/{len(hosts)}; candidates={len(candidate_rows)}")

        host_had_candidate = False
        if sdss is not None:
            rows, failure = query_sdss_host(
                host,
                radius_arcsec=args.radius_arcsec,
                max_candidates=args.max_candidates_per_host,
                redshift_tolerance=args.redshift_tolerance,
                sdss=sdss,
            )
            candidate_rows.extend(rows)
            service_counts["sdss_candidates"] += len(rows)
            host_had_candidate = host_had_candidate or bool(rows)
            if failure:
                failure_rows.append({**host, "service": "sdss", "failure_reason": failure})

        if desi_client is not None:
            rows, failure = query_desi_host(
                host,
                radius_arcsec=args.radius_arcsec,
                box_arcsec=box_arcsec,
                max_candidates=args.max_candidates_per_host,
                redshift_tolerance=args.redshift_tolerance,
                client=desi_client,
                find_limit=args.desi_find_limit,
            )
            candidate_rows.extend(rows)
            service_counts["desi_candidates"] += len(rows)
            host_had_candidate = host_had_candidate or bool(rows)
            if failure:
                failure_rows.append({**host, "service": "desi", "failure_reason": failure})

        if not host_had_candidate:
            service_counts["hosts_without_any_candidate"] += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    best = best_rows(candidate_rows)
    summary = {
        "provenance": str(provenance_path),
        "chimera_fits": str(chimera_fits),
        "n_provenance_rows": len(provenance_rows),
        "n_unique_hosts_queried": len(hosts),
        "n_unique_hosts_missing_coordinates": len(missing_coordinate_rows),
        "services": list(args.services),
        "radius_arcsec": args.radius_arcsec,
        "redshift_tolerance": args.redshift_tolerance,
        "n_candidates": len(candidate_rows),
        "n_best_matches": len(best),
        "n_hosts_with_any_candidate": len({str(row["cosmos_id"]) for row in candidate_rows}),
        "n_hosts_with_redshift_consistent_candidate": len(
            {str(row["cosmos_id"]) for row in candidate_rows if row.get("redshift_consistent") == "True"}
        ),
        "service_counts": dict(service_counts),
    }

    write_csv(output_dir / "galaxy_spectrum_candidates.csv", candidate_rows)
    write_csv(output_dir / "best_galaxy_spectrum_matches.csv", best)
    write_csv(output_dir / "galaxy_spectrum_query_failures.csv", failure_rows)
    write_csv(output_dir / "missing_host_coordinates.csv", missing_coordinate_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
