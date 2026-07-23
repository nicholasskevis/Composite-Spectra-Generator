#!/usr/bin/env python
"""Search SDSS for replacement QSO spectra by sky coordinate.

This script reads missing-QSO rows from the source-match audit and queries SDSS
near each quasar coordinate.  It writes candidate matches and a builder override
table that can be passed to ``build_all_chimera_composite_spectra.py`` with
``--qso-spectrum-overrides`` after the replacement spectra are downloaded.

Requires ``astroquery`` and network access.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DATA_DIR = Path("/home/nicho/GRAHSP_my/data")
DEFAULT_URL_TEMPLATES = [
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--missing-objects-csv",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--max-candidates-per-object", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--url-template", action="append", default=None)
    return parser.parse_args()


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


def parse_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def table_value(row: Any, names: tuple[str, ...]) -> Any:
    available = {name.lower(): name for name in row.colnames}
    for name in names:
        actual = available.get(name.lower())
        if actual is not None:
            return row[actual]
    return None


def table_column(table: Any, names: tuple[str, ...]) -> Any:
    available = {name.lower(): name for name in table.colnames}
    for name in names:
        actual = available.get(name.lower())
        if actual is not None:
            return table[actual]
    raise KeyError(f"none of columns {names} found; available columns: {table.colnames}")


def validate_builder_spectrum(path: Path) -> None:
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None or not hasattr(hdul[1].data, "names"):
            raise ValueError("missing HDU 1 binary table")
        names = set(hdul[1].data.names)
        if "flux" not in names:
            raise ValueError("missing flux column")
        if "loglam" not in names and "lambda" not in names:
            raise ValueError("missing loglam/lambda column")


def download_url(url: str, destination: Path, *, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".fits", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            tmp_path.write_bytes(response.read())
        validate_builder_spectrum(tmp_path)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def candidate_url(templates: list[str], plate: int, mjd: int, fiber: int) -> list[str]:
    return [template.format(plate=plate, mjd=mjd, fiber=fiber) for template in templates]


def main() -> int:
    args = parse_args()
    try:
        from astroquery.sdss import SDSS
    except ImportError as exc:
        raise SystemExit("astroquery is required: python -m pip install astroquery") from exc

    project_root = args.project_root.expanduser().resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).expanduser().resolve()
    missing_csv = (
        args.missing_objects_csv
        or WORKFLOW_ROOT / "outputs" / "source_match_audit" / "missing_dr7q_objects.csv"
    ).expanduser().resolve()
    output_dir = (args.output_dir or WORKFLOW_ROOT / "outputs" / "sdss_replacement_spectra").expanduser().resolve()
    spectrum_dir = data_dir / "dr7q_spectra"
    templates = args.url_template or DEFAULT_URL_TEMPLATES

    rows = read_csv(missing_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    candidate_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    print(f"Missing objects: {missing_csv}")
    print(f"Objects to query: {len(rows)}")
    print(f"Output dir: {output_dir}")

    for idx, row in enumerate(rows, start=1):
        ra = parse_float(row.get("qso_ra_deg") or row.get("qso_catalog_ra_deg"))
        dec = parse_float(row.get("qso_dec_deg") or row.get("qso_catalog_dec_deg"))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            failure_rows.append({**row, "failure_reason": "missing_qso_coordinates"})
            continue
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        try:
            matches = SDSS.query_region(
                coord,
                radius=args.radius_arcsec * u.arcsec,
                spectro=True,
            )
        except Exception as exc:
            failure_rows.append({**row, "failure_reason": f"query_failed: {exc}"})
            continue
        if matches is None or len(matches) == 0:
            failure_rows.append({**row, "failure_reason": "no_sdss_spectrum_within_radius"})
            continue

        try:
            match_ra = table_column(matches, ("ra", "RA"))
            match_dec = table_column(matches, ("dec", "DEC"))
        except KeyError as exc:
            failure_rows.append({**row, "failure_reason": f"query_missing_coordinate_columns: {exc}"})
            continue
        matched_coords = SkyCoord(ra=match_ra * u.deg, dec=match_dec * u.deg)
        separations = coord.separation(matched_coords).arcsec
        order = sorted(range(len(matches)), key=lambda i: float(separations[i]))
        best_override_written = False

        for rank, match_idx in enumerate(order[: args.max_candidates_per_object], start=1):
            match = matches[match_idx]
            plate = parse_int(table_value(match, ("plate", "PLATE", "plateID", "plateid")))
            mjd = parse_int(table_value(match, ("mjd", "MJD")))
            fiber = parse_int(table_value(match, ("fiberID", "fiberid", "fiber", "FIBER")))
            if plate is None or mjd is None or fiber is None:
                continue
            destination = spectrum_dir / f"spec-{plate}-{mjd}-{fiber:04d}.fits"
            urls = candidate_url(templates, plate, mjd, fiber)
            download_status = "not_requested"
            used_url = ""
            if args.download:
                if destination.exists() and not args.overwrite:
                    download_status = "exists"
                else:
                    for url in urls:
                        try:
                            download_url(url, destination, timeout=args.timeout)
                            download_status = "downloaded"
                            used_url = url
                            break
                        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
                            continue
                    else:
                        download_status = "download_failed"

            payload = {
                "chimera_id": row.get("chimera_id", ""),
                "dr7q_requested_key": row.get("dr7q_requested_key", row.get("dr7q_spectrum_id", "")),
                "requested_plate": row.get("dr7q_plate", ""),
                "requested_mjd": row.get("dr7q_mjd", ""),
                "requested_fiber": row.get("dr7q_fiber", ""),
                "rank": rank,
                "sep_arcsec": float(separations[match_idx]),
                "replacement_plate": plate,
                "replacement_mjd": mjd,
                "replacement_fiber": fiber,
                "spectrum_path": str(destination),
                "download_status": download_status,
                "download_url": used_url,
                "candidate_urls": " ".join(urls),
            }
            candidate_rows.append(payload)
            if rank == 1 and not best_override_written and download_status in {"not_requested", "exists", "downloaded"}:
                override_rows.append(payload)
                best_override_written = True

        if idx % 100 == 0:
            print(f"Queried {idx}/{len(rows)}")

    write_csv(output_dir / "sdss_replacement_candidates.csv", candidate_rows)
    write_csv(output_dir / "qso_spectrum_overrides.csv", override_rows)
    write_csv(output_dir / "sdss_replacement_failures.csv", failure_rows)
    print(f"Candidates: {len(candidate_rows)}")
    print(f"Overrides: {len(override_rows)}")
    print(f"Failures: {len(failure_rows)}")
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
