#!/usr/bin/env python
"""Download missing SDSS DR7Q spectra listed by the source-match audit.

The input is usually ``missing_dr7q_objects.csv`` or ``missing_dr7q_spectra.csv``
from ``audit_spectrum_source_matches.py``.  Files are written with the BOSS-style
name expected by the composite-spectrum builder:

    spec-PLATE-MJD-FIBER.fits

The downloader is deliberately conservative: it validates that downloaded FITS
files contain a table with ``flux`` and ``loglam``/``lambda`` columns before
keeping them.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DATA_DIR = Path("/home/nicho/GRAHSP_my/data")
DEFAULT_URL_TEMPLATES = [
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
    "https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fiber:04d}.fits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data-dir/dr7q_spectra.",
    )
    parser.add_argument(
        "--url-template",
        action="append",
        default=None,
        help=(
            "URL template. May be repeated. Available fields: plate, mjd, fiber. "
            "Default tries several public SDSS SAS releases."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned URL table without downloading files.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
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


def parse_int(row: dict[str, str], *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return int(float(value))
    return None


def unique_requested_spectra(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        plate = parse_int(row, "dr7q_plate", "plate", "qso_catalog_plate")
        mjd = parse_int(row, "dr7q_mjd", "mjd", "qso_catalog_smjd")
        fiber = parse_int(row, "dr7q_fiber", "fiber", "qso_catalog_fiber")
        if plate is None or mjd is None or fiber is None:
            continue
        key = (plate, mjd, fiber)
        if key in seen:
            continue
        seen.add(key)
        out.append({"plate": plate, "mjd": mjd, "fiber": fiber})
    return out


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


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).expanduser().resolve()
    missing_csv = (
        args.missing_csv
        or WORKFLOW_ROOT / "outputs" / "source_match_audit" / "missing_dr7q_objects.csv"
    ).expanduser().resolve()
    output_dir = (args.output_dir or data_dir / "dr7q_spectra").expanduser().resolve()
    templates = args.url_template or DEFAULT_URL_TEMPLATES

    rows = unique_requested_spectra(read_rows(missing_csv))
    if args.limit is not None:
        rows = rows[: args.limit]

    log_rows: list[dict[str, Any]] = []
    print(f"Missing CSV: {missing_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Spectra to inspect: {len(rows)}")

    for idx, row in enumerate(rows, start=1):
        plate = row["plate"]
        mjd = row["mjd"]
        fiber = row["fiber"]
        filename = f"spec-{plate}-{mjd}-{fiber:04d}.fits"
        destination = output_dir / filename
        payload: dict[str, Any] = {
            "plate": plate,
            "mjd": mjd,
            "fiber": fiber,
            "destination": str(destination),
        }
        if destination.exists() and not args.overwrite:
            payload["status"] = "exists"
            log_rows.append(payload)
            continue
        if args.dry_run:
            payload["status"] = "planned"
            payload["candidate_urls"] = " ".join(template.format(plate=plate, mjd=mjd, fiber=fiber) for template in templates)
            log_rows.append(payload)
            continue
        for template in templates:
            url = template.format(plate=plate, mjd=mjd, fiber=fiber)
            try:
                download_url(url, destination, timeout=args.timeout)
                payload["status"] = "downloaded"
                payload["url"] = url
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                payload.setdefault("failed_urls", [])
                payload["failed_urls"].append(f"{url} :: {exc}")
        else:
            payload["status"] = "failed"
        log_rows.append(payload)
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(rows)}")

    report_path = output_dir / "download_missing_dr7q_spectra_report.csv"
    write_csv(report_path, log_rows)
    counts: dict[str, int] = {}
    for row in log_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Status counts: {counts}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
