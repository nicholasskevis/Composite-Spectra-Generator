#!/usr/bin/env python
"""Audit Chimera source-spectrum availability before building composites.

This script rebuilds the matching step from the raw inputs:

* Chimera provenance rows
* optional active ``fit_manifest.csv`` sample
* explicit zCOSMOS alpha/delta match table
* zCOSMOS/CESAM FITS headers and readme mappings
* SDSS DR7Q spectrum filenames

It does not build spectra.  It only reports which Chimera objects have the
source galaxy and QSO spectra needed to build a composite spectrum.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

from build_all_chimera_composite_spectra import (
    DEFAULT_EXTERNAL_DATA_DIR,
    REPO_ROOT,
    WORKFLOW_ROOT,
    build_dr7q_spectrum_index,
    build_galaxy_spectrum_index,
    build_zcosmos_match_index,
    find_zcosmos_matches_path,
    load_fit_manifest_ids,
    load_provenance,
    merge_galaxy_indices,
)

DEFAULT_CHIMERA_DIR = Path("/home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing dr7q_spectra, zCOSMOS_data, and COSMOS2015. "
            "Default: /home/nicho/GRAHSP_my/data if present, otherwise project-root/data. "
            "This does not choose the Chimera FITS catalog; use --chimera-dir for that."
        ),
    )
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument(
        "--use-existing-provenance",
        action="store_true",
        help="Use an existing chimera_provenance.csv instead of rebuilding provenance from the Chimera FITS file.",
    )
    parser.add_argument(
        "--chimera-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Chimera FITS files. Default: "
            "/home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11 if present, "
            "otherwise data-dir/chimeras-2023-10-11."
        ),
    )
    parser.add_argument(
        "--chimera-fits",
        type=Path,
        default=None,
        help="Chimera FITS file used to rebuild provenance. Default: chimera-dir/chimeras-fullinfo.fits.",
    )
    parser.add_argument("--fit-manifest", type=Path, default=None)
    parser.add_argument("--ignore-fit-manifest", action="store_true")
    parser.add_argument("--zcosmos-matches", type=Path, default=None)
    parser.add_argument(
        "--dr7q-catalog",
        type=Path,
        default=None,
        help="DR7Q catalog FITS file. Default: data-dir/dr7q_photometry/dr7qso.fit.",
    )
    parser.add_argument(
        "--coord-match-arcsec",
        type=float,
        default=1.0,
        help="Maximum separation for fallback sky-coordinate galaxy matching. Default: 1 arcsec.",
    )
    parser.add_argument(
        "--disable-coordinate-fallback",
        action="store_true",
        help="Disable fallback matching from Chimera galaxy RA/Dec to zCOSMOS FITS RA/Dec.",
    )
    parser.add_argument(
        "--qso-coord-match-arcsec",
        type=float,
        default=1.0,
        help="Maximum separation for fallback sky-coordinate DR7Q matching. Default: 1 arcsec.",
    )
    parser.add_argument(
        "--disable-qso-coordinate-fallback",
        action="store_true",
        help="Disable fallback matching from Chimera QSO RA/Dec to downloaded DR7Q spectrum RA/Dec.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: spectra_workflow/outputs/source_match_audit.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
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
    return out if np.isfinite(out) else float("nan")


def normalize_qso_name(value: Any) -> str:
    text = clean_text(value).upper()
    text = re.sub(r"^SDSS\s*", "", text)
    text = re.sub(r"^J", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def rebuild_provenance_from_chimera(path: Path) -> list[dict[str, Any]]:
    required = {
        "id",
        "ID_COSMOS",
        "redshift",
        "chimera_QSO_weight",
        "Plate_QSO",
        "MJD_QSO",
        "Fiber_QSO",
        "z_QSO",
    }
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.names)
        missing = sorted(required.difference(names))
        if missing:
            raise KeyError(f"{path} is missing required Chimera provenance columns: {missing}")
        out: list[dict[str, Any]] = []
        for row in data:
            chimera_id = clean_text(row["id"])
            qso_name = ""
            for candidate in ("SDSS_QSO", "ID_SDSS"):
                if candidate in names:
                    qso_name = clean_text(row[candidate])
                    if qso_name:
                        break
            plate = int(scalar(row["Plate_QSO"]))
            mjd = int(scalar(row["MJD_QSO"]))
            fiber = int(scalar(row["Fiber_QSO"]))
            out.append(
                {
                    "chimera_id": chimera_id,
                    "cosmos_id": int(scalar(row["ID_COSMOS"])),
                    "galaxy_ra_deg": float(scalar(row["ALPHA_J2000_GAL"])) if "ALPHA_J2000_GAL" in names else np.nan,
                    "galaxy_dec_deg": float(scalar(row["DELTA_J2000_GAL"])) if "DELTA_J2000_GAL" in names else np.nan,
                    "qso_ra_deg": float(scalar(row["RAJ2000_QSO"])) if "RAJ2000_QSO" in names else np.nan,
                    "qso_dec_deg": float(scalar(row["DEJ2000_QSO"])) if "DEJ2000_QSO" in names else np.nan,
                    "dr7q_name": qso_name,
                    "dr7q_plate": plate,
                    "dr7q_mjd": mjd,
                    "dr7q_fiber": fiber,
                    "dr7q_spectrum_id": f"{plate}-{mjd}-{fiber}",
                    "chimera_redshift": float(scalar(row["redshift"])),
                    "dr7q_redshift": float(scalar(row["z_QSO"])),
                    "chimera_qso_weight": float(scalar(row["chimera_QSO_weight"])),
                }
            )
    return out


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


def first_path(paths: list[Path]) -> str:
    return str(paths[0]) if paths else ""


def source_label(explicit_paths: list[Path], fallback_paths: list[Path], coordinate_paths: list[Path]) -> str:
    explicit = bool(explicit_paths)
    fallback = bool(fallback_paths)
    coordinate = bool(coordinate_paths)
    if explicit and fallback:
        return "explicit_and_header"
    if explicit:
        return "explicit_zcosmos_match"
    if fallback:
        return "header_or_readme"
    if coordinate:
        return "coordinate_fallback"
    return "none"


def status_label(has_galaxy: bool, has_dr7q: bool) -> str:
    if has_galaxy and has_dr7q:
        return "matched"
    if has_galaxy:
        return "missing_dr7q"
    if has_dr7q:
        return "missing_galaxy"
    return "missing_both"


def qso_source_label(source: str) -> str:
    return source or "none"


def qso_missing_reason(has_dr7q: bool, catalog_entry: dict[str, Any]) -> str:
    if has_dr7q:
        return ""
    if catalog_entry:
        return "catalog_match_but_spectrum_not_downloaded_locally"
    return "no_dr7q_catalog_match"


def build_zcosmos_coordinate_index(directories: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for directory in directories:
        if not directory.exists():
            diagnostics.append(f"missing optional directory: {directory}")
            continue
        for path in sorted(directory.glob("*.fits")):
            if path.name.endswith(":Zone.Identifier"):
                continue
            try:
                with fits.open(path, memmap=True) as hdul:
                    header = hdul[1].header if len(hdul) > 1 else hdul[0].header
                    ra = header.get("RA", hdul[0].header.get("RA"))
                    dec = header.get("DEC", hdul[0].header.get("DEC"))
                    if ra is None or dec is None:
                        diagnostics.append(f"missing RA/DEC header in {path}")
                        continue
                    rows.append(
                        {
                            "path": path,
                            "ra_deg": float(ra),
                            "dec_deg": float(dec),
                            "object": clean_text(header.get("OBJECT", hdul[0].header.get("OBJECT", ""))),
                        }
                    )
            except Exception as exc:
                diagnostics.append(f"could not inspect coordinates for {path}: {exc}")
    return rows, diagnostics


def coordinate_match(
    ra_deg: Any,
    dec_deg: Any,
    zcosmos_coords: SkyCoord | None,
    zcosmos_rows: list[dict[str, Any]],
    *,
    max_sep_arcsec: float,
) -> tuple[list[Path], float, str]:
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
    except (TypeError, ValueError):
        return [], float("nan"), ""
    if not (np.isfinite(ra) and np.isfinite(dec)) or zcosmos_coords is None or not zcosmos_rows:
        return [], float("nan"), ""
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    idx, sep2d, _ = target.match_to_catalog_sky(zcosmos_coords)
    idx_int = int(np.asarray(idx).reshape(-1)[0])
    sep_arcsec = float(np.asarray(sep2d.arcsec).reshape(-1)[0])
    if sep_arcsec <= max_sep_arcsec:
        row = zcosmos_rows[idx_int]
        return [row["path"]], sep_arcsec, row.get("object", "")
    return [], sep_arcsec, ""


def build_dr7q_catalog_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int, int], dict[str, Any]], list[str]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    diagnostics: list[str] = []
    if not path.is_file():
        return by_name, by_key, [f"missing DR7Q catalog: {path}"]

    required = {"SDSSJ", "RA", "DEC", "PLATE", "SMJD", "RMJD", "FIBER"}
    try:
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            names = set(data.names)
            missing = sorted(required.difference(names))
            if missing:
                return by_name, by_key, [f"{path} missing DR7Q catalog columns: {missing}"]
            for row in data:
                plate = int(scalar(row["PLATE"]))
                smjd = int(scalar(row["SMJD"]))
                rmjd = int(scalar(row["RMJD"]))
                fiber = int(scalar(row["FIBER"]))
                entry = {
                    "sdssj": clean_text(row["SDSSJ"]),
                    "oname": clean_text(row["ONAME"]) if "ONAME" in names else "",
                    "ra_deg": finite_float(row["RA"]),
                    "dec_deg": finite_float(row["DEC"]),
                    "plate": plate,
                    "smjd": smjd,
                    "rmjd": rmjd,
                    "fiber": fiber,
                    "redshift": finite_float(row["z"]) if "z" in names else float("nan"),
                }
                for candidate in (entry["sdssj"], entry["oname"]):
                    key = normalize_qso_name(candidate)
                    if key:
                        by_name.setdefault(key, entry)
                by_key.setdefault((plate, smjd, fiber), entry)
                by_key.setdefault((plate, rmjd, fiber), entry)
    except Exception as exc:
        diagnostics.append(f"could not read DR7Q catalog {path}: {exc}")
    return by_name, by_key, diagnostics


def build_dr7q_coordinate_index(dr7q_index: dict[tuple[int, int, int], Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for key, path in sorted(dr7q_index.items()):
        try:
            with fits.open(path, memmap=True) as hdul:
                header = hdul[0].header
                ra = header.get("RA")
                dec = header.get("DEC")
                if ra is None or dec is None:
                    diagnostics.append(f"missing RA/DEC header in {path}")
                    continue
                rows.append(
                    {
                        "path": path,
                        "key": key,
                        "ra_deg": float(ra),
                        "dec_deg": float(dec),
                    }
                )
        except Exception as exc:
            diagnostics.append(f"could not inspect DR7Q coordinates for {path}: {exc}")
    return rows, diagnostics


def match_dr7q_spectrum(
    row: dict[str, Any],
    requested_key: tuple[int, int, int],
    dr7q_index: dict[tuple[int, int, int], Path],
    dr7q_catalog_by_name: dict[str, dict[str, Any]],
    dr7q_catalog_by_key: dict[tuple[int, int, int], dict[str, Any]],
    dr7q_coords: SkyCoord | None,
    dr7q_coord_rows: list[dict[str, Any]],
    *,
    max_sep_arcsec: float,
    coordinate_enabled: bool,
) -> dict[str, Any]:
    direct_path = dr7q_index.get(requested_key)
    if direct_path is not None:
        return {
            "path": direct_path,
            "source": "exact_plate_mjd_fiber",
            "matched_key": requested_key,
            "catalog_entry": dr7q_catalog_by_key.get(requested_key, {}),
            "sep_arcsec": float("nan"),
        }

    catalog_entry = dr7q_catalog_by_name.get(normalize_qso_name(row.get("dr7q_name", "")), {})
    for key_name in ("smjd", "rmjd"):
        if catalog_entry:
            alt_key = (
                int(catalog_entry["plate"]),
                int(catalog_entry[key_name]),
                int(catalog_entry["fiber"]),
            )
            alt_path = dr7q_index.get(alt_key)
            if alt_path is not None:
                return {
                    "path": alt_path,
                    "source": f"catalog_plate_{key_name}_fiber",
                    "matched_key": alt_key,
                    "catalog_entry": catalog_entry,
                    "sep_arcsec": float("nan"),
                }

    if coordinate_enabled:
        try:
            ra = finite_float(row.get("qso_ra_deg"))
            dec = finite_float(row.get("qso_dec_deg"))
            if not np.isfinite(ra) or not np.isfinite(dec):
                ra = finite_float(catalog_entry.get("ra_deg")) if catalog_entry else float("nan")
                dec = finite_float(catalog_entry.get("dec_deg")) if catalog_entry else float("nan")
        except AttributeError:
            ra = dec = float("nan")
        if np.isfinite(ra) and np.isfinite(dec) and dr7q_coords is not None and dr7q_coord_rows:
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            idx, sep2d, _ = target.match_to_catalog_sky(dr7q_coords)
            idx_int = int(np.asarray(idx).reshape(-1)[0])
            sep_arcsec = float(np.asarray(sep2d.arcsec).reshape(-1)[0])
            if sep_arcsec <= max_sep_arcsec:
                coord_row = dr7q_coord_rows[idx_int]
                return {
                    "path": coord_row["path"],
                    "source": "coordinate_fallback",
                    "matched_key": coord_row["key"],
                    "catalog_entry": catalog_entry,
                    "sep_arcsec": sep_arcsec,
                }

    return {
        "path": None,
        "source": "none",
        "matched_key": requested_key,
        "catalog_entry": catalog_entry,
        "sep_arcsec": float("nan"),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).expanduser().resolve()
    default_chimera_dir = DEFAULT_CHIMERA_DIR if DEFAULT_CHIMERA_DIR.exists() else data_dir / "chimeras-2023-10-11"
    chimera_dir = (args.chimera_dir or default_chimera_dir).expanduser().resolve()
    chimera_fits = (args.chimera_fits or chimera_dir / "chimeras-fullinfo.fits").expanduser().resolve()
    provenance_path = (args.provenance or project_root / "chimera_provenance.csv").expanduser().resolve()
    fit_manifest_path = (args.fit_manifest or project_root / "fit_manifest.csv").expanduser().resolve()
    dr7q_catalog_path = (args.dr7q_catalog or data_dir / "dr7q_photometry" / "dr7qso.fit").expanduser().resolve()
    output_dir = (
        args.output_dir or WORKFLOW_ROOT / "outputs" / "source_match_audit"
    ).expanduser().resolve()

    if args.use_existing_provenance:
        rows = load_provenance(provenance_path)
        provenance_source = str(provenance_path)
        provenance_mode = "existing_csv"
    else:
        rows = rebuild_provenance_from_chimera(chimera_fits)
        provenance_source = str(chimera_fits)
        provenance_mode = "rebuilt_from_chimera_fits"

    fit_manifest_ids: set[str] = set()
    if not args.ignore_fit_manifest and fit_manifest_path.is_file():
        fit_manifest_ids = load_fit_manifest_ids(fit_manifest_path)

    indexed_rows = list(enumerate(rows))
    if fit_manifest_ids:
        indexed_rows = [(idx, row) for idx, row in indexed_rows if row.get("chimera_id") in fit_manifest_ids]
    indexed_rows = indexed_rows[args.start_index :]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]

    galaxy_dirs = [
        data_dir / "zCOSMOS_data",
        data_dir / "zCOSMOS selected",
        data_dir / "cesam_vudz",
        data_dir / "cesam_vuds",
    ]
    zcosmos_matches_path = find_zcosmos_matches_path(args.zcosmos_matches, project_root, data_dir)

    print(f"Project root: {project_root}")
    print(f"Data dir: {data_dir}")
    print(f"Chimera dir: {chimera_dir}")
    print(f"Provenance mode: {provenance_mode}")
    print(f"Provenance source: {provenance_source}")
    if fit_manifest_ids:
        print(f"Restricted to fit manifest: {fit_manifest_path} ({len(fit_manifest_ids)} IDs)")
    else:
        print("Fit manifest restriction: off")
    print(f"Rows selected for audit: {len(indexed_rows)}")
    print(f"Output dir: {output_dir}")

    if zcosmos_matches_path is not None:
        explicit_index, explicit_diagnostics = build_zcosmos_match_index(zcosmos_matches_path, galaxy_dirs)
        print(f"Explicit zCOSMOS match table: {zcosmos_matches_path}")
        print(f"Explicit zCOSMOS matched COSMOS IDs: {len(explicit_index)}")
    else:
        explicit_index = {}
        explicit_diagnostics = ["no explicit zCOSMOS match table found"]
        print("Explicit zCOSMOS match table: not found")

    fallback_index, fallback_diagnostics = build_galaxy_spectrum_index(galaxy_dirs)
    galaxy_index = merge_galaxy_indices(explicit_index, fallback_index)
    if args.disable_coordinate_fallback:
        coordinate_rows = []
        coordinate_diagnostics = ["coordinate fallback disabled"]
        zcosmos_coords = None
    else:
        coordinate_rows, coordinate_diagnostics = build_zcosmos_coordinate_index(galaxy_dirs)
        zcosmos_coords = (
            SkyCoord(
                ra=[row["ra_deg"] for row in coordinate_rows] * u.deg,
                dec=[row["dec_deg"] for row in coordinate_rows] * u.deg,
            )
            if coordinate_rows
            else None
        )
    dr7q_index = build_dr7q_spectrum_index(data_dir / "dr7q_spectra")
    dr7q_catalog_by_name, dr7q_catalog_by_key, dr7q_catalog_diagnostics = build_dr7q_catalog_index(dr7q_catalog_path)
    if args.disable_qso_coordinate_fallback:
        dr7q_coordinate_rows = []
        dr7q_coordinate_diagnostics = ["DR7Q coordinate fallback disabled"]
        dr7q_coords = None
    else:
        dr7q_coordinate_rows, dr7q_coordinate_diagnostics = build_dr7q_coordinate_index(dr7q_index)
        dr7q_coords = (
            SkyCoord(
                ra=[row["ra_deg"] for row in dr7q_coordinate_rows] * u.deg,
                dec=[row["dec_deg"] for row in dr7q_coordinate_rows] * u.deg,
            )
            if dr7q_coordinate_rows
            else None
        )
    print(f"Header/readme galaxy matched COSMOS IDs: {len(fallback_index)}")
    print(f"zCOSMOS coordinate-indexed spectra: {len(coordinate_rows)}")
    print(f"Coordinate fallback radius: {args.coord_match_arcsec:g} arcsec")
    print(f"Combined galaxy matched COSMOS IDs: {len(galaxy_index)}")
    print(f"DR7Q spectra indexed: {len(dr7q_index)}")
    print(f"DR7Q catalog path: {dr7q_catalog_path}")
    print(f"DR7Q catalog names indexed: {len(dr7q_catalog_by_name)}")
    print(f"DR7Q coordinate-indexed local spectra: {len(dr7q_coordinate_rows)}")
    print(f"DR7Q coordinate fallback radius: {args.qso_coord_match_arcsec:g} arcsec")

    audit_rows: list[dict[str, Any]] = []
    missing_galaxy_ids: set[int] = set()
    missing_dr7q_keys: set[tuple[int, int, int]] = set()

    for row_index, row in indexed_rows:
        chimera_id = row.get("chimera_id", "")
        cosmos_id = int(row["cosmos_id"])
        dr7q_key = (
            int(row["dr7q_plate"]),
            int(row["dr7q_mjd"]),
            int(row["dr7q_fiber"]),
        )
        explicit_paths = explicit_index.get(cosmos_id, [])
        fallback_paths = fallback_index.get(cosmos_id, [])
        galaxy_paths = galaxy_index.get(cosmos_id, [])
        coordinate_paths: list[Path] = []
        coordinate_sep_arcsec = float("nan")
        coordinate_object = ""
        if not galaxy_paths:
            coordinate_paths, coordinate_sep_arcsec, coordinate_object = coordinate_match(
                row.get("galaxy_ra_deg"),
                row.get("galaxy_dec_deg"),
                zcosmos_coords,
                coordinate_rows,
                max_sep_arcsec=args.coord_match_arcsec,
            )
            galaxy_paths = coordinate_paths
        dr7q_match = match_dr7q_spectrum(
            row,
            dr7q_key,
            dr7q_index,
            dr7q_catalog_by_name,
            dr7q_catalog_by_key,
            dr7q_coords,
            dr7q_coordinate_rows,
            max_sep_arcsec=args.qso_coord_match_arcsec,
            coordinate_enabled=not args.disable_qso_coordinate_fallback,
        )
        dr7q_path = dr7q_match["path"]
        dr7q_matched_key = dr7q_match["matched_key"]
        dr7q_catalog_entry = dr7q_match["catalog_entry"]
        has_galaxy = bool(galaxy_paths)
        has_dr7q = dr7q_path is not None
        if not has_galaxy:
            missing_galaxy_ids.add(cosmos_id)
        if not has_dr7q:
            missing_dr7q_keys.add(dr7q_key)

        audit_rows.append(
            {
                "status": status_label(has_galaxy, has_dr7q),
                "row_index": row_index,
                "chimera_id": chimera_id,
                "cosmos_id": cosmos_id,
                "dr7q_spectrum_id": row.get("dr7q_spectrum_id", ""),
                "dr7q_plate": dr7q_key[0],
                "dr7q_mjd": dr7q_key[1],
                "dr7q_fiber": dr7q_key[2],
                "dr7q_requested_key": f"{dr7q_key[0]}-{dr7q_key[1]}-{dr7q_key[2]}",
                "dr7q_matched_key": (
                    f"{dr7q_matched_key[0]}-{dr7q_matched_key[1]}-{dr7q_matched_key[2]}"
                    if dr7q_matched_key
                    else ""
                ),
                "qso_key_was_remapped": bool(has_dr7q and dr7q_matched_key != dr7q_key),
                "qso_match_source": qso_source_label(dr7q_match["source"]),
                "qso_missing_reason": qso_missing_reason(has_dr7q, dr7q_catalog_entry),
                "qso_coordinate_sep_arcsec": dr7q_match["sep_arcsec"],
                "qso_ra_deg": row.get("qso_ra_deg", ""),
                "qso_dec_deg": row.get("qso_dec_deg", ""),
                "qso_catalog_name": dr7q_catalog_entry.get("sdssj", ""),
                "qso_catalog_oname": dr7q_catalog_entry.get("oname", ""),
                "qso_catalog_ra_deg": dr7q_catalog_entry.get("ra_deg", ""),
                "qso_catalog_dec_deg": dr7q_catalog_entry.get("dec_deg", ""),
                "qso_catalog_plate": dr7q_catalog_entry.get("plate", ""),
                "qso_catalog_smjd": dr7q_catalog_entry.get("smjd", ""),
                "qso_catalog_rmjd": dr7q_catalog_entry.get("rmjd", ""),
                "qso_catalog_fiber": dr7q_catalog_entry.get("fiber", ""),
                "chimera_redshift": row.get("chimera_redshift", ""),
                "dr7q_redshift": row.get("dr7q_redshift", ""),
                "chimera_qso_weight": row.get("chimera_qso_weight", ""),
                "galaxy_ra_deg": row.get("galaxy_ra_deg", ""),
                "galaxy_dec_deg": row.get("galaxy_dec_deg", ""),
                "galaxy_match_source": source_label(explicit_paths, fallback_paths, coordinate_paths),
                "n_explicit_galaxy_paths": len(explicit_paths),
                "n_header_galaxy_paths": len(fallback_paths),
                "n_coordinate_galaxy_paths": len(coordinate_paths),
                "n_combined_galaxy_paths": len(galaxy_paths),
                "coordinate_sep_arcsec": coordinate_sep_arcsec,
                "coordinate_zcosmos_object": coordinate_object,
                "galaxy_spectrum_path": first_path(galaxy_paths),
                "dr7q_spectrum_path": str(dr7q_path) if dr7q_path is not None else "",
            }
        )

    status_counts = Counter(row["status"] for row in audit_rows)
    source_counts = Counter(row["galaxy_match_source"] for row in audit_rows)
    qso_source_counts = Counter(row["qso_match_source"] for row in audit_rows)
    qso_missing_reason_counts = Counter(row["qso_missing_reason"] for row in audit_rows if row["qso_missing_reason"])
    summary = {
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "chimera_dir": str(chimera_dir),
        "chimera_fits": str(chimera_fits),
        "dr7q_catalog": str(dr7q_catalog_path),
        "provenance_mode": provenance_mode,
        "provenance_source": provenance_source,
        "n_provenance_rows": len(rows),
        "fit_manifest": str(fit_manifest_path) if fit_manifest_ids else "",
        "fit_manifest_restricted": bool(fit_manifest_ids),
        "n_fit_manifest_ids": len(fit_manifest_ids),
        "zcosmos_matches_path": str(zcosmos_matches_path) if zcosmos_matches_path else "",
        "n_explicit_zcosmos_cosmos_ids": len(explicit_index),
        "n_header_or_readme_galaxy_cosmos_ids": len(fallback_index),
        "n_coordinate_indexed_zcosmos_spectra": len(coordinate_rows),
        "coordinate_match_arcsec": args.coord_match_arcsec,
        "coordinate_fallback_enabled": not args.disable_coordinate_fallback,
        "n_combined_galaxy_cosmos_ids": len(galaxy_index),
        "n_dr7q_spectra": len(dr7q_index),
        "n_dr7q_catalog_names": len(dr7q_catalog_by_name),
        "n_coordinate_indexed_dr7q_spectra": len(dr7q_coordinate_rows),
        "qso_coordinate_match_arcsec": args.qso_coord_match_arcsec,
        "qso_coordinate_fallback_enabled": not args.disable_qso_coordinate_fallback,
        "n_rows_audited": len(audit_rows),
        "status_counts": dict(status_counts),
        "galaxy_match_source_counts": dict(source_counts),
        "qso_match_source_counts": dict(qso_source_counts),
        "qso_missing_reason_counts": dict(qso_missing_reason_counts),
        "n_unique_missing_galaxy_cosmos_ids": len(missing_galaxy_ids),
        "n_unique_missing_dr7q_keys": len(missing_dr7q_keys),
        "n_explicit_zcosmos_diagnostics": len(explicit_diagnostics),
        "n_header_or_readme_diagnostics": len(fallback_diagnostics),
        "n_coordinate_diagnostics": len(coordinate_diagnostics),
        "n_dr7q_catalog_diagnostics": len(dr7q_catalog_diagnostics),
        "n_dr7q_coordinate_diagnostics": len(dr7q_coordinate_diagnostics),
    }

    write_csv(output_dir / "rebuilt_chimera_provenance.csv", rows)
    write_csv(output_dir / "spectrum_source_match_audit.csv", audit_rows)
    write_csv(
        output_dir / "missing_galaxy_cosmos_ids.csv",
        [{"cosmos_id": cosmos_id} for cosmos_id in sorted(missing_galaxy_ids)],
    )
    write_csv(
        output_dir / "missing_dr7q_spectra.csv",
        [
            {"dr7q_plate": plate, "dr7q_mjd": mjd, "dr7q_fiber": fiber}
            for plate, mjd, fiber in sorted(missing_dr7q_keys)
        ],
    )
    write_csv(
        output_dir / "missing_dr7q_objects.csv",
        [row for row in audit_rows if not row["dr7q_spectrum_path"]],
    )
    write_csv(
        output_dir / "match_diagnostics.csv",
        [{"source": "explicit_zcosmos", "message": item} for item in explicit_diagnostics]
        + [{"source": "header_or_readme", "message": item} for item in fallback_diagnostics]
        + [{"source": "coordinate_index", "message": item} for item in coordinate_diagnostics]
        + [{"source": "dr7q_catalog", "message": item} for item in dr7q_catalog_diagnostics]
        + [{"source": "dr7q_coordinate_index", "message": item} for item in dr7q_coordinate_diagnostics],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("Status counts:")
    for key in sorted(status_counts):
        print(f"  {key}: {status_counts[key]}")
    print("QSO match source counts:")
    for key in sorted(qso_source_counts):
        print(f"  {key}: {qso_source_counts[key]}")
    print("QSO missing reason counts:")
    for key in sorted(qso_missing_reason_counts):
        print(f"  {key}: {qso_missing_reason_counts[key]}")
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
