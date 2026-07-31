#!/usr/bin/env python
"""Run the full Chimera composite-spectra workflow.

This driver calls the existing workflow scripts in order:

1. audit source-spectrum matches
2. build all available composite spectra
3. audit generated spectra against Chimera photometry
4. build the safe spectra manifest for joint fitting

Raw data remain outside the repository.  Generated products are written under
``spectra_workflow/outputs`` by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astropy.io import fits


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_DATA_DIR = Path("/home/nicho/GRAHSP_my/data")
DEFAULT_CHIMERA_DIR = Path("/home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "External spectroscopy/catalog data directory containing zCOSMOS_data, "
            "dr7q_spectra, dr7q_photometry, etc. Default: /home/nicho/GRAHSP_my/data "
            "if present, otherwise project-root/data. This does not choose the Chimera "
            "FITS catalog; use --chimera-dir for that."
        ),
    )
    parser.add_argument(
        "--chimera-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the Chimera FITS files used to rebuild provenance. "
            "Default: /home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11 "
            "if present, otherwise data-dir/chimeras-2023-10-11."
        ),
    )
    parser.add_argument("--chimera-fits", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--use-existing-provenance", action="store_true")
    parser.add_argument("--fit-manifest", type=Path, default=None)
    parser.add_argument("--ignore-fit-manifest", action="store_true")
    parser.add_argument("--zcosmos-matches", type=Path, default=None)
    parser.add_argument("--dr7q-catalog", type=Path, default=None)
    parser.add_argument(
        "--qso-spectrum-overrides",
        type=Path,
        default=None,
        help="Optional replacement QSO spectrum map produced by search_sdss_replacement_spectra.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root for all workflow products. Default: spectra_workflow/outputs.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--coord-match-arcsec", type=float, default=1.0)
    parser.add_argument("--qso-coord-match-arcsec", type=float, default=1.0)
    parser.add_argument("--disable-coordinate-fallback", action="store_true")
    parser.add_argument("--disable-qso-coordinate-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-extinction", action="store_true")
    parser.add_argument("--error-floor-fraction", type=float, default=0.10)
    parser.add_argument(
        "--resampling-method",
        choices=("flux-conserving", "interp"),
        default="flux-conserving",
    )
    parser.add_argument("--no-resolution-match", action="store_true")
    parser.add_argument("--galaxy-resolving-power", type=float, default=600.0)
    parser.add_argument("--qso-resolving-power", type=float, default=2000.0)
    parser.add_argument("--resolution-kernel-sigma-width", type=float, default=4.0)
    parser.add_argument("--min-valid-pixels", type=int, default=50)
    parser.add_argument("--write-full-table", action="store_true")
    parser.add_argument("--write-rest-table", action="store_true")
    parser.add_argument("--local-window-a", type=float, default=150.0)
    parser.add_argument("--low-qso-weight", type=float, default=1.0e-3)
    parser.add_argument("--high-qso-weight", type=float, default=0.1)
    parser.add_argument("--component-audit", action="store_true")
    parser.add_argument("--include-nonpositive", action="store_true")
    parser.add_argument("--safe-min-scale", type=float, default=0.2)
    parser.add_argument("--safe-max-scale", type=float, default=5.0)
    parser.add_argument("--safe-min-overlap-bands", type=int, default=1)
    parser.add_argument("--safe-max-negative-fraction", type=float, default=0.20)
    parser.add_argument("--safe-keep-nonpositive-flux", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the workflow summary without executing the steps.",
    )
    return parser.parse_args()


def append_path_arg(cmd: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def append_value_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def append_bool_arg(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def run_step(name: str, cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    print(f"\n[{name}]")
    print(" ".join(cmd))
    payload: dict[str, Any] = {"name": name, "command": cmd}
    if dry_run:
        payload["returncode"] = None
        payload["skipped"] = True
        return payload
    result = subprocess.run(cmd, check=True)
    payload["returncode"] = result.returncode
    payload["skipped"] = False
    return payload


def count_fits_rows(path: Path) -> int:
    with fits.open(path, memmap=True) as hdul:
        return int(len(hdul[1].data))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    default_data_dir = DEFAULT_EXTERNAL_DATA_DIR if DEFAULT_EXTERNAL_DATA_DIR.exists() else project_root / "data"
    data_dir = (args.data_dir or default_data_dir).expanduser().resolve()
    default_chimera_dir = DEFAULT_CHIMERA_DIR if DEFAULT_CHIMERA_DIR.exists() else data_dir / "chimeras-2023-10-11"
    chimera_dir = (args.chimera_dir or default_chimera_dir).expanduser().resolve()
    chimera_fits = (args.chimera_fits or chimera_dir / "chimeras-fullinfo.fits").expanduser().resolve()
    if not chimera_fits.is_file():
        raise FileNotFoundError(f"Missing Chimera FITS file: {chimera_fits}")
    chimera_row_count = count_fits_rows(chimera_fits)
    fit_manifest = (args.fit_manifest or project_root / "fit_manifest.csv").expanduser().resolve()
    zcosmos_matches = (
        args.zcosmos_matches
        or WORKFLOW_ROOT / "config" / "chimera_zcosmos_alpha_delta_matches.csv"
    ).expanduser().resolve()
    output_root = (args.output_root or WORKFLOW_ROOT / "outputs").expanduser().resolve()

    source_audit_dir = output_root / "source_match_audit"
    rebuilt_provenance = source_audit_dir / "rebuilt_chimera_provenance.csv"
    source_match_audit = source_audit_dir / "spectrum_source_match_audit.csv"
    spectra_dir = output_root / "all_chimera_spectra"
    spectra_manifest = spectra_dir / "chimera_spectra_manifest.csv"
    spectra_audit_dir = output_root / "chimera_composite_spectra_audit"
    safe_dir = output_root / "safe_chimera_spectra"

    scripts_dir = WORKFLOW_ROOT / "scripts"
    python = sys.executable

    common = [python]
    source_cmd = common + [str(scripts_dir / "audit_spectrum_source_matches.py")]
    append_path_arg(source_cmd, "--project-root", project_root)
    append_path_arg(source_cmd, "--data-dir", data_dir)
    append_path_arg(source_cmd, "--chimera-dir", chimera_dir)
    append_path_arg(source_cmd, "--chimera-fits", chimera_fits)
    append_path_arg(source_cmd, "--provenance", args.provenance)
    append_bool_arg(source_cmd, "--use-existing-provenance", args.use_existing_provenance)
    append_path_arg(source_cmd, "--fit-manifest", fit_manifest)
    append_bool_arg(source_cmd, "--ignore-fit-manifest", args.ignore_fit_manifest)
    append_path_arg(source_cmd, "--zcosmos-matches", zcosmos_matches)
    append_path_arg(source_cmd, "--dr7q-catalog", args.dr7q_catalog)
    append_value_arg(source_cmd, "--coord-match-arcsec", args.coord_match_arcsec)
    append_value_arg(source_cmd, "--qso-coord-match-arcsec", args.qso_coord_match_arcsec)
    append_bool_arg(source_cmd, "--disable-coordinate-fallback", args.disable_coordinate_fallback)
    append_bool_arg(source_cmd, "--disable-qso-coordinate-fallback", args.disable_qso_coordinate_fallback)
    append_path_arg(source_cmd, "--output-dir", source_audit_dir)
    append_value_arg(source_cmd, "--start-index", args.start_index)
    append_value_arg(source_cmd, "--limit", args.limit)

    build_cmd = common + [str(scripts_dir / "build_all_chimera_composite_spectra.py")]
    append_path_arg(build_cmd, "--project-root", project_root)
    append_path_arg(build_cmd, "--data-dir", data_dir)
    append_path_arg(build_cmd, "--provenance", args.provenance or rebuilt_provenance)
    append_path_arg(build_cmd, "--fit-manifest", fit_manifest)
    append_bool_arg(build_cmd, "--ignore-fit-manifest", args.ignore_fit_manifest)
    append_path_arg(build_cmd, "--zcosmos-matches", zcosmos_matches)
    append_path_arg(build_cmd, "--source-match-audit", source_match_audit)
    append_path_arg(build_cmd, "--qso-spectrum-overrides", args.qso_spectrum_overrides)
    append_path_arg(build_cmd, "--output-dir", spectra_dir)
    append_value_arg(build_cmd, "--start-index", args.start_index)
    append_value_arg(build_cmd, "--limit", args.limit)
    append_bool_arg(build_cmd, "--overwrite", args.overwrite)
    append_bool_arg(build_cmd, "--no-extinction", args.no_extinction)
    append_value_arg(build_cmd, "--error-floor-fraction", args.error_floor_fraction)
    append_value_arg(build_cmd, "--resampling-method", args.resampling_method)
    append_bool_arg(build_cmd, "--no-resolution-match", args.no_resolution_match)
    append_value_arg(build_cmd, "--galaxy-resolving-power", args.galaxy_resolving_power)
    append_value_arg(build_cmd, "--qso-resolving-power", args.qso_resolving_power)
    append_value_arg(build_cmd, "--resolution-kernel-sigma-width", args.resolution_kernel_sigma_width)
    append_value_arg(build_cmd, "--min-valid-pixels", args.min_valid_pixels)
    append_bool_arg(build_cmd, "--write-full-table", args.write_full_table)
    append_bool_arg(build_cmd, "--write-rest-table", args.write_rest_table)

    audit_cmd = common + [str(scripts_dir / "audit_chimera_composite_spectra.py")]
    append_path_arg(audit_cmd, "--project-root", project_root)
    append_path_arg(audit_cmd, "--fit-manifest", fit_manifest)
    append_path_arg(audit_cmd, "--chimera-fits", chimera_fits)
    append_path_arg(audit_cmd, "--spectra-manifest", spectra_manifest)
    append_path_arg(audit_cmd, "--output-dir", spectra_audit_dir)
    append_value_arg(audit_cmd, "--local-window-a", args.local_window_a)
    append_value_arg(audit_cmd, "--low-qso-weight", args.low_qso_weight)
    append_value_arg(audit_cmd, "--high-qso-weight", args.high_qso_weight)
    append_value_arg(audit_cmd, "--max-rows", args.limit)
    append_bool_arg(audit_cmd, "--component-audit", args.component_audit)
    append_bool_arg(audit_cmd, "--include-nonpositive", args.include_nonpositive)

    safe_cmd = common + [str(scripts_dir / "build_safe_joint_spectra_manifest.py")]
    append_path_arg(safe_cmd, "--project-root", project_root)
    append_path_arg(safe_cmd, "--fit-manifest", fit_manifest)
    append_path_arg(safe_cmd, "--chimera-fits", chimera_fits)
    append_path_arg(safe_cmd, "--input-spectra-manifest", spectra_manifest)
    append_path_arg(safe_cmd, "--output-dir", safe_dir)
    append_value_arg(safe_cmd, "--min-scale", args.safe_min_scale)
    append_value_arg(safe_cmd, "--max-scale", args.safe_max_scale)
    append_value_arg(safe_cmd, "--min-overlap-bands", args.safe_min_overlap_bands)
    append_value_arg(safe_cmd, "--min-valid-pixels", args.min_valid_pixels)
    append_value_arg(safe_cmd, "--max-negative-fraction", args.safe_max_negative_fraction)
    append_bool_arg(safe_cmd, "--keep-nonpositive-flux", args.safe_keep_nonpositive_flux)
    append_bool_arg(safe_cmd, "--overwrite", args.overwrite)

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "chimera_dir": str(chimera_dir),
        "chimera_fits": str(chimera_fits),
        "chimera_row_count": chimera_row_count,
        "fit_manifest": str(fit_manifest),
        "zcosmos_matches": str(zcosmos_matches),
        "output_root": str(output_root),
        "outputs": {
            "source_match_audit": str(source_audit_dir),
            "all_chimera_spectra": str(spectra_dir),
            "chimera_spectra_manifest": str(spectra_manifest),
            "spectra_audit": str(spectra_audit_dir),
            "safe_chimera_spectra": str(safe_dir),
            "safe_manifest": str(safe_dir / "safe_chimera_spectra_manifest.csv"),
        },
        "steps": [],
    }

    print(f"Chimera FITS source: {chimera_fits}")
    print(f"Chimera rows: {chimera_row_count}")

    for name, cmd in (
        ("source-match-audit", source_cmd),
        ("build-composite-spectra", build_cmd),
        ("audit-composite-spectra", audit_cmd),
        ("build-safe-manifest", safe_cmd),
    ):
        summary["steps"].append(run_step(name, cmd, dry_run=args.dry_run))

    summary_path = output_root / "workflow_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote workflow summary: {summary_path}")
    print(f"Safe manifest: {safe_dir / 'safe_chimera_spectra_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
