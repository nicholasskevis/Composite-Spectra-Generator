#!/usr/bin/env python
"""Plot galaxy, weighted-QSO, and composite spectra for safe Chimera objects.

The output is a multi-page PDF with one safe spectrum per page.  Galaxy and QSO
components are reconstructed from the source FITS files recorded in each safe
ECSV file, using the same transformations as the composite-spectrum builder.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from astropy.table import Table

from build_all_chimera_composite_spectra import (
    apply_foreground_extinction,
    flambda_cgs_to_mjy,
    interp_flux,
    load_sdss_dr7q_spectrum,
    load_zcosmos_spectrum,
    shift_flux_density_to_redshift,
)


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKFLOW_ROOT
    / "outputs"
    / "safe_chimera_spectra"
    / "safe_chimera_spectra_manifest.csv"
)
DEFAULT_OUTPUT = (
    WORKFLOW_ROOT / "outputs" / "safe_chimera_spectra" / "safe_spectra_components_100.pdf"
)
DEFAULT_SOURCE_CANDIDATES = (
    WORKFLOW_ROOT
    / "outputs"
    / "all_chimera_spectra"
    / "chimera_spectrum_source_candidates.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a one-object-per-page PDF showing the galaxy, weighted QSO, "
            "and safe composite spectra."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-candidates",
        type=Path,
        default=DEFAULT_SOURCE_CANDIDATES,
        help=(
            "Optional scored source-candidate CSV from build_all_chimera_composite_spectra.py. "
            "Used to label each plotted spectrum with its selected source origin."
        ),
    )
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible random selection (default: 42).",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Use the first N accepted manifest rows instead of a random sample.",
    )
    return parser.parse_args()


def read_accepted_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("status") == "success" and row.get("action") == "accepted"
    ]


def resolve_path(raw_path: str, manifest: Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [manifest.parent / path, WORKFLOW_ROOT.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve {raw_path!r}")


def read_selected_source_candidates(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    selected: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            chimera_id = row.get("chimera_id", "")
            if not chimera_id:
                continue
            is_selected = row.get("selected_for_composite", "").lower() == "true"
            if is_selected or chimera_id not in selected:
                selected[chimera_id] = row
    return selected


def infer_survey_label(path: str, *, role: str) -> str:
    text = str(path).lower()
    if "zcosmos" in text:
        return "zCOSMOS"
    if "cesam_vuds" in text or "vuds" in text:
        return "VUDS/CESAM"
    if "cesam_vudz" in text or "vudz" in text:
        return "VUDz/CESAM"
    if "dr7q" in text:
        return "SDSS DR7Q"
    if "sdss" in text or "spec-" in Path(str(path)).name.lower():
        return "SDSS"
    if "desi" in text or "sparcl" in text:
        return "DESI"
    return f"unknown {role} source"


def format_source_note(
    table: Table,
    row: dict[str, str],
    selected_sources: dict[str, dict[str, str]],
) -> tuple[str, str, str]:
    chimera_id = row.get("chimera_id", table.meta.get("chimera_id", ""))
    galaxy_path = row.get("galaxy_spectrum_path") or str(table.meta.get("galaxy_spectrum_path", ""))
    qso_path = row.get("qso_spectrum_path") or str(table.meta.get("qso_spectrum_path", ""))
    candidate = selected_sources.get(chimera_id, {})

    galaxy_survey = infer_survey_label(galaxy_path, role="galaxy")
    qso_survey = infer_survey_label(qso_path, role="QSO")
    origin = candidate.get("candidate_source_origin") or table.meta.get("galaxy_spectrum_match_source", "")
    snr = candidate.get("galaxy_spectrum_snr_median") or table.meta.get("galaxy_spectrum_snr_median", "")
    candidate_count = candidate.get("galaxy_spectrum_candidate_count") or table.meta.get("galaxy_spectrum_candidate_count", "")

    pieces = [galaxy_survey]
    if origin:
        pieces.append(str(origin).replace("_", " "))
    if candidate_count not in ("", None):
        pieces.append(f"{candidate_count} candidate(s)")
    if snr not in ("", None):
        try:
            pieces.append(f"median S/N={float(snr):.2g}")
        except (TypeError, ValueError):
            pieces.append(f"median S/N={snr}")

    galaxy_note = "; ".join(pieces)
    qso_note = qso_survey
    summary_note = f"galaxy: {galaxy_note} | QSO: {qso_note}"
    return galaxy_note, qso_note, summary_note


def component_fluxes_mjy(table: Table) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(table["wave_obs"], dtype=float)
    target_redshift = float(table.meta["chimera_redshift"])
    qso_redshift = float(table.meta["dr7q_redshift"])
    qso_weight = float(table.meta["chimera_qso_weight"])
    ebv = float(table.meta.get("cosmos_ebv", 0.0))
    apply_extinction = table.meta.get("extinction_curve", "none") != "none"

    galaxy = load_zcosmos_spectrum(Path(table.meta["galaxy_spectrum_path"]), target_redshift)
    qso = load_sdss_dr7q_spectrum(Path(table.meta["qso_spectrum_path"]), qso_redshift)
    galaxy = shift_flux_density_to_redshift(galaxy, target_redshift)
    qso = shift_flux_density_to_redshift(qso, target_redshift)

    for spectrum in (galaxy, qso):
        flux, _ = apply_foreground_extinction(
            spectrum["wavelength_target_obs_angstrom"],
            spectrum["flux_density_target_obs"],
            ebv,
            apply=apply_extinction,
        )
        spectrum["flux_density_extincted"] = flux

    galaxy_flux = interp_flux(galaxy, wave)
    weighted_qso_flux = qso_weight * interp_flux(qso, wave)
    return (
        flambda_cgs_to_mjy(wave, galaxy_flux),
        flambda_cgs_to_mjy(wave, weighted_qso_flux),
    )


def robust_limits(*arrays: np.ndarray) -> tuple[float, float] | None:
    finite = np.concatenate([np.asarray(array)[np.isfinite(array)] for array in arrays])
    if finite.size == 0:
        return None
    low, high = np.nanpercentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None
    padding = 0.08 * (high - low)
    return float(low - padding), float(high + padding)


def add_page(
    pdf,
    row: dict[str, str],
    manifest: Path,
    selected_sources: dict[str, dict[str, str]],
) -> None:
    import matplotlib.pyplot as plt

    spectrum_path = resolve_path(row["spectrum_path"], manifest)
    table = Table.read(spectrum_path, format="ascii.ecsv")

    # Source paths in old ECSV metadata may be absolute.  Replace them with
    # manifest-provided paths when available so relocated workflows still work.
    if row.get("galaxy_spectrum_path"):
        table.meta["galaxy_spectrum_path"] = str(
            resolve_path(row["galaxy_spectrum_path"], manifest)
        )
    if row.get("qso_spectrum_path"):
        table.meta["qso_spectrum_path"] = str(
            resolve_path(row["qso_spectrum_path"], manifest)
        )

    wave = np.asarray(table["wave_obs"], dtype=float)
    composite = np.asarray(table["flux_mjy"], dtype=float)
    mask = np.asarray(table["mask"], dtype=bool)
    galaxy, weighted_qso = component_fluxes_mjy(table)
    valid = mask & np.isfinite(wave)
    if not np.any(valid):
        raise ValueError(f"{row['chimera_id']} has no valid safe pixels")

    qso_weight = float(table.meta["chimera_qso_weight"])
    galaxy_note, qso_note, summary_note = format_source_note(table, row, selected_sources)
    panels = (
        (f"Galaxy spectrum ({galaxy_note})", galaxy, "#2878B5"),
        (f"QSO spectrum × {qso_weight:g} ({qso_note})", weighted_qso, "#D95319"),
        ("Safe composite spectrum", composite, "black"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)
    for axis, (label, flux, color) in zip(axes, panels):
        plot_mask = valid & np.isfinite(flux)
        axis.plot(wave[plot_mask], flux[plot_mask], color=color, linewidth=0.75)
        axis.set_ylabel("Flux (mJy)")
        axis.set_title(label, loc="left", fontsize=10)
        axis.grid(alpha=0.18, linewidth=0.5)
        limits = robust_limits(flux[plot_mask])
        if limits is not None:
            axis.set_ylim(*limits)

    axes[-1].set_xlabel("Observed wavelength at Chimera redshift (Å)")
    axes[-1].set_xlim(float(np.nanmin(wave[valid])), float(np.nanmax(wave[valid])))
    fig.suptitle(
        f"{row['chimera_id']}   "
        f"z={float(table.meta['chimera_redshift']):.4f}   "
        f"QSO weight={qso_weight:g}\n{summary_note}",
        fontsize=12,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_candidates = args.source_candidates.expanduser().resolve()
    if args.number < 1:
        raise ValueError("--number must be at least 1")

    rows = read_accepted_rows(manifest)
    if not rows:
        raise ValueError(f"no accepted safe spectra found in {manifest}")
    count = min(args.number, len(rows))
    if args.first:
        selected = rows[:count]
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(rows), size=count, replace=False)
        selected = [rows[int(index)] for index in indices]

    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output.parent / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    from matplotlib.backends.backend_pdf import PdfPages

    selected_sources = read_selected_source_candidates(source_candidates)
    if selected_sources:
        print(f"Loaded selected source labels for {len(selected_sources)} Chimera IDs from {source_candidates}")
    else:
        print(f"No source-candidate labels found at {source_candidates}; inferring labels from paths")

    failures: list[str] = []
    with PdfPages(output) as pdf:
        for index, row in enumerate(selected, start=1):
            try:
                add_page(pdf, row, manifest, selected_sources)
            except Exception as exc:
                failures.append(f"{row.get('chimera_id', '<unknown>')}: {exc}")
            else:
                print(f"[{index}/{count}] {row['chimera_id']}")

    if failures:
        failure_path = output.with_suffix(".failures.txt")
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Skipped {len(failures)} spectra; details: {failure_path}")
    print(f"Wrote {count - len(failures)} pages to {output}")
    return 0 if count > len(failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
