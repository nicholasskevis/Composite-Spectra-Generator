from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
GRAHSP_INSTALL_ROOT = REPO_ROOT / "GRAHSP"
CIGALE_ROOT = GRAHSP_INSTALL_ROOT / "GRAHSP"
SAMPLER_SCRIPT = GRAHSP_INSTALL_ROOT / "GRAHSP-run" / "dualsampler.py"

CHIMERA_FILTER_NAMES = (
    "u_sdss",
    "r_sdss",
    "i_sdss",
    "z_sdss",
    "J_2mass",
    "H_2mass",
    "Ks_2mass",
    "spitzer.irac.I1",
    "spitzer.irac.I2",
)

GRAHSP_FILTER_NAME_MAP = {
    "u_sdss": "u_sdss",
    "r_sdss": "r_sdss",
    "i_sdss": "i_sdss",
    "z_sdss": "z_sdss",
    "J_2mass": "J_2mass",
    "H_2mass": "H_2mass",
    "Ks_2mass": "Ks_2mass",
    "spitzer.irac.I1": "IRAC1",
    "spitzer.irac.I2": "IRAC2",
}

GRAHSP_FILTER_NAMES = (
    "u_sdss",
    "r_sdss",
    "i_sdss",
    "z_sdss",
    "J_2mass",
    "H_2mass",
    "Ks_2mass",
    "IRAC1",
    "IRAC2",
)


def _array_index_from_env() -> tuple[str, int] | None:
    for name in ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX", "LSB_JOBINDEX"):
        raw = os.environ.get(name)
        if raw is not None:
            return name, int(raw)
    return None


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(_json_sanitize(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in value)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_from_manifest(raw: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": raw["object_id"],
        "ID_COSMOS": raw["COSMOS_ID0"],
        "COSMOS_ID0": raw["COSMOS_ID0"],
        "redshift": float(raw["redshift"]),
        "chimera_QSO_weight": float(raw["chimera_QSO_weight"]),
        "resample_weight": float(raw["resample_weight"]),
        "log_stellar_mass_truth": float(raw["log_stellar_mass_truth"]),
        "logLbol_QSO": float(raw["logLbol_QSO"]),
        "logLbol_chimera": float(raw["logLbol_chimera"]),
        "luminosity_bin": raw.get("luminosity_bin", ""),
        "fit_index": int(raw["fit_index"]),
    }
    for name in CHIMERA_FILTER_NAMES:
        row[name] = float(raw[name])
        row[f"{name}_err"] = float(raw[f"{name}_err"])
    return row


def _select_manifest_entry(args: argparse.Namespace) -> dict[str, str]:
    rows = _load_manifest(args.manifest)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} manifest rows, found {len(rows)}.")

    scheduler_index = _array_index_from_env()
    object_selector = args.cosmos_id0 is not None or args.object_id is not None
    if args.object_id is not None and args.cosmos_id0 is None:
        raise RuntimeError("--object-id requires --cosmos-id0.")
    if object_selector and (args.array_index is not None or args.fit_index is not None or scheduler_index is not None):
        scheduler_name = scheduler_index[0] if scheduler_index is not None else None
        raise RuntimeError(
            "Conflicting manifest selectors. Use exactly one of: "
            "--fit-index, array selection (--array-index or scheduler array env), or --cosmos-id0/--object-id. "
            f"Got cosmos/object selector plus "
            f"{'--array-index ' if args.array_index is not None else ''}"
            f"{'--fit-index ' if args.fit_index is not None else ''}"
            f"{scheduler_name or ''}."
        )
    if args.fit_index is not None and args.array_index is not None:
        raise RuntimeError("Conflicting manifest selectors: use --fit-index or --array-index, not both.")
    if args.array_index is not None and scheduler_index is not None and int(args.array_index) != int(scheduler_index[1]):
        raise RuntimeError(
            f"Conflicting array selectors: --array-index={args.array_index} but "
            f"{scheduler_index[0]}={scheduler_index[1]}."
        )

    if object_selector:
        matches = [row for row in rows if str(row["COSMOS_ID0"]) == str(args.cosmos_id0)]
        if args.object_id is not None:
            matches = [row for row in matches if str(row["object_id"]) == str(args.object_id)]
        if len(matches) != 1:
            raise RuntimeError(
                f"COSMOS_ID0={args.cosmos_id0!r} matched {len(matches)} rows. "
                "Pass --object-id as well when one COSMOS_ID0 has multiple Chimera rows."
            )
        return matches[0]

    if args.fit_index is not None:
        fit_index = int(args.fit_index)
        if args.array_offset != 0:
            raise RuntimeError("--array-offset only applies to array selection, not --fit-index.")
        if fit_index < 0 or fit_index >= len(rows):
            raise IndexError(f"Fit index {fit_index} is outside 0..{len(rows)-1}.")
        return rows[fit_index]

    if args.array_index is None and scheduler_index is None:
        raise RuntimeError("No manifest selector was provided and no scheduler array index environment variable is set.")
    raw_index = int(args.array_index) if args.array_index is not None else int(scheduler_index[1])
    fit_index = int(raw_index) - int(args.index_base) + int(args.array_offset)
    if fit_index < 0 or fit_index >= len(rows):
        raise IndexError(
            f"Array index {raw_index} with index-base={args.index_base} and array-offset={args.array_offset} "
            f"maps to manifest row {fit_index}, outside 0..{len(rows)-1}."
        )
    return rows[fit_index]


def _finite_or_missing(value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value):
        return -9999.0
    if positive and value <= 0.0:
        return -9999.0
    return value


def _write_grahsp_input_table(row: dict[str, Any], work_dir: Path) -> Path:
    table_data: dict[str, list[Any]] = {
        "id": [str(row["id"])],
        "redshift": [float(row["redshift"])],
        "redshift_err": [0.0],
    }
    for chimera_name, grahsp_name in GRAHSP_FILTER_NAME_MAP.items():
        table_data[grahsp_name] = [_finite_or_missing(row[chimera_name])]
        table_data[f"{grahsp_name}_err"] = [_finite_or_missing(row[f"{chimera_name}_err"], positive=True)]

    input_path = work_dir / "chimera_object.ecsv"
    Table(table_data).write(input_path, format="ascii.ecsv", overwrite=True)
    return input_path


def _write_pcigale_ini(input_path: Path, work_dir: Path) -> Path:
    column_list = []
    for name in GRAHSP_FILTER_NAMES:
        column_list.extend([name, f"{name}_err"])

    pcigale_ini = f"""
data_file = {input_path}
column_list = {', '.join(column_list)}
creation_modules = sfh2exp, m2005, nebular, activate, activatelines, activategtorus, activatepl, activatebol, biattenuation, galdale2014, redshifting
analysis_method = pdf_analysis
cores = 1
cosmology = concordance

[statistics]
  exponent = 2
  systematics_width = 0.03
  variability_uncertainty = true
  attenuation_model_uncertainty = false
  Ly_break_uncertainty = false

[sed_creation_modules]
  [[sfh2exp]]
    tau_main = 500, 1500, 4000, 8000
    tau_burst = 50
    f_burst = 0.0, 0.01, 0.1
    age = 500, 1000, 3000, 6000, 9000, 12000
    burst_age = 20, 100
    sfr_0 = 1
    normalise = True

  [[m2005]]
    imf = 1
    metallicity = 0.02
    separation_age = 10

  [[nebular]]
    logU = -2.0
    zgas = 0.02
    ne = 100
    f_esc = 0.0
    f_dust = 0.0
    lines_width = 300
    emission = True

  [[activate]]
    fracAGN = -1

  [[activatelines]]
    AFeII = 5
    AGNtype = 1
    linewidth = 3000
    Alines = 1
    ABC = 0

  [[activategtorus]]
    fcov = 0.05, 0.2, 0.5
    Si = 0
    COOLlam = 17
    COOLwidth = 0.45
    HOTlam = 2.0
    HOTwidth = 0.5
    HOTfcov = 0.0, 0.5
    SiRatio = 0.29
    SiEmlam = 9841
    SiAbslam = 14224
    SiEmWidth = 1025.3
    SiAbsWidth = 1163.5

  [[activatepl]]
    plslope = -1.8
    plbendloc = 90
    plbendwidth = 0.5
    uvslope = 0
    cutoff = 10000

  [[activatebol]]

  [[biattenuation]]
    OPT_index = -1.2
    NIR_index = -3.0
    norm = 1.2
    lam_break = 1100
    E(B-V) = 0.001, 0.01, 0.1
    E(B-V)-AGN = 0.001, 0.03, 0.3, 1.0
    filters =

  [[galdale2014]]
    alpha = 2.0
    lam_max = -1

  [[redshifting]]
    redshift =

[analysis_configuration]
  analysed_variables = stellar.mass_total, stellar.mass_alive, sfh.sfr, sfh.sfr100Myrs, agn.lumBolBBB, agn.lumBolTOR, agn.fracAGNDale
  save_best_sed = False
  save_chi2 = False
  save_pdf = False
  lim_flag = False
  mock_flag = False
"""
    config_path = work_dir / "pcigale.ini"
    config_path.write_text(textwrap.dedent(pcigale_ini).strip() + "\n", encoding="utf-8")
    return config_path


def _build_grahsp_env(args: argparse.Namespace, work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(args.cigale_root),
        str(PROJECT_ROOT / "src"),
        str(REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["OMP_NUM_THREADS"] = str(args.cores)
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["MPLCONFIGDIR"] = str(work_dir / "mplconfig")
    env["CACHE_MAX"] = str(args.cache_max)
    (work_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
    return env


def _run_grahsp(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.output_dir / "work" / f"{int(row['fit_index']):05d}_COSMOS{_safe_id(str(row['COSMOS_ID0']))}_{_safe_id(str(row['id']))}"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = _write_grahsp_input_table(row, work_dir)
    config_path = _write_pcigale_ini(input_path, work_dir)

    cmd = [
        str(args.python_executable),
        str(args.sampler_script),
        "analyse",
        "--cores",
        str(args.cores),
        "--num-live-points",
        str(args.num_live_points),
        "--num-posterior-samples",
        str(args.num_posterior_samples),
    ]
    if args.plot:
        cmd.append("--plot")

    completed = subprocess.run(
        cmd,
        cwd=work_dir,
        env=_build_grahsp_env(args, work_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    (work_dir / "grahsp_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "grahsp_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"GRAHSP failed with return code {completed.returncode}. "
            f"See {work_dir / 'grahsp_stderr.log'}"
        )

    summary_path = work_dir / "chimera_object.ecsv_analysis_results.txt"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing GRAHSP summary file: {summary_path}")
    summary = Table.read(summary_path, format="ascii.commented_header", delimiter="\t")

    recovered_logm = float(summary["log_stellar_mass_med"][0])
    logm_lo = float(summary["log_stellar_mass_lo"][0])
    logm_hi = float(summary["log_stellar_mass_hi"][0])
    truth_logm = float(row["log_stellar_mass_truth"])
    payload = {
        "status": "success",
        "fit_index": int(row["fit_index"]),
        "object_id": str(row["id"]),
        "COSMOS_ID0": str(row["COSMOS_ID0"]),
        "ID_COSMOS": str(row["ID_COSMOS"]),
        "redshift": float(row["redshift"]),
        "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
        "resample_weight": float(row["resample_weight"]),
        "log_stellar_mass_truth": truth_logm,
        "logLbol_QSO": float(row["logLbol_QSO"]),
        "logLbol_chimera": float(row["logLbol_chimera"]),
        "luminosity_bin": str(row["luminosity_bin"]),
        "recovered_logm": recovered_logm,
        "logm16": logm_lo,
        "logm84": logm_hi,
        "residual_log_ratio": float(recovered_logm - truth_logm),
        "work_dir": work_dir,
        "input_path": input_path,
        "config_path": config_path,
        "summary_path": summary_path,
        "fit_method": "grahsp",
        "optax_steps": "",
        "optax_lr": "",
        "nuts_warmup": "",
        "nuts_samples": "",
        "nuts_chains": "",
        "target_accept_prob": "",
    }
    for optional_name in (
        "log_stellar_mass_mean",
        "log_stellar_mass_std",
        "log_L_AGN_med",
        "chi2_med",
    ):
        if optional_name in summary.colnames:
            payload[optional_name] = float(summary[optional_name][0])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one GRAHSP fit from a precomputed manifest row.")
    parser.add_argument("--manifest", type=Path, default=Path("hpc_outputs/grahsp_loglbol_mass_retrieval/fit_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("hpc_outputs/grahsp_loglbol_mass_retrieval"))
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--array-offset", type=int, default=0, help="Add this offset after converting array-index to zero-based form.")
    parser.add_argument("--fit-index", type=int, default=None, help="Select an absolute manifest fit_index directly.")
    parser.add_argument("--cosmos-id0", default=None)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--expected-count", type=int, default=13558)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--sampler-script", type=Path, default=SAMPLER_SCRIPT)
    parser.add_argument("--cigale-root", type=Path, default=CIGALE_ROOT)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--num-live-points", type=int, default=800)
    parser.add_argument("--num-posterior-samples", type=int, default=3000)
    parser.add_argument("--cache-max", type=int, default=5000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.python_executable = args.python_executable.expanduser().resolve()
    args.sampler_script = args.sampler_script.expanduser().resolve()
    args.cigale_root = args.cigale_root.expanduser().resolve()

    raw = _select_manifest_entry(args)
    row = _row_from_manifest(raw)
    stem = f"{row['fit_index']:05d}_COSMOS{_safe_id(str(row['COSMOS_ID0']))}_{_safe_id(str(row['id']))}"
    success_path = args.output_dir / "results" / f"{stem}.json"
    failure_path = args.output_dir / "failures" / f"{stem}.json"

    print(
        f"[grahsp-manifest-fit] fit_index={row['fit_index']} COSMOS_ID0={row['COSMOS_ID0']} "
        f"object_id={row['id']}",
        flush=True,
    )

    if args.dry_run:
        payload = {
            "status": "dry_run",
            "fit_index": int(row["fit_index"]),
            "id": str(row["id"]),
            "COSMOS_ID0": str(row["COSMOS_ID0"]),
            "redshift": float(row["redshift"]),
            "logLbol_chimera": float(row["logLbol_chimera"]),
        }
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    try:
        payload = _run_grahsp(row, args)
        _atomic_write_json(success_path, payload)
        print(f"[grahsp-manifest-fit] wrote {success_path}", flush=True)
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "fit_index": int(row["fit_index"]),
            "object_id": str(row["id"]),
            "COSMOS_ID0": str(row["COSMOS_ID0"]),
            "ID_COSMOS": str(row["ID_COSMOS"]),
            "redshift": float(row["redshift"]),
            "chimera_QSO_weight": float(row["chimera_QSO_weight"]),
            "log_stellar_mass_truth": float(row["log_stellar_mass_truth"]),
            "logLbol_QSO": float(row["logLbol_QSO"]),
            "logLbol_chimera": float(row["logLbol_chimera"]),
            "luminosity_bin": str(row["luminosity_bin"]),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(failure_path, payload)
        print(f"[grahsp-manifest-fit] failed; wrote {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
