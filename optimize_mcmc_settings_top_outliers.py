#!/usr/bin/env python3
"""Grid-search JAXSEDFit MCMC settings for the largest Chimera mass outliers."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import optimize_mcmc_settings as single_object


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_OUTLIERS = HERE / "top100_mass_retrieval_outliers_per_logLbol_bin.csv"
DEFAULT_OUTPUT_DIR = HERE / "notebook_outputs" / "top_outlier_mcmc_setting_optimization"
MCMC_SETTING_NAMES = ("num_warmup", "num_samples", "target_accept_prob", "dense_mass", "max_tree_depth")


def _float_value(row: dict[str, str], name: str, default: float = np.nan) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return default


def _settings_key(settings: dict[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _load_top_outliers(path: Path, limit: int, rank_column: str) -> list[dict[str, str]]:
    with path.expanduser().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"outlier table is empty: {path}")
    if "object_id" not in rows[0]:
        raise ValueError(f"outlier table must contain object_id: {path}")
    if rank_column not in rows[0]:
        raise ValueError(f"outlier table must contain rank column {rank_column!r}: {path}")

    rows = sorted(rows, key=lambda row: abs(_float_value(row, rank_column)), reverse=True)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        object_id = str(row["object_id"])
        if object_id in seen:
            continue
        seen.add(object_id)
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _load_existing_trials(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add((str(record["object_id"]), _settings_key(record["settings"])))
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_trials(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flat_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {f"setting_{key}": value for key, value in settings.items()}


def _settings_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = (*MCMC_SETTING_NAMES, "map_steps", "learning_rate")
    values = (
        args.warmup,
        args.samples,
        args.target_accept,
        args.dense_mass,
        args.tree_depth,
        args.map_steps,
        args.learning_rate,
    )
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def _run_one_setting(
    row: dict[str, Any],
    dsps_ssp_fn: Path,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_args = argparse.Namespace(**vars(args))
    run_args.map_steps = int(settings["map_steps"])
    run_args.learning_rate = float(settings["learning_rate"])
    mcmc_settings = {name: settings[name] for name in MCMC_SETTING_NAMES}
    return single_object._run_one(row, dsps_ssp_fn, mcmc_settings, run_args)


def summarize_trials(trials_path: Path, output_dir: Path, expected_object_count: int | None = None) -> dict[str, Any]:
    records = _read_trials(trials_path)
    successes = [record for record in records if record.get("status") == "success"]
    if not successes:
        raise RuntimeError(f"no successful trials to summarize in {trials_path}")

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_settings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in successes:
        by_object[str(record["object_id"])].append(record)
        by_settings[_settings_key(record["settings"])].append(record)

    per_object_rows = []
    for object_id, object_records in sorted(by_object.items()):
        best = min(object_records, key=lambda record: record["absolute_residual_dex"])
        per_object_rows.append({
            "object_id": object_id,
            "fit_index": best.get("fit_index", ""),
            "luminosity_bin": best.get("luminosity_bin", ""),
            "truth_log_stellar_mass": best.get("truth_log_stellar_mass", ""),
            "best_recovered_log_stellar_mass": best.get("recovered_log_stellar_mass", ""),
            "best_residual_dex": best.get("residual_dex", ""),
            "best_absolute_residual_dex": best.get("absolute_residual_dex", ""),
            "posterior_p16": best.get("posterior_16_84", ["", ""])[0],
            "posterior_p84": best.get("posterior_16_84", ["", ""])[1],
            **_flat_settings(best["settings"]),
        })

    global_rows = []
    for key, setting_records in sorted(by_settings.items()):
        residuals = np.asarray([record["residual_dex"] for record in setting_records], dtype=float)
        abs_residuals = np.abs(residuals)
        global_rows.append({
            "settings_key": key,
            "n_success": int(len(setting_records)),
            "n_objects_success": int(len({record["object_id"] for record in setting_records})),
            "mean_residual_dex": float(np.mean(residuals)),
            "std_residual_dex": float(np.std(residuals)),
            "mean_absolute_residual_dex": float(np.mean(abs_residuals)),
            "median_absolute_residual_dex": float(np.median(abs_residuals)),
            "rms_residual_dex": float(np.sqrt(np.mean(np.square(residuals)))),
            "max_absolute_residual_dex": float(np.max(abs_residuals)),
            **_flat_settings(setting_records[0]["settings"]),
        })

    global_rows.sort(
        key=lambda row: (
            -row["n_objects_success"],
            row["mean_absolute_residual_dex"],
            row["median_absolute_residual_dex"],
            row["max_absolute_residual_dex"],
        )
    )
    best_global = global_rows[0]

    bin_rows = []
    by_bin_settings: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in successes:
        by_bin_settings[(str(record.get("luminosity_bin", "")), _settings_key(record["settings"]))].append(record)
    for (luminosity_bin, key), setting_records in sorted(by_bin_settings.items()):
        residuals = np.asarray([record["residual_dex"] for record in setting_records], dtype=float)
        abs_residuals = np.abs(residuals)
        bin_rows.append({
            "luminosity_bin": luminosity_bin,
            "settings_key": key,
            "n_success": int(len(setting_records)),
            "n_objects_success": int(len({record["object_id"] for record in setting_records})),
            "mean_residual_dex": float(np.mean(residuals)),
            "std_residual_dex": float(np.std(residuals)),
            "mean_absolute_residual_dex": float(np.mean(abs_residuals)),
            "median_absolute_residual_dex": float(np.median(abs_residuals)),
            "rms_residual_dex": float(np.sqrt(np.mean(np.square(residuals)))),
            "max_absolute_residual_dex": float(np.max(abs_residuals)),
            **_flat_settings(setting_records[0]["settings"]),
        })
    bin_rows.sort(
        key=lambda row: (
            row["luminosity_bin"],
            -row["n_objects_success"],
            row["mean_absolute_residual_dex"],
            row["median_absolute_residual_dex"],
            row["max_absolute_residual_dex"],
        )
    )
    best_by_bin = {}
    for row in bin_rows:
        best_by_bin.setdefault(row["luminosity_bin"], row)

    failed_records = [record for record in records if record.get("status") != "success"]
    summary = {
        "trials_path": str(trials_path),
        "n_trials": len(records),
        "n_success": len(successes),
        "n_failed": len(failed_records),
        "n_objects_with_success": len(by_object),
        "n_expected_objects": expected_object_count,
        "best_global_settings": {
            key.replace("setting_", ""): value
            for key, value in best_global.items()
            if key.startswith("setting_")
        },
        "best_global_metrics": {
            key: value
            for key, value in best_global.items()
            if not key.startswith("setting_") and key != "settings_key"
        },
        "best_settings_by_luminosity_bin": {
            luminosity_bin: {
                key.replace("setting_", ""): value
                for key, value in row.items()
                if key.startswith("setting_")
            }
            for luminosity_bin, row in best_by_bin.items()
        },
        "best_metrics_by_luminosity_bin": {
            luminosity_bin: {
                key: value
                for key, value in row.items()
                if not key.startswith("setting_") and key != "settings_key"
            }
            for luminosity_bin, row in best_by_bin.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "best_settings_per_object.csv", per_object_rows)
    _write_csv(output_dir / "settings_global_ranking.csv", global_rows)
    _write_csv(output_dir / "settings_by_luminosity_bin_ranking.csv", bin_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outliers-csv", type=Path, default=DEFAULT_OUTLIERS)
    parser.add_argument("--limit", type=int, default=100, help="Use the largest N unique outliers; 0 means all rows.")
    parser.add_argument("--rank-column", default="abs_residual_log_ratio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jaxsedfit-root", type=Path, default=single_object.DEFAULT_JAXSEDFIT_ROOT)
    parser.add_argument("--manifest", type=Path, default=single_object.DEFAULT_MANIFEST)
    parser.add_argument("--dsps-ssp-fn", type=Path)
    parser.add_argument("--warmup", type=lambda x: single_object._csv_values(x, int), default=[500, 1000])
    parser.add_argument("--samples", type=lambda x: single_object._csv_values(x, int), default=[300, 500])
    parser.add_argument("--target-accept", type=lambda x: single_object._csv_values(x, float), default=[0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--dense-mass", type=lambda x: single_object._csv_values(x, single_object._bool_value), default=[False, True])
    parser.add_argument("--tree-depth", type=lambda x: single_object._csv_values(x, int), default=[6, 8, 10])
    parser.add_argument("--map-steps", type=lambda x: single_object._csv_values(x, int), default=[300, 500])
    parser.add_argument("--learning-rate", type=lambda x: single_object._csv_values(x, float), default=[3e-3, 5e-3])
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true", help="Rerun trials already present in trials.jsonl.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    trials_path = output_dir / "trials.jsonl"

    if args.summarize_only:
        summary = summarize_trials(trials_path, output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    root = args.jaxsedfit_root.expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    from jaxsedfit.benchmark import CHIMERA_FILTER_NAMES

    selected = _load_top_outliers(args.outliers_csv, args.limit, args.rank_column)
    grid = _settings_grid(args)
    print(f"Selected objects: {len(selected)}", flush=True)
    print(f"Settings per object: {len(grid)}", flush=True)
    print(f"Planned trials: {len(selected) * len(grid)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    dsps_ssp_fn = single_object._find_dsps_file(root, args.dsps_ssp_fn)
    completed = set() if args.rerun else _load_existing_trials(trials_path)

    for object_number, outlier in enumerate(selected, start=1):
        object_id = str(outlier["object_id"])
        row = single_object._load_manifest_row(
            args.manifest, "CHIMERA_ID", object_id, CHIMERA_FILTER_NAMES
        )
        for setting_number, settings in enumerate(grid, start=1):
            full_settings = {**settings, "num_chains": args.chains, "seed": args.seed}
            key = (object_id, _settings_key(full_settings))
            if key in completed:
                continue
            print(
                f"[{object_number}/{len(selected)}] {object_id} setting {setting_number}/{len(grid)}: {settings}",
                flush=True,
            )
            base_record = {
                "object_id": object_id,
                "fit_index": row.get("fit_index", ""),
                "luminosity_bin": row.get("luminosity_bin", ""),
                "truth_log_stellar_mass": float(row["log_stellar_mass_truth"]),
                "initial_residual_log_ratio": _float_value(outlier, "residual_log_ratio"),
                "initial_abs_residual_log_ratio": abs(_float_value(outlier, "residual_log_ratio")),
            }
            try:
                result = _run_one_setting(row, dsps_ssp_fn, settings, args)
                _append_jsonl(trials_path, {**base_record, **result, "status": "success"})
            except Exception as exc:
                full_settings = {**settings, "num_chains": args.chains, "seed": args.seed}
                _append_jsonl(trials_path, {
                    **base_record,
                    "settings": full_settings,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })

    summary = summarize_trials(trials_path, output_dir, expected_object_count=len(selected))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
