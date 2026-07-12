from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "optimize_mcmc_settings_top_outliers.py"
SPEC = importlib.util.spec_from_file_location("optimize_mcmc_settings_top_outliers", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_load_top_outliers_sorts_deduplicates_and_limits(tmp_path):
    path = tmp_path / "outliers.csv"
    path.write_text(
        "object_id,abs_residual_log_ratio\n"
        "a,1.0\n"
        "b,3.0\n"
        "a,5.0\n"
        "c,2.0\n",
        encoding="utf-8",
    )

    rows = module._load_top_outliers(path, limit=2, rank_column="abs_residual_log_ratio")

    assert [row["object_id"] for row in rows] == ["a", "b"]


def test_settings_grid_includes_map_and_learning_rate():
    args = module.build_parser().parse_args([
        "--warmup", "100",
        "--samples", "50",
        "--target-accept", "0.8",
        "--dense-mass", "false",
        "--tree-depth", "6",
        "--map-steps", "300,500",
        "--learning-rate", "0.003,0.005",
    ])

    grid = module._settings_grid(args)

    assert len(grid) == 4
    assert grid[0]["map_steps"] == 300
    assert grid[-1]["learning_rate"] == 0.005


def test_summarize_trials_reports_per_object_and_global_best(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    trials = output_dir / "trials.jsonl"
    records = [
        {
            "object_id": "a",
            "luminosity_bin": "L < 42",
            "settings": {"num_warmup": 100, "num_samples": 50},
            "status": "success",
            "absolute_residual_dex": 0.2,
            "residual_dex": 0.2,
            "recovered_log_stellar_mass": 10.2,
            "truth_log_stellar_mass": 10.0,
            "posterior_16_84": [10.0, 10.4],
        },
        {
            "object_id": "a",
            "luminosity_bin": "L < 42",
            "settings": {"num_warmup": 200, "num_samples": 50},
            "status": "success",
            "absolute_residual_dex": 0.1,
            "residual_dex": 0.1,
            "recovered_log_stellar_mass": 10.1,
            "truth_log_stellar_mass": 10.0,
            "posterior_16_84": [9.9, 10.3],
        },
        {
            "object_id": "b",
            "luminosity_bin": "42 < L < 43",
            "settings": {"num_warmup": 100, "num_samples": 50},
            "status": "success",
            "absolute_residual_dex": 0.3,
            "residual_dex": -0.3,
            "recovered_log_stellar_mass": 9.7,
            "truth_log_stellar_mass": 10.0,
            "posterior_16_84": [9.5, 9.9],
        },
        {
            "object_id": "b",
            "luminosity_bin": "42 < L < 43",
            "settings": {"num_warmup": 200, "num_samples": 50},
            "status": "success",
            "absolute_residual_dex": 0.4,
            "residual_dex": -0.4,
            "recovered_log_stellar_mass": 9.6,
            "truth_log_stellar_mass": 10.0,
            "posterior_16_84": [9.4, 9.8],
        },
    ]
    trials.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summary = module.summarize_trials(trials, output_dir, expected_object_count=2)

    assert summary["n_objects_with_success"] == 2
    assert summary["best_global_settings"] == {"num_warmup": 100, "num_samples": 50}
    assert summary["best_settings_by_luminosity_bin"]["L < 42"] == {"num_warmup": 200, "num_samples": 50}
    assert summary["best_settings_by_luminosity_bin"]["42 < L < 43"] == {"num_warmup": 100, "num_samples": 50}
    assert (output_dir / "best_settings_per_object.csv").is_file()
    assert (output_dir / "settings_global_ranking.csv").is_file()
    assert (output_dir / "settings_by_luminosity_bin_ranking.csv").is_file()


def test_summarize_trials_reads_slurm_trial_json_files(tmp_path):
    output_dir = tmp_path / "out"
    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True)
    (trial_dir / "000000_a.json").write_text(json.dumps({
        "object_id": "a",
        "luminosity_bin": "L < 42",
        "settings": {"num_warmup": 100},
        "status": "success",
        "absolute_residual_dex": 0.2,
        "residual_dex": 0.2,
        "recovered_log_stellar_mass": 10.2,
        "truth_log_stellar_mass": 10.0,
        "posterior_16_84": [10.0, 10.4],
    }), encoding="utf-8")

    summary = module.summarize_trials(output_dir / "trials.jsonl", output_dir, expected_object_count=1)

    assert summary["n_success"] == 1
    assert summary["best_global_settings"] == {"num_warmup": 100}
