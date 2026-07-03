from __future__ import annotations

import argparse
import csv

from hpc import run_grahsp_manifest_fit


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fit_index", "object_id", "COSMOS_ID0"])
        writer.writeheader()
        writer.writerows(rows)


def _args(manifest, **overrides):
    defaults = {
        "manifest": manifest,
        "expected_count": None,
        "cosmos_id0": None,
        "object_id": None,
        "fit_index": None,
        "array_index": None,
        "array_offset": 0,
        "index_base": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_fit_index_selection_ignores_scheduler_array_env(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
            {"fit_index": "1", "object_id": "obj-b", "COSMOS_ID0": "11"},
        ],
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    row = run_grahsp_manifest_fit._select_manifest_entry(_args(manifest, fit_index=1))

    assert row["fit_index"] == "1"
    assert row["object_id"] == "obj-b"


def test_pcigale_ini_includes_scaling_limits(tmp_path):
    input_path = tmp_path / "chimera_object.ecsv"
    input_path.write_text("# input placeholder\n", encoding="utf-8")

    config_path = run_grahsp_manifest_fit._write_pcigale_ini(input_path, tmp_path)
    config = config_path.read_text(encoding="utf-8")

    assert "[scaling_limits]" in config
    assert "mass_min = 5" in config
    assert "mass_max = 15" in config
    assert "sfr_min = 0" in config
    assert "sfr_max = 100000" in config
    assert "L_min = 38" in config
    assert "L_max = 50" in config
