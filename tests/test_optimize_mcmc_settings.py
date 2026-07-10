from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "optimize_mcmc_settings.py"
SPEC = importlib.util.spec_from_file_location("optimize_mcmc_settings", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_select_row_supports_both_lookup_modes():
    rows = [{"object_id": "sky-id", "ID_COSMOS": "42"}]
    assert module._select_row(rows, "CHIMERA_ID", "sky-id") is rows[0]
    assert module._select_row(rows, "COSMOS_ID", "42") is rows[0]


def test_settings_grid_is_cartesian_product():
    args = module.build_parser().parse_args([
        "--warmup", "100,200", "--samples", "300", "--target-accept", "0.8,0.9",
        "--dense-mass", "false,true", "--tree-depth", "7",
    ])
    grid = module._settings_grid(args)
    assert len(grid) == 8
    assert grid[0] == {
        "num_warmup": 100, "num_samples": 300, "target_accept_prob": 0.8,
        "dense_mass": False, "max_tree_depth": 7,
    }
