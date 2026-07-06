from __future__ import annotations

import csv

import pytest

from hpc import run_grahsp_jaxsedfit_comparison as comparison


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fit_index", "object_id", "COSMOS_ID0"])
        writer.writeheader()
        writer.writerows(rows)


def test_select_manifest_row_requires_unique_object_id(tmp_path):
    manifest = tmp_path / "fit_manifest.csv"
    _write_manifest(
        manifest,
        [
            {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
            {"fit_index": "1", "object_id": "obj-b", "COSMOS_ID0": "11"},
        ],
    )

    row = comparison._select_manifest_row(manifest, "obj-b")

    assert row["fit_index"] == "1"
    assert row["COSMOS_ID0"] == "11"


def test_select_manifest_row_rejects_missing_object(tmp_path):
    manifest = tmp_path / "fit_manifest.csv"
    _write_manifest(manifest, [{"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"}])

    with pytest.raises(RuntimeError, match="matched 0"):
        comparison._select_manifest_row(manifest, "missing")


def test_find_jaxsedfit_root_accepts_src_layout(tmp_path):
    root = tmp_path / "jaxsedfit"
    (root / "src" / "jaxsedfit").mkdir(parents=True)

    assert comparison._find_jaxsedfit_root(root) == root.resolve()
