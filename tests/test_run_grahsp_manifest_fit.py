from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import types

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


def test_grahsp_env_loads_compatibility_sitecustomize_first(tmp_path):
    args = argparse.Namespace(cigale_root=tmp_path / "GRAHSP", cores=1, cache_max=5000)

    env = run_grahsp_manifest_fit._build_grahsp_env(args, tmp_path)
    first_pythonpath = env["PYTHONPATH"].split(run_grahsp_manifest_fit.os.pathsep)[0]

    assert first_pythonpath.endswith("hpc/grahsp_compat")
    assert env["PLOT_CORNER"] == "1"
    assert env["PLOT_TRACE"] == "1"


def test_grahsp_command_always_requests_plot_summary(tmp_path):
    args = argparse.Namespace(
        python_executable=tmp_path / "python",
        sampler_script=tmp_path / "dualsampler.py",
        cores=1,
        num_live_points=800,
        num_posterior_samples=3000,
    )

    cmd = run_grahsp_manifest_fit._build_grahsp_command(args)

    assert "--plot" in cmd
    assert "--mass-max" not in cmd


def test_collect_grahsp_artifacts_copies_standard_outputs(tmp_path):
    work_dir = tmp_path / "work" / "00001_COSMOS10_obj-a"
    plot_dir = work_dir / "grahsp_obj-a_varV2" / "plots"
    output_dir = tmp_path / "out"
    plot_dir.mkdir(parents=True)
    for name in (
        "sed_mJy.pdf",
        "sed_lum.pdf",
        "corner.pdf",
        "posteriors.pdf",
        "derived.pdf",
        "sed_mJy.csv.gz",
        "sed_lum.csv.gz",
        "trace.pdf",
    ):
        (plot_dir / name).write_text(name, encoding="utf-8")

    artifacts = run_grahsp_manifest_fit._collect_grahsp_artifacts(work_dir, output_dir, "00001_COSMOS10_obj-a")

    assert artifacts["grahsp_plot_dir"] == str(plot_dir)
    assert artifacts["sed_pdf_path"] == str(output_dir / "sed_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["sed_lum_pdf_path"] == str(output_dir / "sed_lum_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["corner_pdf_path"] == str(output_dir / "corner_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["trace_pdf_path"] == str(output_dir / "trace_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["sed_mjy_csv_path"] == str(output_dir / "sed_csvs" / "00001_COSMOS10_obj-a_mJy.csv.gz")
    assert (output_dir / "sed_pdfs" / "00001_COSMOS10_obj-a.pdf").read_text(encoding="utf-8") == "sed_mJy.pdf"
    assert str(plot_dir / "sed_lum.pdf") in artifacts["grahsp_artifact_paths"]


def test_sitecustomize_accepts_tuple_sfh(monkeypatch):
    class _SED:
        def __init__(self):
            self.info = {}
            self._sfh = None

        @property
        def sfh(self):
            return self._sfh

        @sfh.setter
        def sfh(self, value):
            self._sfh = value

        def add_info(self, key, value, *args, **kwargs):
            self.info[key] = value

    pcigale = types.ModuleType("pcigale")
    sed = types.ModuleType("pcigale.sed")
    sed.SED = _SED
    monkeypatch.setitem(sys.modules, "pcigale", pcigale)
    monkeypatch.setitem(sys.modules, "pcigale.sed", sed)

    sitecustomize_path = run_grahsp_manifest_fit.PROJECT_ROOT / "hpc" / "grahsp_compat" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("grahsp_compat_sitecustomize_test", sitecustomize_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    instance = _SED()
    instance.sfh = ([0.0, 1.0, 2.0], [0.5, 1.5, 2.5])

    assert instance.sfh.tolist() == [0.5, 1.5, 2.5]
    assert instance.info["sfh.sfr"] == 2.5
    assert instance.info["sfh.age"] == 3
