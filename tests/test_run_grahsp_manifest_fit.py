from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import math
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
    for name in ("sed_mJy.pdf", "sed_lum.pdf", "corner.pdf", "posteriors.pdf", "derived.pdf", "trace.pdf"):
        (plot_dir / name).write_text(name, encoding="utf-8")
    sed_csv = "\n".join(
        [
            "wavelength,total,Stellar (attenuated),AGN disk",
            "0.255,10,4,2",
            "0.510,20,8,4",
            "1.020,30,16,8",
        ]
    ) + "\n"
    with gzip.open(plot_dir / "sed_mJy.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write(sed_csv)
    with gzip.open(plot_dir / "sed_lum.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write(sed_csv)

    row = {
        "redshift": 1.0,
        **{
            name: float(i + 1)
            for i, name in enumerate(run_grahsp_manifest_fit.CHIMERA_FILTER_NAMES)
        },
        **{
            f"{name}_err": 0.1 * float(i + 1)
            for i, name in enumerate(run_grahsp_manifest_fit.CHIMERA_FILTER_NAMES)
        },
    }

    artifacts = run_grahsp_manifest_fit._collect_grahsp_artifacts(work_dir, output_dir, "00001_COSMOS10_obj-a", row)

    assert artifacts["grahsp_plot_dir"] == str(plot_dir)
    assert artifacts["sed_pdf_path"] == str(output_dir / "sed_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["sed_lum_pdf_path"] == str(output_dir / "sed_lum_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["corner_pdf_path"] == str(output_dir / "corner_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["trace_pdf_path"] == str(output_dir / "trace_pdfs" / "00001_COSMOS10_obj-a.pdf")
    assert artifacts["sed_mjy_csv_path"] == str(output_dir / "sed_csvs" / "00001_COSMOS10_obj-a_mJy.csv.gz")
    assert artifacts["notebook_sed_csv_path"] == str(output_dir / "notebook_sed_csvs" / "00001_COSMOS10_obj-a_notebook_sed.csv")
    assert artifacts["photometry_csv_path"] == str(output_dir / "photometry_csvs" / "00001_COSMOS10_obj-a_photometry.csv")
    assert (output_dir / "sed_pdfs" / "00001_COSMOS10_obj-a.pdf").read_text(encoding="utf-8") == "sed_mJy.pdf"
    assert str(plot_dir / "sed_lum.pdf") in artifacts["grahsp_artifact_paths"]


def test_write_grahsp_notebook_sed_adds_rest_frame_and_shape_columns(tmp_path):
    sed_path = tmp_path / "sed_mJy.csv"
    sed_path.write_text(
        "\n".join(
            [
                "wavelength,total,total_errup,total_errlo,Stellar (attenuated),Stellar (attenuated)_errup,Stellar (attenuated)_errlo,AGN disk,AGN disk_errup,AGN disk_errlo",
                "0.255,10,12,8,4,5,3,2,3,1",
                "0.510,20,22,18,8,9,7,4,5,3",
                "1.020,30,33,27,16,18,14,8,10,6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "notebook_sed.csv"

    norms = run_grahsp_manifest_fit._write_grahsp_notebook_sed(sed_path, output_path, redshift=0.0)
    rows = list(csv.DictReader(open(output_path, encoding="utf-8")))

    assert rows[0]["wavelength_obs_a"] == "2550.0"
    assert rows[1]["wavelength_rest_a"] == "5100.0"
    assert "stellar_attenuated_mjy" in rows[0]
    assert "stellar_attenuated_flambda_shape_norm5100" in rows[0]
    assert math.isclose(float(rows[1]["stellar_attenuated_flambda_shape_norm5100"]), 1.0)
    assert math.isclose(float(rows[1]["agn_disk_flambda_shape_norm5100"]), 1.0)
    assert math.isfinite(norms["stellar_attenuated_flambda_shape_norm5100_scale"])


def test_write_grahsp_photometry_table_includes_effective_wavelengths(tmp_path):
    row = {}
    for i, name in enumerate(run_grahsp_manifest_fit.CHIMERA_FILTER_NAMES):
        row[name] = float(i + 1)
        row[f"{name}_err"] = 0.1 * float(i + 1)
    output_path = tmp_path / "photometry.csv"

    run_grahsp_manifest_fit._write_grahsp_photometry_table(row, output_path)
    rows = list(csv.DictReader(open(output_path, encoding="utf-8")))

    assert rows[0]["chimera_filter"] == "u_sdss"
    assert rows[0]["effective_wavelength_a"] == "3543.0"
    assert rows[-1]["grahsp_filter"] == "IRAC2"
    assert rows[-1]["effective_wavelength_a"] == "44930.0"


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
