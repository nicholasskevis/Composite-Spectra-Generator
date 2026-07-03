from __future__ import annotations

import argparse
import csv
import importlib
import sys
import types

import pytest


@pytest.fixture()
def run_manifest_fit(monkeypatch):
    class _Array(list):
        def reshape(self, *args):
            return self

    def _asarray(value, dtype=None):
        if isinstance(value, _Array):
            return value
        if isinstance(value, list):
            return _Array(value)
        return value

    def _percentile(values, quantiles):
        ordered = sorted(values)
        if list(quantiles) == [16.0, 50.0, 84.0]:
            return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        raise NotImplementedError

    numpy = types.ModuleType("numpy")
    numpy.ndarray = _Array
    numpy.generic = type("generic", (), {"item": lambda self: self})
    numpy.asarray = _asarray
    numpy.percentile = _percentile

    for package_name in ("jaxsedfit", "grahspj"):
        benchmark = types.ModuleType(f"{package_name}.benchmark")
        benchmark.CHIMERA_FILTER_NAMES = []
        benchmark.build_chimera_fit_config = lambda *args, **kwargs: None

        core = types.ModuleType(f"{package_name}.core")
        core.JAXSEDFit = object
        if package_name == "grahspj":
            core.GRAHSPJ = object

        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
        monkeypatch.setitem(sys.modules, f"{package_name}.benchmark", benchmark)
        monkeypatch.setitem(sys.modules, f"{package_name}.core", core)

    monkeypatch.setitem(sys.modules, "numpy", numpy)
    sys.modules.pop("hpc.run_manifest_fit", None)
    module = importlib.import_module("hpc.run_manifest_fit")
    yield module
    sys.modules.pop("hpc.run_manifest_fit", None)


def _write_manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fit_index", "object_id", "COSMOS_ID0"])
        writer.writeheader()
        writer.writerows(rows)


def _args(manifest, object_id, expected_count=None):
    return argparse.Namespace(manifest=manifest, object_id=object_id, expected_count=expected_count)


def _fit_args(tmp_path):
    return argparse.Namespace(
        dsps_ssp_fn=tmp_path / "ssp.h5",
        seed_base=100,
        sampler="optax+nuts",
        optax_steps=300,
        optax_lr=1.0e-2,
        nuts_warmup=500,
        nuts_samples=300,
        nuts_chains=1,
        ns_live_points=None,
        ns_max_samples=None,
        ns_dlogz=None,
        ns_resamples=None,
        ns_difficult_model=False,
        ns_parameter_estimation=False,
        ns_num_parallel_workers=None,
        ns_init_efficiency_threshold=None,
        ns_max_likelihood_evals=None,
        ns_efficiency_threshold=None,
        target_accept_prob=0.85,
        progress_bar=False,
        backend="jaxsedfit",
    )


def test_backend_aliases_resolve_to_installed_jaxsedfit_package(run_manifest_fit):
    assert run_manifest_fit._normalize_backend("jaxsedfit") == "jaxsedfit"
    assert run_manifest_fit._normalize_backend("jaxsed") == "jaxsedfit"
    assert run_manifest_fit._normalize_backend("grahspj") == "jaxsedfit"


def test_select_manifest_entry_uses_object_id(run_manifest_fit, tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
            {"fit_index": "1", "object_id": "obj-b", "COSMOS_ID0": "11"},
        ],
    )

    row = run_manifest_fit._select_manifest_entry(_args(manifest, "obj-b", expected_count=2))

    assert row["fit_index"] == "1"
    assert row["object_id"] == "obj-b"


def test_object_id_selection_ignores_scheduler_array_env(run_manifest_fit, tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
            {"fit_index": "10000", "object_id": "obj-second-chunk", "COSMOS_ID0": "11"},
        ],
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    row = run_manifest_fit._select_manifest_entry(_args(manifest, "obj-second-chunk"))

    assert row["fit_index"] == "10000"


def test_select_manifest_entry_requires_object_id(run_manifest_fit, tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [{"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"}])

    with pytest.raises(RuntimeError, match="--object-id is required"):
        run_manifest_fit._select_manifest_entry(_args(manifest, None))


def test_select_manifest_entry_rejects_unknown_object_id(run_manifest_fit, tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [{"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"}])

    with pytest.raises(RuntimeError, match="matched 0 manifest rows"):
        run_manifest_fit._select_manifest_entry(_args(manifest, "missing"))


def test_select_manifest_entry_rejects_duplicate_object_id(run_manifest_fit, tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
            {"fit_index": "1", "object_id": "obj-a", "COSMOS_ID0": "11"},
        ],
    )

    with pytest.raises(RuntimeError, match="matched 2 manifest rows"):
        run_manifest_fit._select_manifest_entry(_args(manifest, "obj-a"))


def test_patch_backend_config_compat_adds_missing_speclite_names(run_manifest_fit):
    filters = types.SimpleNamespace(curves=[], use_grahsp_database=False)
    cfg = types.SimpleNamespace(filters=filters)

    run_manifest_fit._patch_backend_config_compat(cfg)

    assert filters.speclite_names == {}


def test_patch_backend_config_compat_preserves_existing_speclite_names(run_manifest_fit):
    filters = types.SimpleNamespace(speclite_names={"u_sdss": "sdss2010-u"})
    cfg = types.SimpleNamespace(filters=filters)

    run_manifest_fit._patch_backend_config_compat(cfg)

    assert filters.speclite_names == {"u_sdss": "sdss2010-u"}

def test_run_fit_saves_sed_pdf_and_records_path(run_manifest_fit, tmp_path, monkeypatch):
    captured = {}

    cfg = types.SimpleNamespace(
        inference=types.SimpleNamespace(seed=None),
        prior_config={"log_stellar_mass": {"loc": 10.0}},
        galaxy=types.SimpleNamespace(dsps_ssp_fn=str(tmp_path / "ssp.h5")),
    )
    class _FakeFitter:
        def __init__(self, config):
            self.config = config
            self.samples = {"log_stellar_mass": [9.0, 10.0, 11.0]}

        def fit(self, **kwargs):
            captured["fit"] = kwargs
            return {"summary": {"converged": True}}

        def plot_corner(self, **kwargs):
            captured["corner"] = kwargs
            return None

        def plot_trace(self, **kwargs):
            captured["trace"] = kwargs
            return None

    monkeypatch.setattr(
        run_manifest_fit,
        "_load_backend",
        lambda backend: ([], lambda row, dsps_ssp_fn: cfg, _FakeFitter),
    )

    row = {
        "fit_index": 7,
        "id": "obj-a",
        "COSMOS_ID0": "10",
        "ID_COSMOS": "10",
        "redshift": 0.5,
        "chimera_QSO_weight": 0.25,
        "resample_weight": 1.0,
        "log_stellar_mass_truth": 10.0,
        "logLbol_QSO": 44.0,
        "logLbol_chimera": 44.1,
        "luminosity_bin": "low",
    }
    sed_pdf_path = tmp_path / "sed_pdfs" / "00007_COSMOS10_obj-a.pdf"
    corner_pdf_path = tmp_path / "corner_pdfs" / "00007_COSMOS10_obj-a.pdf"
    trace_pdf_path = tmp_path / "trace_pdfs" / "00007_COSMOS10_obj-a.pdf"

    payload = run_manifest_fit._run_fit(row, _fit_args(tmp_path), sed_pdf_path, corner_pdf_path, trace_pdf_path)

    assert captured["fit"]["fit_method"] == "optax+nuts"
    assert captured["fit"]["save_fig"] is True
    assert captured["fit"]["fig_path"] == sed_pdf_path
    assert captured["fit"]["fig_path"].suffix == ".pdf"
    assert captured["fit"]["fig_path"].parent.name == "sed_pdfs"
    assert captured["corner"]["output_path"] == corner_pdf_path
    assert captured["corner"]["output_path"].suffix == ".pdf"
    assert captured["corner"]["output_path"].parent.name == "corner_pdfs"
    assert captured["trace"]["output_path"] == trace_pdf_path
    assert captured["trace"]["output_path"].suffix == ".pdf"
    assert captured["trace"]["output_path"].parent.name == "trace_pdfs"
    assert payload["sed_pdf_path"] == str(sed_pdf_path)
    assert payload["corner_pdf_path"] == str(corner_pdf_path)
    assert payload["trace_pdf_path"] == str(trace_pdf_path)
    assert payload["sampler"] == "optax+nuts"


def test_run_fit_passes_nested_sampler_options(run_manifest_fit, tmp_path, monkeypatch):
    captured = {}

    cfg = types.SimpleNamespace(
        inference=types.SimpleNamespace(seed=None),
        prior_config={"log_stellar_mass": {"loc": 10.0}},
        galaxy=types.SimpleNamespace(dsps_ssp_fn=str(tmp_path / "ssp.h5")),
    )
    class _FakeFitter:
        def __init__(self, config):
            self.config = config
            self.samples = {"log_stellar_mass": [9.0, 10.0, 11.0]}

        def fit(self, **kwargs):
            captured["fit"] = kwargs
            return {"summary": {"converged": True}}

        def plot_corner(self, **kwargs):
            return None

        def plot_trace(self, **kwargs):
            return None

    monkeypatch.setattr(
        run_manifest_fit,
        "_load_backend",
        lambda backend: ([], lambda row, dsps_ssp_fn: cfg, _FakeFitter),
    )

    args = _fit_args(tmp_path)
    args.sampler = "ns"
    args.ns_live_points = 700
    args.ns_max_samples = 8000
    args.ns_dlogz = 10.0
    args.ns_resamples = 2000
    args.ns_difficult_model = True
    args.ns_parameter_estimation = True
    args.ns_num_parallel_workers = 4
    args.ns_init_efficiency_threshold = 0.2
    args.ns_max_likelihood_evals = 100_000
    args.ns_efficiency_threshold = 0.001

    row = {
        "fit_index": 7,
        "id": "obj-a",
        "COSMOS_ID0": "10",
        "ID_COSMOS": "10",
        "redshift": 0.5,
        "chimera_QSO_weight": 0.25,
        "resample_weight": 1.0,
        "log_stellar_mass_truth": 10.0,
        "logLbol_QSO": 44.0,
        "logLbol_chimera": 44.1,
        "luminosity_bin": "low",
    }

    payload = run_manifest_fit._run_fit(
        row,
        args,
        tmp_path / "sed_pdfs" / "00007_COSMOS10_obj-a.pdf",
        tmp_path / "corner_pdfs" / "00007_COSMOS10_obj-a.pdf",
        tmp_path / "trace_pdfs" / "00007_COSMOS10_obj-a.pdf",
    )

    assert captured["fit"]["fit_method"] == "ns"
    assert captured["fit"]["ns_live_points"] == 700
    assert captured["fit"]["ns_max_samples"] == 8000
    assert captured["fit"]["ns_dlogz"] == 10.0
    assert captured["fit"]["ns_resamples"] == 2000
    assert captured["fit"]["ns_difficult_model"] is True
    assert captured["fit"]["ns_parameter_estimation"] is True
    assert captured["fit"]["ns_num_parallel_workers"] == 4
    assert captured["fit"]["ns_init_efficiency_threshold"] == 0.2
    assert captured["fit"]["ns_max_likelihood_evals"] == 100_000
    assert captured["fit"]["ns_efficiency_threshold"] == 0.001
    assert payload["sampler"] == "ns"
    assert payload["ns_live_points"] == 700
    assert payload["ns_max_samples"] == 8000
    assert payload["ns_dlogz"] == 10.0
    assert payload["ns_resamples"] == 2000
    assert payload["ns_difficult_model"] is True
    assert payload["ns_parameter_estimation"] is True
    assert payload["ns_num_parallel_workers"] == 4
    assert payload["ns_init_efficiency_threshold"] == 0.2
    assert payload["ns_max_likelihood_evals"] == 100_000
    assert payload["ns_efficiency_threshold"] == 0.001
