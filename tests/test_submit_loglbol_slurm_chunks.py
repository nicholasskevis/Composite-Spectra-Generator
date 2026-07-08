from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from hpc.submit_loglbol_slurm_chunks import (
    _batch_script,
    _chunks,
    _filter_missing_tasks,
    _resolve_first_existing,
    _run_name,
    _safe_run_label,
)


def test_run_name_uses_month_day_time_and_label():
    assert _run_name("nicholas", datetime(2026, 6, 10, 15, 13)) == "june10_1513_nicholas"


def test_run_name_sanitizes_custom_job_name():
    assert _run_name("My LogLbol Run", datetime(2026, 6, 10, 15, 13)) == "june10_1513_my_loglbol_run"


def test_safe_run_label_rejects_empty_labels():
    with pytest.raises(RuntimeError, match="--job-name"):
        _safe_run_label(" - ")


def test_chunks_split_manifest_rows_at_four_thousand():
    tasks = [{"fit_index": str(i), "object_id": f"obj-{i}", "COSMOS_ID0": str(i)} for i in range(13558)]

    chunks = _chunks(tasks, 4000)

    assert [(start, end, len(chunk)) for start, end, chunk in chunks] == [
        (0, 3999, 4000),
        (4000, 7999, 4000),
        (8000, 11999, 4000),
        (12000, 13557, 1558),
    ]


def test_filter_missing_tasks_skips_existing_results_and_optionally_failures(tmp_path):
    tasks = [
        {"fit_index": "0", "object_id": "obj-a", "COSMOS_ID0": "10"},
        {"fit_index": "1", "object_id": "obj-b", "COSMOS_ID0": "11"},
        {"fit_index": "2", "object_id": "obj-c", "COSMOS_ID0": "12"},
    ]
    result = tmp_path / "results" / "00000_COSMOS10_obj-a.json"
    failure = tmp_path / "failures" / "00001_COSMOS11_obj-b.json"
    result.parent.mkdir()
    failure.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    failure.write_text("{}", encoding="utf-8")

    selected, skipped = _filter_missing_tasks(tasks, tmp_path, rerun_failures=False)
    selected_with_failures, skipped_with_failures = _filter_missing_tasks(tasks, tmp_path, rerun_failures=True)

    assert [task["fit_index"] for task in selected] == ["2"]
    assert skipped == 2
    assert [task["fit_index"] for task in selected_with_failures] == ["1", "2"]
    assert skipped_with_failures == 1


def test_resolve_first_existing_prefers_grahsp_install_layout(tmp_path):
    project_root = tmp_path / "My-AGN-research-repository"
    sampler_script = tmp_path / "GRAHSP" / "GRAHSP-run" / "dualsampler.py"
    cigale_root = tmp_path / "GRAHSP" / "GRAHSP"
    project_root.mkdir()
    sampler_script.parent.mkdir(parents=True)
    sampler_script.write_text("# test sampler\n", encoding="utf-8")
    cigale_root.mkdir(parents=True)

    assert _resolve_first_existing(
        project_root,
        [Path("../sampler/dualsampler.py"), Path("../GRAHSP/GRAHSP-run/dualsampler.py")],
        kind="file",
    ) == sampler_script.resolve()
    assert _resolve_first_existing(
        project_root,
        [Path("../cigale"), Path("../GRAHSP/GRAHSP")],
        kind="dir",
    ) == cigale_root.resolve()


def test_batch_script_creates_sed_pdf_directory():
    script = _batch_script(
        job_name="june10_1513_nicholas-00000-09999",
        array="0-9999",
        partition="day_amd",
        time_limit="02:00:00",
        cpus_per_task=1,
        mem_per_cpu="8g",
        project_root=Path("/project"),
        output_dir=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas"),
        manifest=Path("/project/fit_manifest.csv"),
        dsps_ssp_fn=Path("/project/tempdata.h5"),
        task_file=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas/slurm_tasks/object_ids.txt"),
        expected_count=13558,
        conda_env="nicholas",
        backend="jaxsedfit",
        grahsp_runner=Path("/project/hpc/run_grahsp_manifest_fit.py"),
        grahsp_sampler_script=Path("/project/../sampler/dualsampler.py"),
        grahsp_cigale_root=Path("/project/../cigale"),
        grahsp_mass_max=13.0,
    )

    assert '"${OUTPUT_DIR}/sed_pdfs"' in script
    assert '"${OUTPUT_DIR}/corner_pdfs"' in script
    assert '"${OUTPUT_DIR}/trace_pdfs"' in script


def test_batch_script_uses_sampler_and_conditional_nested_sampler_args():
    script = _batch_script(
        job_name="june10_1513_nicholas-00000-09999",
        array="0-9999",
        partition="day_amd",
        time_limit="02:00:00",
        cpus_per_task=1,
        mem_per_cpu="8g",
        project_root=Path("/project"),
        output_dir=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas"),
        manifest=Path("/project/fit_manifest.csv"),
        dsps_ssp_fn=Path("/project/tempdata.h5"),
        task_file=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas/slurm_tasks/object_ids.txt"),
        expected_count=13558,
        conda_env="nicholas",
        backend="grahspj",
        grahsp_runner=Path("/project/hpc/run_grahsp_manifest_fit.py"),
        grahsp_sampler_script=Path("/project/../sampler/dualsampler.py"),
        grahsp_cigale_root=Path("/project/../cigale"),
        grahsp_mass_max=13.0,
    )

    assert "BACKEND=grahspj" in script
    assert "--backend grahspj" in script
    assert 'SAMPLER="${SAMPLER:-optax+nuts}"' in script
    assert '--sampler "${SAMPLER}"' in script
    assert "--fit-method" not in script
    assert 'if [ -n "${NS_LIVE_POINTS:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-live-points "${NS_LIVE_POINTS}")' in script
    assert 'if [ -n "${NS_MAX_SAMPLES:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-max-samples "${NS_MAX_SAMPLES}")' in script
    assert 'if [ -n "${NS_DLOGZ:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-dlogz "${NS_DLOGZ}")' in script
    assert "ns_flag_enabled()" in script
    assert 'if ns_flag_enabled "${NS_DIFFICULT_MODEL:-}"; then' in script
    assert "NS_ARGS+=(--ns-difficult-model)" in script
    assert 'if ns_flag_enabled "${NS_PARAMETER_ESTIMATION:-}"; then' in script
    assert "NS_ARGS+=(--ns-parameter-estimation)" in script
    assert 'if [ -n "${NS_NUM_PARALLEL_WORKERS:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-num-parallel-workers "${NS_NUM_PARALLEL_WORKERS}")' in script
    assert 'if [ -n "${NS_INIT_EFFICIENCY_THRESHOLD:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-init-efficiency-threshold "${NS_INIT_EFFICIENCY_THRESHOLD}")' in script
    assert 'if [ -n "${NS_MAX_LIKELIHOOD_EVALS:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-max-likelihood-evals "${NS_MAX_LIKELIHOOD_EVALS}")' in script
    assert 'if [ -n "${NS_EFFICIENCY_THRESHOLD:-}" ]; then' in script
    assert 'NS_ARGS+=(--ns-efficiency-threshold "${NS_EFFICIENCY_THRESHOLD}")' in script
    assert '"${NS_ARGS[@]}"' in script


def test_batch_script_routes_grahsp_to_grahsp_runner():
    script = _batch_script(
        job_name="june10_1513_nicholas-00000-09999",
        array="0-9999",
        partition="day_amd",
        time_limit="02:00:00",
        cpus_per_task=1,
        mem_per_cpu="8g",
        project_root=Path("/project"),
        output_dir=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas"),
        manifest=Path("/project/fit_manifest.csv"),
        dsps_ssp_fn=Path("/project/tempdata.h5"),
        task_file=Path("/project/hpc_outputs/loglbol_mass_retrieval/june10_1513_nicholas/slurm_tasks/tasks.txt"),
        expected_count=13558,
        conda_env="nicholas",
        backend="grahsp",
        grahsp_runner=Path("/project/hpc/run_grahsp_manifest_fit.py"),
        grahsp_sampler_script=Path("/project/../sampler/dualsampler.py"),
        grahsp_cigale_root=Path("/project/../cigale"),
        grahsp_mass_max=13.0,
    )

    assert "BACKEND=grahsp" in script
    assert 'case "${BACKEND}" in' in script
    assert 'python "${GRAHSP_RUNNER}"' in script
    assert '--fit-index "${FIT_INDEX}"' in script
    assert '--sampler-script "${GRAHSP_SAMPLER_SCRIPT}"' in script
    assert 'GRAHSP_MASS_MAX=13' in script
    assert '--mass-max "${GRAHSP_MASS_MAX}"' in script
    assert '--backend grahspj' in script
