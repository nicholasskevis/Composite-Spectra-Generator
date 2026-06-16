from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from submit_loglbol_slurm_chunks import _batch_script, _run_name, _safe_run_label


def test_run_name_uses_month_day_time_and_label():
    assert _run_name("nicholas", datetime(2026, 6, 10, 15, 13)) == "june10_1513_nicholas"


def test_run_name_sanitizes_custom_job_name():
    assert _run_name("My LogLbol Run", datetime(2026, 6, 10, 15, 13)) == "june10_1513_my_loglbol_run"


def test_safe_run_label_rejects_empty_labels():
    with pytest.raises(RuntimeError, match="--job-name"):
        _safe_run_label(" - ")


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
    )

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
