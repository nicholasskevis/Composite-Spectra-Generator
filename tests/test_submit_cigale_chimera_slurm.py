from pathlib import Path

from hpc.submit_cigale_chimera_slurm import SlurmSettings, _iter_chunks, build_slurm_script, patch_pcigale_ini


def test_patch_pcigale_ini_updates_data_file_and_cores():
    text = """
data_file = old_input.fits
cores = 2
analysis_method = pdf_analysis
""".lstrip()

    patched = patch_pcigale_ini(text, "input.fits", 8)

    assert "data_file = input.fits" in patched
    assert "cores = 8" in patched
    assert "old_input.fits" not in patched


def test_patch_pcigale_ini_adds_missing_keys():
    patched = patch_pcigale_ini("analysis_method = pdf_analysis\n", "input.fits", 4)

    assert patched.startswith("cores = 4\ndata_file = input.fits\n")


def test_build_slurm_script_runs_check_then_run():
    settings = SlurmSettings(
        job_name="chimera_cigale_yang",
        partition="day",
        time_limit="24:00:00",
        cpus_per_task=8,
        mem="32G",
        conda_env="jaxsedfit",
        pcigale_command="pcigale",
    )

    script = build_slurm_script(
        Path("/home/ns2385/project_pi_pn38/ns2385/cigale_chimera_runs/test"),
        Path("/home/ns2385/cigale/cigale-v2025.1"),
        settings,
    )

    assert "#SBATCH --partition=day" in script
    assert "conda activate jaxsedfit" in script
    assert "pcigale check" in script
    assert "pcigale run" in script
    assert "PYTHONPATH" in script


def test_iter_chunks_splits_like_slurm_manifests():
    assert list(_iter_chunks(13558, 4000)) == [
        (0, 4000),
        (4000, 8000),
        (8000, 12000),
        (12000, 13558),
    ]


def test_iter_chunks_allows_single_full_job():
    assert list(_iter_chunks(13558, 0)) == [(0, 13558)]
