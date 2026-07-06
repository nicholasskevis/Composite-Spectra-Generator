# HPC LogLbol Fitting

Run these commands from the repository root. The runtime scripts needed by the Slurm workflow live in this folder:

- `run.xsh` is the main entrypoint for one configured object or all-object Slurm submission.
- `submit_loglbol_slurm_chunks.py` submits chunked Slurm arrays for all manifest rows.
- `run_manifest_fit.py` runs one jaxsedfit/GRAHSPJ manifest row.
- `run_grahsp_manifest_fit.py` runs one external GRAHSP manifest row.

The submitter expects these repository-root inputs by default:

- `fit_manifest.csv`
- `tempdata.h5` for the jaxsedfit/GRAHSPJ backend
- `/home/<user>/GRAHSP/GRAHSP-run/dualsampler.py` and `/home/<user>/GRAHSP/GRAHSP` for the external GRAHSP backend

Example dry runs:

```bash
python hpc/run.xsh --dry-run
python hpc/run.xsh --all-objects --dry-run
python hpc/run.xsh --all-objects --backend grahsp --dry-run
```

Example submission:

```bash
python hpc/run.xsh --all-objects --backend grahspj --job-name chimera_jaxsedfit
```

The submitter defaults to chunks of 4000 Slurm array tasks, so the 13558-row manifest becomes four array jobs.
If a run only produced part of the manifest, submit the remaining rows into the same run directory:

```bash
python hpc/run.xsh \
  --all-objects \
  --backend grahsp \
  --run-dir hpc_outputs/loglbol_mass_retrieval/<run_name> \
  --only-missing
```

Add `--rerun-failures` if you also want to rerun rows that already have JSON files in `failures/`.

For the external GRAHSP backend, the Python environment must include the CIGALE/GRAHSP runtime dependencies, including `configobj`, `sqlalchemy`, `numba`, and `ultranest`.
If your GRAHSP checkout is somewhere else, pass `--grahsp-sampler-script` and `--grahsp-cigale-root` to `hpc/submit_loglbol_slurm_chunks.py`.

If external GRAHSP already finished before the notebook-ready SED exports were added, you do not need to rerun the sampler as long as each work directory still has `grahsp_*/plots/sed_mJy.csv.gz`.
Backfill the extra CSVs and update the result JSONs with:

```bash
python hpc/backfill_grahsp_notebook_outputs.py \
  --output-dir hpc_outputs/loglbol_mass_retrieval/<run_name> \
  --manifest fit_manifest.csv
```
