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
python hpc/run.xsh --compare-backends --dry-run
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

If external GRAHSP already finished before raw SED/photometry CSV collection was added, you do not need to rerun the sampler as long as each work directory still has `grahsp_*/plots/sed_mJy.csv.gz`.
Backfill the CSVs and update the result JSONs with:

```bash
python hpc/backfill_grahsp_notebook_outputs.py \
  --output-dir hpc_outputs/loglbol_mass_retrieval/<run_name> \
  --manifest fit_manifest.csv
```

By default, external GRAHSP runs now retain the CSV products needed to rebuild SED plots and remove/copy no PDF plot products. This keeps home-directory usage much lower. If you specifically need the GRAHSP PDFs for one run, call `hpc/run_grahsp_manifest_fit.py` directly with `--keep-pdfs`.

## Single-Object JAXSEDFit vs External GRAHSP Comparison

Notebook 13 is also available as a non-interactive HPC script. The GRAHSP side uses the external GRAHSP runner, not the `grahspj`/JAXSEDFit backend alias:

```bash
python hpc/run.xsh \
  --compare-backends \
  --object-id 022754.38-073455.0_869049_0.0001 \
  --n-wave 1024 \
  --grahsp-mass-max 13 \
  --progress-bar
```

This writes one folder under:

```text
hpc_outputs/loglbol_mass_retrieval/grahsp_vs_jaxsedfit_single/
```

Each run folder contains `jaxsedfit_result.json`, `jaxsedfit_samples.h5`, `grahsp_result.json`, `comparison_summary.json`, the JAXSEDFit PDFs, the external GRAHSP work/artifact folders, and `grahsp_vs_jaxsedfit_sed.png` when both backends complete.
The comparison PNG titles include the recovered stellar mass above each SED panel.
If `jaxsedfit_samples.h5` already exists, a later GRAHSP-only run with `--skip-jaxsedfit` will still rebuild the joint PNG after the external GRAHSP side finishes.
If only `jaxsedfit_result.json` exists, the script falls back to a summary joint PNG: it uses the printed JAXSEDfit mass metadata and observed photometry on the left, but it cannot redraw the JAXSEDfit model SED curves without `jaxsedfit_samples.h5`.
The GRAHSP side keeps CSV products by default; pass `--keep-grahsp-pdfs` to `hpc/run_grahsp_jaxsedfit_comparison.py` only when you need the original GRAHSP PDF products too.

For a quick path/manifest check on the login node:

```bash
python hpc/run.xsh --compare-backends --dry-run
```

For a very short smoke test, run the comparison script directly with reduced sampler settings:

```bash
python hpc/run_grahsp_jaxsedfit_comparison.py \
  --object-id 022754.38-073455.0_869049_0.0001 \
  --n-wave 1024 \
  --nuts-warmup 100 \
  --nuts-samples 100 \
  --grahsp-live-points 200 \
  --grahsp-posterior-samples 500
```
