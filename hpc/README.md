# HPC LogLbol Fitting

Run these commands from the repository root. The runtime scripts needed by the Slurm workflow live in this folder:

- `run.xsh` runs one configured single-object fit.
- `submit_loglbol_slurm_chunks.py` submits chunked Slurm arrays.
- `run_manifest_fit.py` runs one jaxsedfit/GRAHSPJ manifest row.
- `run_grahsp_manifest_fit.py` runs one external GRAHSP manifest row.

The submitter expects these repository-root inputs by default:

- `fit_manifest.csv`
- `tempdata.h5` for the jaxsedfit/GRAHSPJ backend
- sibling checkouts `../sampler/dualsampler.py` and `../cigale` for the external GRAHSP backend

Example dry runs:

```bash
python hpc/run.xsh --dry-run
python hpc/submit_loglbol_slurm_chunks.py --backend grahspj --dry-run
python hpc/submit_loglbol_slurm_chunks.py --backend grahsp --dry-run
```

Example submission:

```bash
python hpc/submit_loglbol_slurm_chunks.py \
  --manifest fit_manifest.csv \
  --backend grahspj \
  --job-name chimera_jaxsedfit
```

For the external GRAHSP backend, the Python environment must include the CIGALE/GRAHSP runtime dependencies, including `configobj`, `sqlalchemy`, `numba`, and `ultranest`.
