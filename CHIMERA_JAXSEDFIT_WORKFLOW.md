# Chimera JAXSEDfit Workflow

This repository now has the files needed to fit Chimera sources with JAXSEDfit/GRAHSPJ using the `hpc/run.xsh` framework.

## 1. Build the manifest

```bash
python build_chimera_fit_manifest.py
```

This writes:

- `fit_manifest.csv`
- `fit_manifest.summary.json`

By default it reads Chimera data from `../grahspj/data/chimeras-2023-10-11` and keeps the same 13,558 log-luminosity-selected rows used by the previous HPC workflow.

## 2. Run one source through `hpc/run.xsh`

```bash
python hpc/run.xsh --sampler optax+nuts
```

`hpc/run.xsh` intentionally keeps the run settings as constants at the top of the file. It uses:

- `MANIFEST = Path("fit_manifest.csv")`
- `DSPS_SSP_FN = Path("tempdata.h5")`
- `OBJECT_ID = "013549.53+241149.7_243632_0.0001"`
- `EXPECTED_COUNT = 13558`
- `OUTPUT_ROOT = Path("hpc_outputs/loglbol_mass_retrieval")`
- `OUTPUT_LABEL = "manual_single_013549"`

Use `--dry-run` to print the underlying `python hpc/run_manifest_fit.py ...` command without fitting. To run a different single source through this fixed launcher, update `OBJECT_ID` and `OUTPUT_LABEL` in `hpc/run.xsh`.

## 3. Submit all sources as Slurm chunks

```bash
python hpc/run.xsh --all-objects --backend grahspj --job-name chimera_jaxsedfit
```

Use `--dry-run` first to write the chunk task files and print the planned Slurm arrays without submitting. This creates a timestamped run directory under `hpc_outputs/loglbol_mass_retrieval/`.

## 4. Merge results and plot properties

After jobs finish, point the plotter at the run directory:

```bash
python plot_chimera_fit_results.py \
  --output-dir hpc_outputs/loglbol_mass_retrieval/<run-directory>
```

This writes:

- `chimera_jaxsedfit_properties.csv`
- `chimera_jaxsedfit_failures.csv`
- `chimera_jaxsedfit_properties.png`

The plot shows Chimera AGN luminosity, recovered stellar mass, and recovered star formation for each successful fit. SFR is populated when the backend exposes an SFR-like posterior sample. Each successful fit JSON also records the SED, corner, and trace PDF paths.
