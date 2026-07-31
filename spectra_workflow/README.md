# Chimera Spectra Workflow

This folder keeps the scripts needed to build and check Chimera composite spectra for the joint photometry + spectroscopy work. It intentionally does not contain the raw spectra/data files. Keep those outside the repo, for example:

```bash
/home/nicho/GRAHSP_my/data
```

The workflow expects the project repo to contain:

- `fit_manifest.csv`
- the scripts in `spectra_workflow/scripts/`
- `spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv`

`chimera_provenance.csv` is rebuilt by the source-match audit from the active `grahspj_latest` Chimera FITS catalog and written under `spectra_workflow/outputs/source_match_audit/`.

The external data directory should contain the source spectra and reference files used by the builder, typically:

- `dr7q_spectra/`
- `zCOSMOS_data/`
- optional `cesam_vudz/` or `cesam_vuds/`
- `COSMOS2015_Laigle+_v1.1.fits`

The Chimera benchmark FITS catalog should come from the active `grahspj_latest` checkout, not from the external raw-data folder:

```bash
/home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11
```

Keeping this path fixed matters: the generated spectra must have Chimera IDs that also exist in the photometry table used by `grahspj_latest`.

All generated workflow products are written under `spectra_workflow/outputs/` by default. The raw source data stay outside this folder.

## Scripts

- `scripts/build_all_chimera_composite_spectra.py`
  Builds composite spectra from galaxy spectra + weighted DR7Q spectra.

- `scripts/audit_chimera_composite_spectra.py`
  Compares generated spectra against the Chimera benchmark photometry and writes per-band/object diagnostics.

- `scripts/audit_spectrum_source_matches.py`
  Rebuilds Chimera provenance from the Chimera FITS file, then audits the Chimera-to-zCOSMOS/DR7Q matching step and reports which objects have the source spectra needed for composite construction.

- `scripts/build_safe_joint_spectra_manifest.py`
  Applies conservative checks and writes a safe manifest for joint fitting.

- `scripts/build_renormalized_chimera_composite_spectra.py`
  Experimental builder that rescales the galaxy component before adding the weighted QSO spectrum.

- `scripts/chimera_composite_spectra.py`
  Single-object / notebook-18 command-line version. Useful for debugging one Chimera ID.

- `scripts/run_full_spectra_workflow.py`
  Runs the full workflow: source-match audit, composite-spectrum construction, photometry audit, and safe-manifest creation.

- `scripts/download_missing_dr7q_spectra.py`
  Downloads missing DR7Q spectra listed by the source-match audit, using plate-MJD-fiber keys.

- `scripts/search_sdss_replacement_spectra.py`
  Searches SDSS by QSO sky coordinate for replacement spectra and writes a QSO override table for the builder.

- `scripts/plot_safe_spectra_pdf.py`
  Selects 100 accepted safe spectra and writes a one-object-per-page PDF showing
  the galaxy, weighted QSO, and safe composite spectra.

## Recommended Order

Run from the repo root:

```bash
cd /home/nicho/GRAHSP_my/My-AGN-research-repository
```

### One-command workflow

To rebuild the source-match audit, all available spectra, the photometry audit, and the safe manifest in one run:

```bash
python spectra_workflow/scripts/run_full_spectra_workflow.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --data-dir /home/nicho/GRAHSP_my/data \
  --chimera-dir /home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11 \
  --fit-manifest fit_manifest.csv \
  --ignore-fit-manifest \
  --zcosmos-matches spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv \
  --coord-match-arcsec 1.0 \
  --qso-coord-match-arcsec 1.0 \
  --overwrite
```

This writes:

- `spectra_workflow/outputs/source_match_audit/`
- `spectra_workflow/outputs/all_chimera_spectra/`
- `spectra_workflow/outputs/chimera_composite_spectra_audit/`
- `spectra_workflow/outputs/safe_chimera_spectra/`
- `spectra_workflow/outputs/workflow_summary.json`

Use `--dry-run` first if you want to print the exact commands without executing them. Use `--limit 100` for a quick smoke test.

By default the builder shifts the QSO to the exact best galaxy spectroscopic redshift, keeps the native zCOSMOS observed-frame wavelength grid, degrades the higher-resolution spectrum to the lower assumed resolving power, and then uses flux-conserving pixel-overlap resampling before combining. The current defaults are `--galaxy-resolving-power 600`, `--qso-resolving-power 2000`, and `--resampling-method flux-conserving`, so the SDSS/DR7Q QSO spectrum is usually smoothed to the zCOSMOS-like resolution and rebinned by integrated pixel overlap. Use `--no-resolution-match` or `--resampling-method interp` only for diagnostics.

### Plot 100 safe spectra

```bash
python spectra_workflow/scripts/plot_safe_spectra_pdf.py
```

The default selection is reproducibly randomized with seed 42 and is written to
`spectra_workflow/outputs/safe_chimera_spectra/safe_spectra_components_100.pdf`.
Use `--seed`, `--number`, and `--output` to customize it, or `--first` to use
the first rows in the safe manifest.

The one-command workflow passes the source-match audit products into the builder. This matters because the audit can recover galaxy spectra by coordinate fallback; using `build_all_chimera_composite_spectra.py` directly without `--source-match-audit` only uses the ID/header indices and can miss many spectra that the audit found.

## Recovering More Source Spectra

After running the source-match audit, inspect:

```text
spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv
spectra_workflow/outputs/source_match_audit/missing_dr7q_spectra.csv
```

The audit distinguishes wrong-key cases from spectra that are simply not downloaded locally. In the current local data, most missing DR7Q cases have a valid DR7Q catalog match but no local FITS file.

### Direct DR7Q download by plate-MJD-fiber

Dry-run first:

```bash
python spectra_workflow/scripts/download_missing_dr7q_spectra.py \
  --missing-csv spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv \
  --data-dir /home/nicho/GRAHSP_my/data \
  --dry-run \
  --limit 20
```

Then download:

```bash
python spectra_workflow/scripts/download_missing_dr7q_spectra.py \
  --missing-csv spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv \
  --data-dir /home/nicho/GRAHSP_my/data
```

Downloaded spectra are written to:

```text
/home/nicho/GRAHSP_my/data/dr7q_spectra/
```

The script validates that each downloaded FITS file has the `flux` and `loglam`/`lambda` columns needed by the composite builder before keeping it.

### Search SDSS by QSO coordinate

This is useful when the exact DR7Q spectrum is unavailable locally but a newer SDSS/BOSS/eBOSS spectrum exists at the same sky position. It requires `astroquery` and network access.

```bash
python -m pip install astroquery
python spectra_workflow/scripts/search_sdss_replacement_spectra.py \
  --missing-objects-csv spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv \
  --data-dir /home/nicho/GRAHSP_my/data \
  --radius-arcsec 2.0 \
  --output-dir spectra_workflow/outputs/sdss_replacement_spectra
```

This writes:

- `sdss_replacement_candidates.csv`
- `qso_spectrum_overrides.csv`
- `sdss_replacement_failures.csv`

To also download the best replacement spectra:

```bash
python spectra_workflow/scripts/search_sdss_replacement_spectra.py \
  --missing-objects-csv spectra_workflow/outputs/source_match_audit/missing_dr7q_objects.csv \
  --data-dir /home/nicho/GRAHSP_my/data \
  --radius-arcsec 2.0 \
  --output-dir spectra_workflow/outputs/sdss_replacement_spectra \
  --download
```

Then rebuild spectra using the replacement table:

```bash
python spectra_workflow/scripts/run_full_spectra_workflow.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --data-dir /home/nicho/GRAHSP_my/data \
  --chimera-dir /home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11 \
  --fit-manifest fit_manifest.csv \
  --ignore-fit-manifest \
  --zcosmos-matches spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv \
  --qso-spectrum-overrides spectra_workflow/outputs/sdss_replacement_spectra/qso_spectrum_overrides.csv \
  --overwrite
```

### Radius tests

The matching radii are deliberately conservative. To test whether more matches are available, rerun the source-match audit with slightly larger radii:

```bash
python spectra_workflow/scripts/audit_spectrum_source_matches.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --data-dir /home/nicho/GRAHSP_my/data \
  --fit-manifest fit_manifest.csv \
  --zcosmos-matches spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv \
  --coord-match-arcsec 1.5 \
  --qso-coord-match-arcsec 2.0 \
  --output-dir spectra_workflow/outputs/source_match_audit_radius_test
```

Treat new coordinate-only matches as candidates until they have been checked, because larger radii increase the chance of false associations.

### 1. Build all available composite spectra

Optional preflight check:

```bash
python spectra_workflow/scripts/audit_spectrum_source_matches.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --data-dir /home/nicho/GRAHSP_my/data \
  --chimera-dir /home/nicho/GRAHSP_my/grahspj_latest/data/chimeras-2023-10-11 \
  --fit-manifest fit_manifest.csv \
  --zcosmos-matches spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv \
  --coord-match-arcsec 1.0 \
  --qso-coord-match-arcsec 1.0 \
  --output-dir spectra_workflow/outputs/source_match_audit
```

The source-match audit first tries ID-based galaxy matching, then falls back to sky-coordinate matching from Chimera `ALPHA_J2000_GAL`/`DELTA_J2000_GAL` to zCOSMOS FITS `RA`/`DEC` when no COSMOS-ID match is found.

For DR7Q spectra, the audit first tries the Chimera plate-MJD-fiber key. If that file is not present, it looks up the quasar in `dr7q_photometry/dr7qso.fit`, tries both catalog `SMJD` and `RMJD` keys, and finally tries a sky-coordinate match against the downloaded DR7Q FITS headers. The audit CSV records the requested key, the actually matched key, and whether the spectrum was recovered by an alternate key or is simply not downloaded locally.

This writes:

- `spectrum_source_match_audit.csv`
- `rebuilt_chimera_provenance.csv`
- `missing_galaxy_cosmos_ids.csv`
- `missing_dr7q_spectra.csv`
- `missing_dr7q_objects.csv`
- `match_diagnostics.csv`
- `summary.json`

```bash
python spectra_workflow/scripts/build_all_chimera_composite_spectra.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --data-dir /home/nicho/GRAHSP_my/data \
  --provenance spectra_workflow/outputs/source_match_audit/rebuilt_chimera_provenance.csv \
  --fit-manifest fit_manifest.csv \
  --ignore-fit-manifest \
  --zcosmos-matches spectra_workflow/config/chimera_zcosmos_alpha_delta_matches.csv \
  --source-match-audit spectra_workflow/outputs/source_match_audit/spectrum_source_match_audit.csv \
  --output-dir spectra_workflow/outputs/all_chimera_spectra \
  --overwrite
```

By default the builder also uses `fit_manifest.csv` if it exists, so it only creates spectra for the active fitting sample. Use `--ignore-fit-manifest` when you want every available spectrum from the rebuilt provenance table.

This writes:

- `spectra_workflow/outputs/all_chimera_spectra/chimera_spectra_manifest.csv`
- `spectra_workflow/outputs/all_chimera_spectra/chimera_spectra_failures.csv`
- ECSV spectra under `spectra_workflow/outputs/all_chimera_spectra/spectra/`

### 2. Audit the spectra against photometry

```bash
python spectra_workflow/scripts/audit_chimera_composite_spectra.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --spectra-manifest spectra_workflow/outputs/all_chimera_spectra/chimera_spectra_manifest.csv \
  --output-dir spectra_workflow/outputs/chimera_composite_spectra_audit
```

This writes:

- `composite_band_audit.csv`
- `object_summary.csv`
- `summary_stats.csv`
- low/high-QSO-weight subsets
- `failures.csv`

The audit uses local spectral flux near each filter effective wavelength. It is a fast sanity check, not full filter-curve synthetic photometry.

### 3. Build the safe joint-fit manifest

```bash
python spectra_workflow/scripts/build_safe_joint_spectra_manifest.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --fit-manifest fit_manifest.csv \
  --input-spectra-manifest spectra_workflow/outputs/all_chimera_spectra/chimera_spectra_manifest.csv \
  --output-dir spectra_workflow/outputs/safe_chimera_spectra \
  --overwrite
```

The safe script checks that each spectrum:

- has required columns: `wave_obs`, `flux_mjy`, `flux_err_mjy`, `mask`
- has finite wavelength/flux/error values
- has positive errors
- has at least 50 valid pixels by default
- has no more than 20 percent nonpositive original flux pixels by default
- overlaps with at least one of `r_sdss`, `i_sdss`, or `z_sdss`
- has median spectrum/photometry scale between 0.2 and 5.0 by default

It writes:

- `safe_chimera_spectra_manifest.csv`
- `safe_chimera_spectra_rejected.csv`
- stricter-mask ECSV spectra under `spectra/`

### 4. Optional: build renormalized spectra

```bash
python spectra_workflow/scripts/build_renormalized_chimera_composite_spectra.py \
  --project-root /home/nicho/GRAHSP_my/My-AGN-research-repository \
  --fit-manifest fit_manifest.csv \
  --input-spectra-manifest spectra_workflow/outputs/all_chimera_spectra/chimera_spectra_manifest.csv \
  --output-dir spectra_workflow/outputs/renormalized_chimera_spectra \
  --renorm-mode galaxy-to-total \
  --overwrite
```

This is the optional architecture test where the galaxy spectrum is rescaled against overlapping photometry before adding the weighted QSO spectrum.

## Running Joint Fits With the Safe Manifest

Use the safe manifest with the existing HPC launcher:

```bash
python hpc/run.xsh \
  --all-objects \
  --joint-spectra \
  --spectra-manifest spectra_workflow/outputs/safe_chimera_spectra/safe_chimera_spectra_manifest.csv \
  --output-dir /home/ns2385/project_pi_pn38/ns2385/joint_spectro_loglbol_mass_retrieval \
  --job-name chimera_joint_spectro \
  --partition day \
  --time 02:00:00 \
  --conda-env jaxsedfit \
  --n-wave 1024
```

## Process Notes 

This script rebuilds a composite spectrum from scratch for one Chimera object. It reads `chimera_provenance.csv`, finds the matching COSMOS/zCOSMOS galaxy spectrum and DR7Q AGN spectrum, standardizes wavelength and flux-density units, shifts the QSO spectrum onto the exact galaxy spectroscopic redshift frame, matches spectral resolution by smoothing the higher-resolution component down to the lower-resolution component, applies COSMOS foreground extinction, and combines them as:

```text
F_lambda,composite = F_lambda,galaxy + chimera_qso_weight * F_lambda,qso
```

Important logic note: spectra are naturally stored as flux density, `erg cm^-2 s^-1 Angstrom^-1`. The components should be summed in flux-density units after putting them on the same observed-frame wavelength grid. A per-bin integrated flux can then be computed as `F_lambda * delta_lambda`; converting to integrated flux before interpolation can create binning-dependent artifacts.

The notebook workflow is:

1. Configure the Chimera object and data paths. Leave `CHIMERA_ID = None` to use the matched default, or set it to a specific `chimera_id` from `chimera_provenance.csv`.

2. Open `chimera_provenance.csv`, select the object, and pull out the COSMOS galaxy ID, DR7Q spectrum key, Chimera redshift, DR7Q redshift, and QSO weight.

3. Read the COSMOS catalogue row for the selected galaxy. The `EBV` value is used as the COSMOS foreground color excess for the synthetic Chimera line of sight.

4. Build an index from COSMOS/zCOSMOS object IDs to FITS files, then find the DR7Q spectrum using the plate-MJD-fiber key. The zCOSMOS index uses both FITS headers and ESO readme mappings because the archive filenames do not always contain the COSMOS ID.

5. Convert any logarithmic SDSS wavelength grid into Angstrom, put SDSS flux density into `erg cm^-2 s^-1 Angstrom^-1`, and keep zCOSMOS flux density in its native physical units.

6. Shift the QSO spectrum through rest frame and back to the best available galaxy spectroscopic redshift. The galaxy spectrum keeps its native observed-frame wavelength grid. For flux density, the redshift transformation is:

```text
lambda_rest = lambda_obs / (1 + z_source)
F_lambda_rest = F_lambda_obs * (1 + z_source)
F_lambda_target_obs = F_lambda_rest / (1 + z_chimera)
```

7. Match the spectral resolutions before combining. The workflow assumes zCOSMOS/CESAM galaxy spectra have resolving power `R = 600` and SDSS/DR7Q QSO spectra have resolving power `R = 2000` unless told otherwise. It computes the lower target resolution and applies a wavelength-dependent Gaussian convolution only to the higher-resolution component:

```text
sigma_kernel(lambda)^2 = sigma_target(lambda)^2 - sigma_source(lambda)^2
```

If a component is already at the lower resolution, it is left alone. The workflow never deconvolves or sharpens a spectrum. The ECSV metadata records the assumed resolving powers and which component was convolved. If real per-object or per-pixel LSF curves become available, those should replace the current constant-`R` approximation.

8. Propagate source-spectrum errors through the same operations. zCOSMOS `ERR` columns are carried forward where present; SDSS inverse variance is converted to flux-density error. During resolution smoothing, errors are propagated in quadrature through the normalized Gaussian weights.

9. Apply wavelength-dependent Milky Way foreground attenuation using the COSMOS `EBV` value. This creates a synthetic observed Chimera spectrum along the COSMOS line of sight. It is not a dereddening step; if intrinsic spectra are needed, skip this attenuation.

10. Create the common observed-frame grid from the native zCOSMOS galaxy observed-frame pixels clipped to the QSO wavelength overlap. Rebin each component onto that grid using flux-conserving pixel-edge overlaps: each source pixel contributes its integrated flux density times overlap width to each target pixel, and the target integrated flux is divided by the target pixel width to recover flux density. This preserves integrated line fluxes and absorption equivalent widths better than point interpolation. `--resampling-method interp` restores the older interpolation behavior for diagnostics only.

11. Convert flux density into integrated flux per spectral bin for downstream work that needs actual bin fluxes. The composite itself is summed in flux-density units first, which keeps the interpolation physically consistent.

12. Write the matched component spectra and composite to an ECSV table with observed-frame wavelengths, flux densities, propagated errors, masks, metadata, and per-bin fluxes.

13. Plot the integrated flux in each observed-frame wavelength bin. These values are `F_lambda * delta_lambda`, with units of `erg cm^-2 s^-1`.

14. Plot the galaxy spectrum, weighted QSO spectrum, and summed composite on the common observed-frame grid.

15. Optionally divide the common wavelength grid by `(1 + z_chimera)` and transform the composite flux density to rest-frame units for line identification and diagnostics.
