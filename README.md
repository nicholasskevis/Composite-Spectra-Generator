# Composite-Spectra-Generator
Creating composite spectra from galaxy + quasar spectra to complement the Chimera benchmark dataset. 

This notebook rebuilds a composite spectrum from scratch for one Chimera object (see Johannes Buchner's GRAHSP code). It reads `chimera_provenance.csv`, finds the matching COSMOS/zCOSMOS galaxy spectrum and DR7Q AGN spectrum, standardizes wavelength and flux-density units, shifts the spectra onto the Chimera redshift frame, applies COSMOS foreground extinction, and combines them as

`F_lambda,composite = F_lambda,galaxy + chimera_qso_weight * F_lambda,qso`.

Important logic note: spectra are naturally stored as flux density (`erg cm^-2 s^-1 Angstrom^-1`). We should sum the components in flux-density units after putting them on the same observed-frame wavelength grid. A per-bin integrated flux can then be computed as `F_lambda * delta_lambda`; converting to integrated flux before interpolation can create binning-dependent artifacts.

To run the code download the SDSS dr7q spectra and the zCOSMOS spectra. The `chimera_provenance.csv` is provided. 
