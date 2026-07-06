from __future__ import annotations

import csv
import gzip
import json

from hpc import backfill_grahsp_notebook_outputs


def test_backfill_updates_existing_grahsp_result(tmp_path):
    manifest = tmp_path / "fit_manifest.csv"
    fieldnames = [
        "fit_index",
        "object_id",
        "COSMOS_ID0",
        "redshift",
        "chimera_QSO_weight",
        "resample_weight",
        "log_stellar_mass_truth",
        "logLbol_QSO",
        "logLbol_chimera",
        "luminosity_bin",
        "u_sdss",
        "u_sdss_err",
        "r_sdss",
        "r_sdss_err",
        "i_sdss",
        "i_sdss_err",
        "z_sdss",
        "z_sdss_err",
        "J_2mass",
        "J_2mass_err",
        "H_2mass",
        "H_2mass_err",
        "Ks_2mass",
        "Ks_2mass_err",
        "spitzer.irac.I1",
        "spitzer.irac.I1_err",
        "spitzer.irac.I2",
        "spitzer.irac.I2_err",
    ]
    row = {
        "fit_index": "0",
        "object_id": "obj-a",
        "COSMOS_ID0": "10",
        "redshift": "0.0",
        "chimera_QSO_weight": "0.0001",
        "resample_weight": "1.0",
        "log_stellar_mass_truth": "10.0",
        "logLbol_QSO": "45.0",
        "logLbol_chimera": "41.0",
        "luminosity_bin": "L < 42",
    }
    for i, name in enumerate(
        ["u_sdss", "r_sdss", "i_sdss", "z_sdss", "J_2mass", "H_2mass", "Ks_2mass", "spitzer.irac.I1", "spitzer.irac.I2"]
    ):
        row[name] = str(float(i + 1))
        row[f"{name}_err"] = str(0.1 * float(i + 1))
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    output_dir = tmp_path / "out"
    stem = "00000_COSMOS10_obj-a"
    work_dir = output_dir / "work" / stem
    plot_dir = work_dir / "grahsp_obj-a_varV2" / "plots"
    plot_dir.mkdir(parents=True)
    sed_csv = "\n".join(
        [
            "wavelength,total,Stellar (attenuated),AGN disk",
            "0.255,10,4,2",
            "0.510,20,8,4",
            "1.020,30,16,8",
        ]
    ) + "\n"
    with gzip.open(plot_dir / "sed_mJy.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write(sed_csv)

    result_path = output_dir / "results" / f"{stem}.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "fit_method": "grahsp",
                "fit_index": 0,
                "COSMOS_ID0": "10",
                "object_id": "obj-a",
                "work_dir": str(work_dir),
            }
        ),
        encoding="utf-8",
    )

    updated, skipped = backfill_grahsp_notebook_outputs.backfill(output_dir, manifest)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert updated == 1
    assert skipped == 0
    assert (output_dir / "sed_csvs" / f"{stem}_mJy.csv.gz").is_file()
    assert (output_dir / "photometry_csvs" / f"{stem}_photometry.csv").is_file()
    assert payload["sed_mjy_csv_path"].endswith("_mJy.csv.gz")
    assert payload["photometry_csv_path"].endswith("_photometry.csv")
