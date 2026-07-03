from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc import run_grahsp_manifest_fit


def _load_rows_by_fit_index(manifest: Path) -> dict[int, dict[str, object]]:
    rows = {}
    for raw in run_grahsp_manifest_fit._load_manifest(manifest):
        row = run_grahsp_manifest_fit._row_from_manifest(raw)
        rows[int(row["fit_index"])] = row
    return rows


def _stem_from_payload(payload: dict[str, object]) -> str:
    return (
        f"{int(payload['fit_index']):05d}_"
        f"COSMOS{run_grahsp_manifest_fit._safe_id(str(payload['COSMOS_ID0']))}_"
        f"{run_grahsp_manifest_fit._safe_id(str(payload['object_id']))}"
    )


def backfill(output_dir: Path, manifest: Path) -> tuple[int, int]:
    output_dir = output_dir.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    rows_by_fit_index = _load_rows_by_fit_index(manifest)
    updated = 0
    skipped = 0

    for result_path in sorted((output_dir / "results").glob("*.json")):
        with open(result_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("fit_method") != "grahsp" or payload.get("status") != "success":
            skipped += 1
            continue

        fit_index = int(payload["fit_index"])
        row = rows_by_fit_index.get(fit_index)
        if row is None:
            skipped += 1
            continue

        stem = _stem_from_payload(payload)
        work_dir = Path(str(payload.get("work_dir") or output_dir / "work" / stem)).expanduser().resolve()
        artifacts = run_grahsp_manifest_fit._collect_grahsp_artifacts(work_dir, output_dir, stem, row)
        if not artifacts.get("notebook_sed_csv_path") and not artifacts.get("photometry_csv_path"):
            skipped += 1
            continue

        payload.update(artifacts)
        run_grahsp_manifest_fit._atomic_write_json(result_path, payload)
        updated += 1

    return updated, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create notebook-ready GRAHSP SED and photometry CSVs for already completed HPC results."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("fit_manifest.csv"))
    args = parser.parse_args(argv)

    updated, skipped = backfill(args.output_dir, args.manifest)
    print(f"[grahsp-backfill] updated={updated} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
