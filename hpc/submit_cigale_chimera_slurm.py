#!/usr/bin/env python3
"""Prepare and submit a full-Chimera CIGALE run on Slurm."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG_ROOT = Path("/home/ns2385/Cigale_run")
DEFAULT_CIGALE_SOURCE_DIR = Path("/home/ns2385/cigale/cigale-v2025.1")
DEFAULT_CHIMERA_INPUT = Path("/home/ns2385/Chimera/chimeras-2023-10-11/chimeras-cigale.fits")
DEFAULT_OUTPUT_ROOT = Path("/home/ns2385/project_pi_pn38/ns2385/cigale_chimera_runs")
DEFAULT_CHUNK_SIZE = 4000


@dataclass(frozen=True)
class SlurmSettings:
    job_name: str
    partition: str
    time_limit: str
    cpus_per_task: int
    mem: str
    conda_env: str
    pcigale_command: str


def _replace_or_prepend_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", flags=re.MULTILINE)
    replacement = rf"\g<1>{value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    return f"{key} = {value}\n{text}"


def patch_pcigale_ini(text: str, data_file: str, cores: int) -> str:
    """Return pcigale.ini text with the run-local input file and core count."""
    text = _replace_or_prepend_assignment(text, "data_file", data_file)
    text = _replace_or_prepend_assignment(text, "cores", str(cores))
    if not text.endswith("\n"):
        text += "\n"
    return text


def _write_input_link_or_copy(source: Path, destination: Path, copy_input: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_input:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def _iter_chunks(n_rows: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0:
        yield 0, n_rows
        return
    for start in range(0, n_rows, chunk_size):
        yield start, min(start + chunk_size, n_rows)


def _timestamped_run_name(model: str) -> str:
    stamp = datetime.now().strftime("%B%d_%H%M").lower()
    return f"{stamp}_chimera_{model.lower()}"


def _quote(path_or_text: object) -> str:
    return shlex.quote(str(path_or_text))


def build_slurm_script(
    run_dir: Path,
    cigale_source_dir: Path,
    settings: SlurmSettings,
) -> str:
    source_setup = ""
    if cigale_source_dir:
        source_setup = (
            f'CIGALE_SOURCE_DIR={_quote(cigale_source_dir)}\n'
            'if [ -d "$CIGALE_SOURCE_DIR" ]; then\n'
            '  export PYTHONPATH="$CIGALE_SOURCE_DIR:${PYTHONPATH:-}"\n'
            "fi\n"
        )

    return f"""#!/usr/bin/env bash
#SBATCH --job-name={settings.job_name}
#SBATCH --partition={settings.partition}
#SBATCH --time={settings.time_limit}
#SBATCH --cpus-per-task={settings.cpus_per_task}
#SBATCH --mem={settings.mem}
#SBATCH --output={run_dir}/slurm_%j.out
#SBATCH --error={run_dir}/slurm_%j.err

set -euo pipefail

module reset || true
source ~/.bashrc || true
conda activate {shlex.quote(settings.conda_env)}

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{settings.cpus_per_task}}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{settings.cpus_per_task}}}"
{source_setup}
cd {_quote(run_dir)}

echo "Run directory: $(pwd)"
echo "pcigale command: {settings.pcigale_command}"
which {_quote(settings.pcigale_command)}
{_quote(settings.pcigale_command)} check
{_quote(settings.pcigale_command)} run
"""


def _submit(script_path: Path, cwd: Path) -> str:
    result = subprocess.run(
        ["sbatch", str(script_path)],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}")
    return result.stdout.strip()


def _read_fits_table(path: Path):
    try:
        from astropy.table import Table
    except ImportError as exc:
        raise ImportError("Chunked CIGALE submission requires astropy to read/write FITS tables.") from exc
    return Table.read(path)


def _write_chunk_tables(source: Path, chunks_dir: Path, chunk_size: int, overwrite: bool) -> list[tuple[Path, int, int]]:
    table = _read_fits_table(source)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[tuple[Path, int, int]] = []
    for start, stop in _iter_chunks(len(table), chunk_size):
        chunk_path = chunks_dir / f"chimera_rows_{start:05d}_{stop - 1:05d}.fits"
        if chunk_path.exists() and not overwrite:
            raise FileExistsError(f"Chunk table already exists: {chunk_path}")
        table[start:stop].write(chunk_path, overwrite=True)
        chunk_paths.append((chunk_path, start, stop))
    return chunk_paths


def _write_run_files(
    run_dir: Path,
    source_ini: Path,
    input_fits: Path,
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    target_ini = run_dir / "pcigale.ini"
    ini_text = source_ini.read_text()
    target_ini.write_text(patch_pcigale_ini(ini_text, "input.fits", args.cpus_per_task))
    _write_input_link_or_copy(input_fits, run_dir / "input.fits", args.copy_input)
    (run_dir / "launcher_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target_ini


def _prepare_run(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    model_dir = args.config_root / args.model
    source_ini = args.config if args.config is not None else model_dir / "pcigale.ini"
    if not source_ini.is_file():
        raise FileNotFoundError(f"Missing pcigale.ini: {source_ini}")
    if not args.chimera_input.is_file():
        raise FileNotFoundError(f"Missing Chimera CIGALE input FITS: {args.chimera_input}")

    run_name = args.run_name or _timestamped_run_name(args.model)
    run_dir = args.output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = {
        "model": args.model,
        "source_ini": str(source_ini),
        "chimera_input": str(args.chimera_input),
        "input_mode": "copy" if args.copy_input else "symlink",
        "run_dir": str(run_dir),
        "cigale_source_dir": str(args.cigale_source_dir),
        "pcigale_command": args.pcigale_command,
        "chunk_size": args.chunk_size,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if args.chunk_size <= 0:
        _write_run_files(run_dir, source_ini, args.chimera_input, args, {**base_metadata, "chunked": False})
        return run_dir, source_ini, [run_dir]

    chunk_paths = _write_chunk_tables(args.chimera_input, run_dir / "input_chunks", args.chunk_size, args.overwrite)
    chunk_run_dirs: list[Path] = []
    chunk_manifest = []
    for chunk_path, start, stop in chunk_paths:
        chunk_run_dir = run_dir / "chunks" / f"rows_{start:05d}_{stop - 1:05d}"
        if chunk_run_dir.exists() and any(chunk_run_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Chunk run directory already exists and is not empty: {chunk_run_dir}")
        chunk_metadata = {
            **base_metadata,
            "chunked": True,
            "chunk_start": start,
            "chunk_stop_exclusive": stop,
            "chunk_input": str(chunk_path),
            "chunk_run_dir": str(chunk_run_dir),
        }
        _write_run_files(chunk_run_dir, source_ini, chunk_path, args, chunk_metadata)
        chunk_run_dirs.append(chunk_run_dir)
        chunk_manifest.append(
            {
                "chunk_index": len(chunk_manifest),
                "start": start,
                "stop_exclusive": stop,
                "n_rows": stop - start,
                "input_fits": str(chunk_path),
                "run_dir": str(chunk_run_dir),
            }
        )

    (run_dir / "launcher_metadata.json").write_text(json.dumps({**base_metadata, "chunked": True, "n_chunks": len(chunk_run_dirs)}, indent=2) + "\n")
    (run_dir / "chunk_manifest.json").write_text(json.dumps(chunk_manifest, indent=2) + "\n")
    return run_dir, source_ini, chunk_run_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Yang", help="Model folder under --config-root, e.g. Yang, Dale, Fritz, Ciesla, gal.")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Explicit pcigale.ini path. Overrides --config-root/--model.")
    parser.add_argument("--cigale-source-dir", type=Path, default=DEFAULT_CIGALE_SOURCE_DIR)
    parser.add_argument("--chimera-input", type=Path, default=DEFAULT_CHIMERA_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows per CIGALE job. Use 0 to submit one full-table job.")
    parser.add_argument("--copy-input", action="store_true", help="Copy the FITS file instead of symlinking it.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty existing run directory.")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--partition", default="day")
    parser.add_argument("--time", default="24:00:00", dest="time_limit")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--conda-env", default="jaxsedfit")
    parser.add_argument("--pcigale-command", default="pcigale")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the run directory and print the sbatch script without submitting.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare files but do not submit.")
    args = parser.parse_args(argv)

    run_dir, source_ini, run_dirs = _prepare_run(args)
    job_name = args.job_name or f"chimera_cigale_{args.model.lower()}"
    settings = SlurmSettings(
        job_name=job_name,
        partition=args.partition,
        time_limit=args.time_limit,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        conda_env=args.conda_env,
        pcigale_command=args.pcigale_command,
    )
    print(f"Model: {args.model}")
    print(f"Source config: {source_ini}")
    print(f"Input FITS: {args.chimera_input}")
    print(f"Run directory: {run_dir}")
    print(f"CIGALE jobs: {len(run_dirs)}")

    scripts: list[Path] = []
    for index, one_run_dir in enumerate(run_dirs):
        chunk_settings = settings
        if len(run_dirs) > 1:
            chunk_settings = SlurmSettings(
                job_name=f"{settings.job_name}_{index:03d}",
                partition=settings.partition,
                time_limit=settings.time_limit,
                cpus_per_task=settings.cpus_per_task,
                mem=settings.mem,
                conda_env=settings.conda_env,
                pcigale_command=settings.pcigale_command,
            )
        script = build_slurm_script(one_run_dir, args.cigale_source_dir, chunk_settings)
        script_path = one_run_dir / "run_cigale.slurm"
        script_path.write_text(script)
        os.chmod(script_path, 0o755)
        scripts.append(script_path)
    print(f"Slurm scripts: {len(scripts)}")

    if args.dry_run:
        print("\n--- first sbatch script ---")
        print(scripts[0].read_text().rstrip())
        return 0
    if args.prepare_only:
        print("Prepared run directory; not submitting because --prepare-only was set.")
        return 0

    submissions = []
    for script_path in scripts:
        output = _submit(script_path, script_path.parent)
        if output:
            submissions.append({"script": str(script_path), "sbatch_output": output})
    if submissions:
        (run_dir / "sbatch_submissions.json").write_text(json.dumps(submissions, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
