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


DEFAULT_CONFIG_ROOT = Path("/home/ns2385/Cigale_run")
DEFAULT_CIGALE_SOURCE_DIR = Path("/home/ns2385/cigale/cigale-v2025.1")
DEFAULT_CHIMERA_INPUT = Path("/home/ns2385/Chimera/chimeras-2023-10-11/chimeras-cigale.fits")
DEFAULT_OUTPUT_ROOT = Path("/home/ns2385/project_pi_pn38/ns2385/cigale_chimera_runs")


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


def _prepare_run(args: argparse.Namespace) -> tuple[Path, Path]:
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

    target_ini = run_dir / "pcigale.ini"
    ini_text = source_ini.read_text()
    target_ini.write_text(patch_pcigale_ini(ini_text, "input.fits", args.cpus_per_task))
    _write_input_link_or_copy(args.chimera_input, run_dir / "input.fits", args.copy_input)

    metadata = {
        "model": args.model,
        "source_ini": str(source_ini),
        "chimera_input": str(args.chimera_input),
        "input_mode": "copy" if args.copy_input else "symlink",
        "run_dir": str(run_dir),
        "cigale_source_dir": str(args.cigale_source_dir),
        "pcigale_command": args.pcigale_command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "launcher_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir, source_ini


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Yang", help="Model folder under --config-root, e.g. Yang, Dale, Fritz, Ciesla, gal.")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Explicit pcigale.ini path. Overrides --config-root/--model.")
    parser.add_argument("--cigale-source-dir", type=Path, default=DEFAULT_CIGALE_SOURCE_DIR)
    parser.add_argument("--chimera-input", type=Path, default=DEFAULT_CHIMERA_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
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

    run_dir, source_ini = _prepare_run(args)
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
    script = build_slurm_script(run_dir, args.cigale_source_dir, settings)
    script_path = run_dir / "run_cigale.slurm"
    script_path.write_text(script)
    os.chmod(script_path, 0o755)

    print(f"Model: {args.model}")
    print(f"Source config: {source_ini}")
    print(f"Input FITS: {args.chimera_input}")
    print(f"Run directory: {run_dir}")
    print(f"Slurm script: {script_path}")

    if args.dry_run:
        print("\n--- sbatch script ---")
        print(script.rstrip())
        return 0
    if args.prepare_only:
        print("Prepared run directory; not submitting because --prepare-only was set.")
        return 0

    output = _submit(script_path, run_dir)
    if output:
        (run_dir / "sbatch_submission.txt").write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
