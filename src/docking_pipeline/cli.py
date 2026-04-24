from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from docking_pipeline.controller import (
    generate_run_artifacts,
    submit_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docking-pipeline",
        description="Controller for the Uni-Dock2 -> Uni-Mol -> gnina docking workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate a pipeline YAML config.")
    validate.add_argument("config", type=Path)

    plan = subparsers.add_parser("plan", help="Print the planned pipeline stages.")
    plan.add_argument("config", type=Path)

    run = subparsers.add_parser("run", help="Generate Slurm scripts for a run (and optionally submit).")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated Slurm workflow immediately with dependencies.",
    )
    run.add_argument(
        "--submit-all",
        action="store_true",
        help="Submit and run the whole workflow sequentially (login-node helper script).",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Allow using an existing work_dir and overwrite generated sbatch scripts.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate artifacts but do not run sbatch (implies --no-submit).",
    )

    submit = subparsers.add_parser("submit", help="Submit an existing run directory.")
    submit.add_argument("run_dir", type=Path, help="Run directory containing run.yaml and slurm/*.sbatch")

    return parser


def validate_config(config_path: Path) -> int:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    print(f"Config found: {config_path}")
    print("Full schema validation is a TODO.")
    return 0


def print_plan(config_path: Path) -> int:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    stages = [
        "prepare manifest and SDF chunks",
        "unidock2 fast",
        "select top 20%",
        "unidock2 balance",
        "select top 15%",
        "ECFP6 KMeans clustering",
        "Uni-Mol Docking V2",
        "gnina rescoring",
        "final score and rank",
    ]
    print(f"Pipeline config: {config_path}")
    for idx, stage in enumerate(stages, start=1):
        print(f"{idx}. {stage}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate-config":
        return validate_config(args.config)
    if args.command == "plan":
        return print_plan(args.config)
    if args.command == "run":
        run_dir = generate_run_artifacts(args.config, force=args.force)
        print(f"Run dir: {run_dir}")
        if args.dry_run:
            return 0
        if args.submit and args.submit_all:
            parser.error("--submit and --submit-all are mutually exclusive")
        if args.submit:
            submit_run(run_dir)
        if args.submit_all:
            # Run the auto submit helper.
            os.environ.setdefault("RUN_DIR", str(run_dir))
            subprocess.run(["bash", str(run_dir / "slurm" / "submit_workflow_deps.sh")], check=True)
        return 0
    if args.command == "submit":
        submit_run(args.run_dir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
