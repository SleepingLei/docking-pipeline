from __future__ import annotations

import argparse
from pathlib import Path


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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
