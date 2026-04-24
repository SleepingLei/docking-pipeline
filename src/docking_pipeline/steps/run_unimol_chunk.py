from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, write_csv


@dataclass(frozen=True)
class BatchRow:
    input_ligand: str
    input_docking_grid: str
    output_ligand_name: str


def _read_batch_csv(path: Path) -> list[BatchRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows: list[BatchRow] = []
        for row in r:
            rows.append(
                BatchRow(
                    input_ligand=str(row.get("input_ligand", "")).strip(),
                    input_docking_grid=str(row.get("input_docking_grid", "")).strip(),
                    output_ligand_name=str(row.get("output_ligand_name", "")).strip(),
                )
            )
        return rows


def _write_batch_csv(path: Path, rows: list[BatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_ligand", "input_docking_grid", "output_ligand_name"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "input_ligand": r.input_ligand,
                    "input_docking_grid": r.input_docking_grid,
                    "output_ligand_name": r.output_ligand_name,
                }
            )


def _unimol_expected_out(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.sdf"


def _demo_cmd(
    *,
    demo_py: Path,
    cfg,
    batch_csv: Path,
    out_dir: Path,
) -> list[str]:
    # Keep CLI args consistent with previous slurm invocation.
    cmd: list[str] = [
        sys.executable,
        str(demo_py),
        "--mode",
        "batch_one2many",
        "--batch-size",
        str(cfg.unimol.batch_size),
        "--nthreads",
        str(cfg.slurm.defaults.cpus_per_task),
        "--conf-size",
        str(cfg.unimol.conf_size),
    ]
    if cfg.unimol.cluster_conformers:
        cmd.append("--cluster")
    cmd.extend(
        [
            "--input-protein",
            str(cfg.inputs.receptor_pdb),
            "--input-batch-file",
            str(batch_csv),
            "--output-ligand-dir",
            str(out_dir),
        ]
    )
    if cfg.unimol.use_current_ligand_conf:
        cmd.append("--use_current_ligand_conf")
    if cfg.unimol.steric_clash_fix:
        cmd.append("--steric-clash-fix")
    cmd.extend(["--model-dir", str(cfg.unimol.model_path)])
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fault-tolerant Uni-Mol docking chunk runner. "
            "Runs the official demo.py on a chunk batch.csv, and if it crashes, bisects the chunk "
            "to isolate failing ligands and still produce outputs for the rest. "
            "Always writes unimol_summary.csv."
        )
    )
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument(
        "--max-subprocess-runs",
        type=int,
        default=200,
        help="Hard limit on how many demo.py invocations we will attempt when bisecting failures.",
    )
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir

    chunk_dir = run_dir / "unimol" / "chunks" / f"chunk_{args.chunk_id}"
    batch_csv = chunk_dir / "batch.csv"
    out_dir = chunk_dir / "out_sdf"
    ensure_dir(out_dir)
    tmp_dir = chunk_dir / "_runner_tmp"
    ensure_dir(tmp_dir)

    demo_py = Path(cfg.unimol.repo_dir) / "interface" / "demo.py"
    if not demo_py.exists():
        raise FileNotFoundError(demo_py)

    rows_all = _read_batch_csv(batch_csv)
    if not rows_all:
        raise RuntimeError(f"Empty batch.csv: {batch_csv}")

    failures: dict[str, str] = {}
    subprocess_runs = 0

    def run_subset(rows: list[BatchRow]) -> None:
        nonlocal subprocess_runs

        # Skip ligands already produced.
        rows = [r for r in rows if not _unimol_expected_out(out_dir, r.output_ligand_name).exists()]
        if not rows:
            return

        if subprocess_runs >= args.max_subprocess_runs:
            for r in rows:
                failures.setdefault(r.output_ligand_name, "too_many_subprocess_runs")
            return

        # Write temp batch file.
        subprocess_runs += 1
        tmp_batch = tmp_dir / f"batch_try_{subprocess_runs}.csv"
        _write_batch_csv(tmp_batch, rows)

        cmd = _demo_cmd(demo_py=demo_py, cfg=cfg, batch_csv=tmp_batch, out_dir=out_dir)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            return

        # Crash: bisect.
        err_tail = (proc.stderr or "").strip().splitlines()[-30:]
        err_msg = "\n".join(err_tail) if err_tail else f"demo.py returncode={proc.returncode}"

        if len(rows) == 1:
            failures[rows[0].output_ligand_name] = err_msg
            return

        mid = len(rows) // 2
        run_subset(rows[:mid])
        run_subset(rows[mid:])

    # First attempt: run whole chunk in one go for performance.
    run_subset(rows_all)

    # Summarize
    summary_rows: list[dict[str, object]] = []
    for r in rows_all:
        out_path = _unimol_expected_out(out_dir, r.output_ligand_name)
        if out_path.exists():
            status = "ok"
            err = ""
        else:
            err = failures.get(r.output_ligand_name, "")
            status = "failed" if err else "missing_output"
        summary_rows.append(
            {
                "output_ligand_name": r.output_ligand_name,
                "status": status,
                "output_sdf": str(out_path) if out_path.exists() else "",
                "error": err,
            }
        )

    write_csv(
        chunk_dir / "unimol_summary.csv",
        summary_rows,
        fieldnames=["output_ligand_name", "status", "output_sdf", "error"],
    )

    ok = sum(1 for r in summary_rows if r["status"] == "ok")
    failed = sum(1 for r in summary_rows if r["status"] != "ok")
    print(f"[unimol-runner] chunk={args.chunk_id} ok={ok} failed={failed} subprocess_runs={subprocess_runs}")

    # Never fail the whole chunk: downstream will simply see fewer outputs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

