from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path

from docking_pipeline.pipeline_config import DockingPipelineConfig, load_config, normalize_config
from docking_pipeline.slurm_render import render_workflow_sbatch


def generate_run_artifacts(config_path: Path, *, force: bool = False) -> Path:
    cfg = load_config(config_path)
    cfg = normalize_config(cfg, config_path=config_path)

    run_dir = cfg.run.work_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    run_yaml = run_dir / "run.yaml"
    if run_yaml.exists() and not force:
        # Resume mode: do not overwrite run.yaml by default.
        pass
    else:
        run_yaml.write_text(cfg.to_yaml(), encoding="utf-8")

    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "slurm").mkdir(exist_ok=True)

    workflow = render_workflow_sbatch(cfg, run_yaml_path=run_yaml)
    for rel_path, content in workflow.items():
        out_path = run_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    return run_dir


def _run_cmd(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def submit_run(run_dir: Path) -> None:
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(run_yaml)
    cfg = DockingPipelineConfig.from_yaml(run_yaml.read_text(encoding="utf-8"))

    slurm_dir = run_dir / "slurm"
    submit_sh = slurm_dir / "submit_workflow.sh"
    if not submit_sh.exists():
        raise FileNotFoundError(submit_sh)

    env = os.environ.copy()
    env.setdefault("RUN_YAML", str(run_yaml))
    env.setdefault("PROJECT_DIR", str(cfg.run.project_dir))

    # Submit script prints job IDs; we stream output to user.
    subprocess.run(["bash", str(submit_sh)], cwd=str(run_dir), check=True, env=env)

