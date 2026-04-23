from __future__ import annotations

import textwrap
from pathlib import Path

from docking_pipeline.pipeline_config import DockingPipelineConfig


def _sbatch_header(
    *,
    job_name: str,
    partition: str,
    time: str,
    cpus_per_task: int,
    mem: str,
    gres: str | None,
    account: str | None,
    output: str,
) -> str:
    lines: list[str] = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -t {time}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {output}",
    ]
    if account:
        lines.append(f"#SBATCH -A {account}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    # SBATCH directives must appear before any executable commands.
    lines.append("set -euo pipefail")
    return "\n".join(lines) + "\n"


def _python_env_exports(project_dir: Path) -> str:
    # Avoid requiring editable install on the remote side.
    return textwrap.dedent(
        f"""\
        export PROJECT_DIR="{project_dir}"
        export PYTHONPATH="{project_dir}/src:${{PYTHONPATH:-}}"
        """
    )


def render_workflow_sbatch(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> dict[str, str]:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    slurm_dir = run_dir / "slurm"

    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account

    # Stage chunk sizes
    fast_chunk = cfg.unidock2.stages["fast"].chunk_size
    bal_chunk = cfg.unidock2.stages["balance"].chunk_size
    det_chunk = cfg.unidock2.stages["detail"].chunk_size

    # File conventions inside run dir
    fast_chunks_dir = run_dir / "inputs" / "fast" / "chunks"
    bal_chunks_dir = run_dir / "inputs" / "balance" / "chunks"
    det_chunks_dir = run_dir / "inputs" / "detail" / "chunks"
    unimol_chunks_dir = run_dir / "unimol" / "chunks"

    # We do not know num_chunks until prepare/filter steps run; we set array later in submit_workflow.sh.
    scripts: dict[str, str] = {}

    scripts["slurm/00_prepare_inputs.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_prep",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "00_prepare_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.prepare_inputs \\
              --run-yaml "{run_yaml_path}"
            """
        )
    )

    scripts["slurm/10_unidock_fast_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_ud2_fast",
            partition=cfg.slurm.gpu_partition_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            account=acct,
            output=str(logs_dir / "10_unidock_fast_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "{run_dir}/unidock_fast/config.yaml" \\
              -o "{run_dir}/unidock_fast/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              -l "{fast_chunks_dir}/chunk${{SLURM_ARRAY_TASK_ID}}.sdf"

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "{run_dir}/unidock_fast/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              --out-csv "{run_dir}/unidock_fast/chunk_summaries/summary_${{SLURM_ARRAY_TASK_ID}}.csv"
            """
        )
    )

    scripts["slurm/11_select_fast_top.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_sel_fast",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "11_select_fast_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.select_top_fraction \\
              --run-yaml "{run_yaml_path}" \\
              --stage fast \\
              --top-fraction {cfg.selection.after_fast_top_fraction} \\
              --out-ligand-ids "{run_dir}/selection/fast_top/ligand_ids.txt"

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.filter_stage_inputs \\
              --run-yaml "{run_yaml_path}" \\
              --in-chunks-dir "{fast_chunks_dir}" \\
              --ligand-ids "{run_dir}/selection/fast_top/ligand_ids.txt" \\
              --out-chunks-dir "{bal_chunks_dir}" \\
              --chunk-size {bal_chunk}
            """
        )
    )

    scripts["slurm/20_unidock_balance_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_ud2_bal",
            partition=cfg.slurm.gpu_partition_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            account=acct,
            output=str(logs_dir / "20_unidock_balance_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "{run_dir}/unidock_balance/config.yaml" \\
              -o "{run_dir}/unidock_balance/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              -l "{bal_chunks_dir}/chunk${{SLURM_ARRAY_TASK_ID}}.sdf"

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "{run_dir}/unidock_balance/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              --out-csv "{run_dir}/unidock_balance/chunk_summaries/summary_${{SLURM_ARRAY_TASK_ID}}.csv"
            """
        )
    )

    scripts["slurm/21_select_balance_top.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_sel_bal",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "21_select_balance_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.select_top_fraction \\
              --run-yaml "{run_yaml_path}" \\
              --stage balance \\
              --top-fraction {cfg.selection.after_balance_top_fraction} \\
              --out-ligand-ids "{run_dir}/selection/balance_top/ligand_ids.txt"

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.filter_stage_inputs \\
              --run-yaml "{run_yaml_path}" \\
              --in-chunks-dir "{bal_chunks_dir}" \\
              --ligand-ids "{run_dir}/selection/balance_top/ligand_ids.txt" \\
              --out-chunks-dir "{det_chunks_dir}" \\
              --chunk-size {det_chunk}
            """
        )
    )

    scripts["slurm/30_unidock_detail_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_ud2_det",
            partition=cfg.slurm.gpu_partition_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            account=acct,
            output=str(logs_dir / "30_unidock_detail_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "{run_dir}/unidock_detail/config.yaml" \\
              -o "{run_dir}/unidock_detail/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              -l "{det_chunks_dir}/chunk${{SLURM_ARRAY_TASK_ID}}.sdf"

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "{run_dir}/unidock_detail/chunks/poses_${{SLURM_ARRAY_TASK_ID}}.sdf" \\
              --out-csv "{run_dir}/unidock_detail/chunk_summaries/summary_${{SLURM_ARRAY_TASK_ID}}.csv" \\
              --out-best-sdf "{run_dir}/unidock_detail/best_poses/best_${{SLURM_ARRAY_TASK_ID}}.sdf"
            """
        )
    )

    scripts["slurm/40_cluster_select.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_cluster",
            partition=cfg.slurm.bigmem_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "40_cluster_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.cluster_ecfp6_kmeans \\
              --run-yaml "{run_yaml_path}" \\
              --in-stage detail \\
              --clusters {cfg.selection.cluster_count} \\
              --per-cluster {cfg.selection.cluster_representatives_per_cluster} \\
              --out-ligand-ids "{run_dir}/selection/cluster_reps/ligand_ids.txt"
            """
        )
    )

    scripts["slurm/50_unimol_prepare.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_um_prep",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "50_unimol_prepare_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.prepare_unimol_inputs \\
              --run-yaml "{run_yaml_path}" \\
              --ligand-ids "{run_dir}/selection/cluster_reps/ligand_ids.txt"
            """
        )
    )

    scripts["slurm/60_unimol_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_unimol",
            partition=cfg.slurm.gpu_partition_unimol,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem="64G",
            gres="gpu:1",
            account=acct,
            output=str(logs_dir / "60_unimol_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.unimol.env_name} python "{cfg.unimol.repo_dir}/interface/demo.py" \\
              --mode batch_one2many \\
              --batch-size {cfg.unimol.batch_size} \\
              --nthreads {defaults.cpus_per_task} \\
              --conf-size {cfg.unimol.conf_size} \\
              {"--cluster" if cfg.unimol.cluster_conformers else ""} \\
              --input-protein "{cfg.inputs.receptor_pdb}" \\
              --input-batch-file "{unimol_chunks_dir}/chunk_${{SLURM_ARRAY_TASK_ID}}/batch.csv" \\
              --output-ligand-dir "{unimol_chunks_dir}/chunk_${{SLURM_ARRAY_TASK_ID}}/out_sdf" \\
              {"--use_current_ligand_conf" if cfg.unimol.use_current_ligand_conf else ""} \\
              {"--steric-clash-fix" if cfg.unimol.steric_clash_fix else ""} \\
              --model-dir "{cfg.unimol.model_path}"
            """
        )
    )

    scripts["slurm/70_gnina_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_gnina",
            partition=cfg.slurm.gpu_partition_gnina,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            account=acct,
            output=str(logs_dir / "70_gnina_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.run_gnina_chunk \\
              --run-yaml "{run_yaml_path}" \\
              --chunk-id "${{SLURM_ARRAY_TASK_ID}}"
            """
        )
    )

    scripts["slurm/80_finalize.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_final",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            account=acct,
            output=str(logs_dir / "80_finalize_%j.out"),
        )
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.finalize_rank \\
              --run-yaml "{run_yaml_path}"
            """
        )
    )

    scripts["slurm/submit_workflow.sh"] = _render_submit_script(cfg, run_yaml_path=run_yaml_path)

    return scripts


def _render_submit_script(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        RUN_YAML="${{RUN_YAML:-{run_yaml_path}}}"
        RUN_DIR="{run_dir}"
        STATE_DIR="$RUN_DIR/slurm/state"
        mkdir -p "$STATE_DIR"
        PREP_JOBID_FILE="$STATE_DIR/00_prepare_jobid.txt"

        # Idempotent: if prepare already ran and created chunk files, do not resubmit.
        if [[ "${{SKIP_PREPARE:-0}}" == "1" ]]; then
          echo "[submit] 00_prepare_inputs (skipped; SKIP_PREPARE=1)"
        elif compgen -G "$RUN_DIR/inputs/fast/chunks/chunk*.sdf" > /dev/null; then
          echo "[submit] 00_prepare_inputs (skipped; chunk files already exist)"
        elif [[ -f "$PREP_JOBID_FILE" ]]; then
          echo "[submit] 00_prepare_inputs (skipped; already submitted jobid=$(cat "$PREP_JOBID_FILE"))"
        else
          echo "[submit] 00_prepare_inputs"
          j0=$(sbatch "$RUN_DIR/slurm/00_prepare_inputs.sbatch" | awk '{{print $4}}')
          echo "$j0" > "$PREP_JOBID_FILE"
          echo "  jobid=$j0"
          echo
          echo "Wait until 00_prepare_inputs finishes, then submit the next phases."
        fi

        echo
        echo "Stepwise submit (copy/paste):"
        echo "  cd \"$RUN_DIR\""
        echo
        echo "# Uni-Dock2 fast"
        echo "  n=\\$(ls -1 inputs/fast/chunks/chunk*.sdf | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/10_unidock_fast_array.sbatch"
        echo "  sbatch slurm/11_select_fast_top.sbatch"
        echo
        echo "# Uni-Dock2 balance"
        echo "  n=\\$(ls -1 inputs/balance/chunks/chunk*.sdf | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/20_unidock_balance_array.sbatch"
        echo "  sbatch slurm/21_select_balance_top.sbatch"
        echo
        echo "# Uni-Dock2 detail"
        echo "  n=\\$(ls -1 inputs/detail/chunks/chunk*.sdf | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/30_unidock_detail_array.sbatch"
        echo
        echo "# Clustering"
        echo "  sbatch slurm/40_cluster_select.sbatch"
        echo
        echo "# Uni-Mol docking v2"
        echo "  sbatch slurm/50_unimol_prepare.sbatch"
        echo "  n=\\$(ls -d unimol/chunks/chunk_* | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/60_unimol_array.sbatch"
        echo
        echo "# gnina rescoring + finalize"
        echo "  sbatch --array=0-\\$((n-1)) slurm/70_gnina_array.sbatch"
        echo "  sbatch slurm/80_finalize.sbatch"
        echo
        echo "Final output: $RUN_DIR/final/final_scores.csv"
        exit 0
        """
    )
