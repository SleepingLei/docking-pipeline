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
        "set -euo pipefail",
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

        echo "[submit] 00_prepare_inputs"
        j0=$(sbatch "$RUN_DIR/slurm/00_prepare_inputs.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j0"

        # fast
        n_fast=$(ls -1 "$RUN_DIR/inputs/fast/chunks"/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$n_fast" -lt 1 ]]; then
          echo "No fast chunks found under $RUN_DIR/inputs/fast/chunks" >&2
          exit 2
        fi
        last_fast=$((n_fast-1))
        echo "[submit] 10_unidock_fast_array afterok:$j0"
        j1=$(sbatch --dependency=afterok:$j0 --array=0-$last_fast "$RUN_DIR/slurm/10_unidock_fast_array.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j1"

        echo "[submit] 11_select_fast_top afterok:$j1"
        j2=$(sbatch --dependency=afterok:$j1 "$RUN_DIR/slurm/11_select_fast_top.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j2"

        # balance
        n_bal=$(ls -1 "$RUN_DIR/inputs/balance/chunks"/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$n_bal" -lt 1 ]]; then
          echo "No balance chunks found under $RUN_DIR/inputs/balance/chunks" >&2
          exit 2
        fi
        last_bal=$((n_bal-1))
        echo "[submit] 20_unidock_balance_array afterok:$j2"
        j3=$(sbatch --dependency=afterok:$j2 --array=0-$last_bal "$RUN_DIR/slurm/20_unidock_balance_array.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j3"

        echo "[submit] 21_select_balance_top afterok:$j3"
        j4=$(sbatch --dependency=afterok:$j3 "$RUN_DIR/slurm/21_select_balance_top.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j4"

        # detail
        n_det=$(ls -1 "$RUN_DIR/inputs/detail/chunks"/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$n_det" -lt 1 ]]; then
          echo "No detail chunks found under $RUN_DIR/inputs/detail/chunks" >&2
          exit 2
        fi
        last_det=$((n_det-1))
        echo "[submit] 30_unidock_detail_array afterok:$j4"
        j5=$(sbatch --dependency=afterok:$j4 --array=0-$last_det "$RUN_DIR/slurm/30_unidock_detail_array.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j5"

        # cluster
        echo "[submit] 40_cluster_select afterok:$j5"
        j6=$(sbatch --dependency=afterok:$j5 "$RUN_DIR/slurm/40_cluster_select.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j6"

        # unimol prep
        echo "[submit] 50_unimol_prepare afterok:$j6"
        j7=$(sbatch --dependency=afterok:$j6 "$RUN_DIR/slurm/50_unimol_prepare.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j7"

        # unimol array range is derived from generated chunk directories
        echo "[submit] 60_unimol_array afterok:$j7"
        n_um=$(ls -1 "$RUN_DIR/unimol/chunks"/chunk_* 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$n_um" -lt 1 ]]; then
          echo "No unimol chunks found under $RUN_DIR/unimol/chunks" >&2
          exit 2
        fi
        last_um=$((n_um-1))
        j8=$(sbatch --dependency=afterok:$j7 --array=0-$last_um "$RUN_DIR/slurm/60_unimol_array.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j8"

        echo "[submit] 70_gnina_array afterok:$j8"
        j9=$(sbatch --dependency=afterok:$j8 --array=0-$last_um "$RUN_DIR/slurm/70_gnina_array.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j9"

        echo "[submit] 80_finalize afterok:$j9"
        j10=$(sbatch --dependency=afterok:$j9 "$RUN_DIR/slurm/80_finalize.sbatch" | awk '{{print $4}}')
        echo "  jobid=$j10"

        echo "Workflow submitted. Final job: $j10"
        """
    )
