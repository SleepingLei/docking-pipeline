from __future__ import annotations

import textwrap
from pathlib import Path

from docking_pipeline.pipeline_config import DockingPipelineConfig


def _sbatch_header(
    *,
    job_name: str,
    partition: str,
    qos: str | None = None,
    time: str | None,
    cpus_per_task: int,
    mem: str,
    gres: str | None,
    exclusive: bool = False,
    account: str | None,
    output: str,
) -> str:
    err = output[:-4] + ".err" if output.endswith(".out") else output + ".err"
    lines: list[str] = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -o {output}",
        f"#SBATCH -e {err}",
    ]
    if time:
        lines.append(f"#SBATCH -t {time}")
    if account:
        lines.append(f"#SBATCH -A {account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    return "\n".join(lines) + "\n"


def _python_env_exports(project_dir: Path) -> str:
    # Avoid requiring editable install on the remote side.
    return textwrap.dedent(
        f"""\
        export PROJECT_DIR="{project_dir}"
        export PYTHONPATH="{project_dir}/src:${{PYTHONPATH:-}}"
        """
    )

def _bash_prologue() -> str:
    # Keep strict mode but avoid breaking SBATCH parsing by ensuring it comes after all directives.
    return "set -euo pipefail\n"

def _gpu_env_banner(*, task_id_var: str = "SLURM_ARRAY_TASK_ID") -> str:
    """
    Small, cheap runtime banner for debugging GPU visibility.

    We intentionally do not rely solely on Slurm's GPU binding, because some clusters may oversubscribe
    or misreport GPU counts. If the node only has 1 GPU, force CUDA_VISIBLE_DEVICES=0 for stability.
    """
    return textwrap.dedent(
        f"""\
        raw_task_id="${{{task_id_var}:-0}}"
        task_offset="${{TASK_OFFSET:-0}}"
        task_id=$((raw_task_id + task_offset))
        echo "[env] host=$(hostname) task_id=$task_id raw_task_id=$raw_task_id task_offset=$task_offset"
        echo "[env] SLURM_JOB_ID=${{SLURM_JOB_ID:-}} SLURM_ARRAY_TASK_ID=${{{task_id_var}:-}} TASK_OFFSET=${{TASK_OFFSET:-}}"
        echo "[env] SLURM_JOB_GPUS=${{SLURM_JOB_GPUS:-}} CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
        ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        echo "[env] nvidia-smi -L count=$ngpu"

        # Many Slurm setups set CUDA_VISIBLE_DEVICES automatically. If not, try to derive it from
        # Slurm-provided GPU ids so the program can safely use gpu_device_id=0 (first visible GPU).
        if [[ -z "${{CUDA_VISIBLE_DEVICES:-}}" ]]; then
          if [[ -n "${{SLURM_STEP_GPUS:-}}" ]]; then
            export CUDA_VISIBLE_DEVICES="${{SLURM_STEP_GPUS}}"
          elif [[ -n "${{SLURM_JOB_GPUS:-}}" ]]; then
            export CUDA_VISIBLE_DEVICES="${{SLURM_JOB_GPUS}}"
          elif [[ "$ngpu" == "1" ]]; then
            export CUDA_VISIBLE_DEVICES=0
          fi
        fi
        # Normalize possible formats like "gpu:3" -> "3"
        if [[ -n "${{CUDA_VISIBLE_DEVICES:-}}" ]]; then
          export CUDA_VISIBLE_DEVICES="$(echo "${{CUDA_VISIBLE_DEVICES}}" | sed 's/gpu://g; s/ //g')"
        fi
        echo "[env] effective CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
        """
    )

def _chunk_lock_snippet(*, lock_key: str, task_id_var: str = "SLURM_ARRAY_TASK_ID") -> str:
    """
    Avoid duplicate computation for the same chunk output when a job is accidentally resubmitted.

    We use a lock *directory* (mkdir is atomic) so we don't rely on `flock` availability.
    If the lock exists but its recorded PID is no longer alive, we treat it as stale and replace it.
    """
    return textwrap.dedent(
        f"""\
        lockdir="{lock_key}.lockdir"
        if mkdir "$lockdir" 2>/dev/null; then
          echo "$$" > "$lockdir/pid"
          echo "${{SLURM_JOB_ID:-}}" > "$lockdir/slurm_job_id"
          echo "${{{task_id_var}:-}}" > "$lockdir/array_task_id"
        else
          stale=1
          if [[ -f "$lockdir/pid" ]]; then
            oldpid="$(cat "$lockdir/pid" 2>/dev/null | tr -d ' ')"
            if [[ -n "$oldpid" ]] && kill -0 "$oldpid" 2>/dev/null; then
              echo "[resume] lock exists for {lock_key} (pid=$oldpid). Exiting."
              exit 0
            fi
          fi
          echo "[resume] stale lock exists for {lock_key}. Replacing."
          rm -rf "$lockdir" || true
          mkdir "$lockdir"
          echo "$$" > "$lockdir/pid"
          echo "${{SLURM_JOB_ID:-}}" > "$lockdir/slurm_job_id"
          echo "${{{task_id_var}:-}}" > "$lockdir/array_task_id"
        fi
        trap 'rm -rf "$lockdir" || true' EXIT
        """
    )


def _task_local_unidock_cfg_snippet(*, config_path: str, stage_tag: str) -> str:
    """
    Build a per-task UniDock config that points temp_dir_name at a unique local /tmp directory.

    This avoids cross-task contention in a shared temp directory and keeps heavy temporary IO off GPFS.
    """
    return textwrap.dedent(
        f"""\
        task_tmpdir="/tmp/docking_pipeline_{stage_tag}_${{SLURM_JOB_ID:-nojob}}_${{task_id}}"
        task_cfg="/tmp/docking_pipeline_{stage_tag}_${{SLURM_JOB_ID:-nojob}}_${{task_id}}.yaml"
        mkdir -p "$task_tmpdir"
        sed -E 's|^([[:space:]]*temp_dir_name:).*|\\1 "'"$task_tmpdir"'"|' "{config_path}" > "$task_cfg"
        echo "[env] task_tmpdir=$task_tmpdir"
        echo "[env] task_cfg=$task_cfg"
        """
    )


def _unidock_progress_watchdog_snippet(*, timeout_minutes: int, check_interval_minutes: int) -> str:
    timeout_secs = timeout_minutes * 60
    interval_secs = check_interval_minutes * 60
    return textwrap.dedent(
        f"""\
        monitor_unidock_progress() {{
          local pid="$1"
          local watch_dir="$2"
          local last_sig=""
          local last_change
          local timeout_secs={timeout_secs}
          local interval_secs={interval_secs}

          last_change="$(date +%s)"
          while kill -0 "$pid" 2>/dev/null; do
            local size files sig now idle
            if [[ -d "$watch_dir" ]]; then
              size="$(du -sb "$watch_dir" 2>/dev/null | awk '{{print $1}}')"
              files="$(find "$watch_dir" -type f 2>/dev/null | wc -l | tr -d ' ')"
            else
              size=0
              files=0
            fi
            sig="${{size}}:${{files}}"
            if [[ "$sig" != "$last_sig" ]]; then
              last_sig="$sig"
              last_change="$(date +%s)"
              echo "[watchdog] progress sig=$sig"
            fi

            now="$(date +%s)"
            idle=$((now - last_change))
            if (( idle >= timeout_secs )); then
              echo "[watchdog] no tmpdir progress for $idle sec in $watch_dir; terminating pid=$pid" >&2
              kill -TERM "$pid" 2>/dev/null || true
              sleep 30
              kill -KILL "$pid" 2>/dev/null || true
              break
            fi
            sleep "$interval_secs"
          done
        }}
        """
    )


def _array_submit_helpers_snippet(*, max_array_tasks_per_job: int) -> str:
    return textwrap.dedent(
        f"""\
        MAX_ARRAY_TASKS_PER_JOB={max_array_tasks_per_job}

        submit_array_batches() {{
          local script="$1"
          local dep_kind="${{2:-}}"
          local dep_ids="${{3:-}}"
          local n="$4"
          local max_par="$5"

          local start=0
          local ids=()
          while (( start < n )); do
            local end=$((start + MAX_ARRAY_TASKS_PER_JOB - 1))
            if (( end >= n )); then
              end=$((n - 1))
            fi
            local width=$((end - start + 1))
            local arr="0-$((width - 1))"
            if [[ "$max_par" -gt 0 ]]; then
              local par="$max_par"
              if (( par > width )); then
                par="$width"
              fi
              arr="${{arr}}%${{par}}"
            fi

            local jid
            if [[ -n "$dep_kind" && -n "$dep_ids" ]]; then
              jid=$(sbatch --parsable --dependency="${{dep_kind}}:${{dep_ids}}" --export=ALL,TASK_OFFSET="$start" --array="$arr" "$script")
            else
              jid=$(sbatch --parsable --export=ALL,TASK_OFFSET="$start" --array="$arr" "$script")
            fi
            ids+=("$jid")
            start=$((end + 1))
          done

          local joined=""
          local sep=""
          local jid=""
          for jid in "${{ids[@]}}"; do
            joined="${{joined}}${{sep}}${{jid}}"
            sep=":"
          done
          echo "$joined"
        }}
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "00_prepare_%j.out"),
        )
        + _bash_prologue()
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
            qos=cfg.slurm.gpu_qos_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            exclusive=cfg.slurm.gpu_exclusive,
            account=acct,
            output=str(logs_dir / "10_unidock_fast_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _gpu_env_banner()
        + textwrap.dedent(
            f"""\
            poses="{run_dir}/unidock_fast/chunks/poses_${{task_id}}.sdf"

            if [[ -s "$poses" ]]; then
              echo "[resume] fast poses exists: $poses"
            else
              tmp="${{poses}}.tmp.${{SLURM_JOB_ID:-}}.${{task_id}}"
              {_chunk_lock_snippet(lock_key='${poses}', task_id_var='task_id').rstrip()}
              {_task_local_unidock_cfg_snippet(config_path=str(run_dir / "unidock_fast" / "config.yaml"), stage_tag="ud2_fast").rstrip()}
              {_unidock_progress_watchdog_snippet(timeout_minutes=cfg.unidock2.no_progress_timeout_minutes, check_interval_minutes=cfg.unidock2.progress_check_interval_minutes).rstrip()}
              dock_rc=0
              conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "$task_cfg" \\
                -o "$tmp" \\
                -l "{fast_chunks_dir}/chunk${{task_id}}.sdf" &
              dock_pid=$!
              monitor_unidock_progress "$dock_pid" "$task_tmpdir" &
              watchdog_pid=$!
              wait "$dock_pid" || dock_rc=$?
              kill "$watchdog_pid" 2>/dev/null || true
              wait "$watchdog_pid" 2>/dev/null || true
              if [[ "$dock_rc" -ne 0 ]]; then
                echo "[error] unidock2 fast exited rc=$dock_rc" >&2
                exit "$dock_rc"
              fi
              mv -f "$tmp" "$poses"
              rm -f "$task_cfg"
              rm -rf "$task_tmpdir"
            fi
            """
        )
    )

    scripts["slurm/12_summarize_fast_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_sum_fast",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            exclusive=False,
            account=acct,
            output=str(logs_dir / "12_summarize_fast_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            raw_task_id="${{SLURM_ARRAY_TASK_ID:-0}}"
            task_offset="${{TASK_OFFSET:-0}}"
            task_id=$((raw_task_id + task_offset))
            poses="{run_dir}/unidock_fast/chunks/poses_${{task_id}}.sdf"
            summary="{run_dir}/unidock_fast/chunk_summaries/summary_${{task_id}}.csv"

            if [[ -s "$summary" ]]; then
              echo "[resume] fast summary exists: $summary"
              exit 0
            fi
            if [[ ! -s "$poses" ]]; then
              echo "[warn] fast poses missing, skip summary for task $task_id: $poses" >&2
              exit 0
            fi

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "$poses" \\
              --out-csv "$summary"
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "11_select_fast_%j.out"),
        )
        + _bash_prologue()
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
            qos=cfg.slurm.gpu_qos_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            exclusive=cfg.slurm.gpu_exclusive,
            account=acct,
            output=str(logs_dir / "20_unidock_balance_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _gpu_env_banner()
        + textwrap.dedent(
            f"""\
            poses="{run_dir}/unidock_balance/chunks/poses_${{task_id}}.sdf"

            if [[ -s "$poses" ]]; then
              echo "[resume] balance poses exists: $poses"
            else
              tmp="${{poses}}.tmp.${{SLURM_JOB_ID:-}}.${{task_id}}"
              {_chunk_lock_snippet(lock_key='${poses}', task_id_var='task_id').rstrip()}
              {_task_local_unidock_cfg_snippet(config_path=str(run_dir / "unidock_balance" / "config.yaml"), stage_tag="ud2_balance").rstrip()}
              {_unidock_progress_watchdog_snippet(timeout_minutes=cfg.unidock2.no_progress_timeout_minutes, check_interval_minutes=cfg.unidock2.progress_check_interval_minutes).rstrip()}
              dock_rc=0
              conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "$task_cfg" \\
                -o "$tmp" \\
                -l "{bal_chunks_dir}/chunk${{task_id}}.sdf" &
              dock_pid=$!
              monitor_unidock_progress "$dock_pid" "$task_tmpdir" &
              watchdog_pid=$!
              wait "$dock_pid" || dock_rc=$?
              kill "$watchdog_pid" 2>/dev/null || true
              wait "$watchdog_pid" 2>/dev/null || true
              if [[ "$dock_rc" -ne 0 ]]; then
                echo "[error] unidock2 balance exited rc=$dock_rc" >&2
                exit "$dock_rc"
              fi
              mv -f "$tmp" "$poses"
              rm -f "$task_cfg"
              rm -rf "$task_tmpdir"
            fi
            """
        )
    )

    scripts["slurm/22_summarize_balance_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_sum_bal",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            exclusive=False,
            account=acct,
            output=str(logs_dir / "22_summarize_balance_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            raw_task_id="${{SLURM_ARRAY_TASK_ID:-0}}"
            task_offset="${{TASK_OFFSET:-0}}"
            task_id=$((raw_task_id + task_offset))
            poses="{run_dir}/unidock_balance/chunks/poses_${{task_id}}.sdf"
            summary="{run_dir}/unidock_balance/chunk_summaries/summary_${{task_id}}.csv"

            if [[ -s "$summary" ]]; then
              echo "[resume] balance summary exists: $summary"
              exit 0
            fi
            if [[ ! -s "$poses" ]]; then
              echo "[warn] balance poses missing, skip summary for task $task_id: $poses" >&2
              exit 0
            fi

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "$poses" \\
              --out-csv "$summary"
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "21_select_balance_%j.out"),
        )
        + _bash_prologue()
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
            qos=cfg.slurm.gpu_qos_unidock2,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            exclusive=cfg.slurm.gpu_exclusive,
            account=acct,
            output=str(logs_dir / "30_unidock_detail_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _gpu_env_banner()
        + textwrap.dedent(
            f"""\
            poses="{run_dir}/unidock_detail/chunks/poses_${{task_id}}.sdf"

            if [[ -s "$poses" ]]; then
              echo "[resume] detail poses exists: $poses"
            else
              tmp="${{poses}}.tmp.${{SLURM_JOB_ID:-}}.${{task_id}}"
              {_chunk_lock_snippet(lock_key='${poses}', task_id_var='task_id').rstrip()}
              {_task_local_unidock_cfg_snippet(config_path=str(run_dir / "unidock_detail" / "config.yaml"), stage_tag="ud2_detail").rstrip()}
              {_unidock_progress_watchdog_snippet(timeout_minutes=cfg.unidock2.no_progress_timeout_minutes, check_interval_minutes=cfg.unidock2.progress_check_interval_minutes).rstrip()}
              dock_rc=0
              conda run -n {cfg.unidock2.env_name} unidock2 docking -cf "$task_cfg" \\
                -o "$tmp" \\
                -l "{det_chunks_dir}/chunk${{task_id}}.sdf" &
              dock_pid=$!
              monitor_unidock_progress "$dock_pid" "$task_tmpdir" &
              watchdog_pid=$!
              wait "$dock_pid" || dock_rc=$?
              kill "$watchdog_pid" 2>/dev/null || true
              wait "$watchdog_pid" 2>/dev/null || true
              if [[ "$dock_rc" -ne 0 ]]; then
                echo "[error] unidock2 detail exited rc=$dock_rc" >&2
                exit "$dock_rc"
              fi
              mv -f "$tmp" "$poses"
              rm -f "$task_cfg"
              rm -rf "$task_tmpdir"
            fi
            """
        )
    )

    scripts["slurm/31_summarize_detail_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_sum_det",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres=None,
            exclusive=False,
            account=acct,
            output=str(logs_dir / "31_summarize_detail_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            raw_task_id="${{SLURM_ARRAY_TASK_ID:-0}}"
            task_offset="${{TASK_OFFSET:-0}}"
            task_id=$((raw_task_id + task_offset))
            poses="{run_dir}/unidock_detail/chunks/poses_${{task_id}}.sdf"
            summary="{run_dir}/unidock_detail/chunk_summaries/summary_${{task_id}}.csv"
            best="{run_dir}/unidock_detail/best_poses/best_${{task_id}}.sdf"

            if [[ -s "$summary" && -s "$best" ]]; then
              echo "[resume] detail summary/best exists: $summary $best"
              exit 0
            fi
            if [[ ! -s "$poses" ]]; then
              echo "[warn] detail poses missing, skip summary for task $task_id: $poses" >&2
              exit 0
            fi

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.summarize_unidock2_chunk \\
              --in-sdf "$poses" \\
              --out-csv "$summary" \\
              --out-best-sdf "$best"
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "40_cluster_%j.out"),
        )
        + _bash_prologue()
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "50_unimol_prepare_%j.out"),
        )
        + _bash_prologue()
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
            qos=cfg.slurm.gpu_qos_unimol,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem="64G",
            gres="gpu:1",
            exclusive=cfg.slurm.gpu_exclusive,
            account=acct,
            output=str(logs_dir / "60_unimol_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _gpu_env_banner()
        + textwrap.dedent(
            f"""\
            # Fault-tolerant wrapper: prevents one bad ligand from crashing the whole chunk.
            conda run -n {cfg.unimol.env_name} python -m docking_pipeline.steps.run_unimol_chunk \\
              --run-yaml "{run_yaml_path}" \\
              --chunk-id "${{SLURM_ARRAY_TASK_ID:-0}}"
            """
        )
    )

    scripts["slurm/70_gnina_array.sbatch"] = (
        _sbatch_header(
            job_name=f"{cfg.run.name}_gnina",
            partition=cfg.slurm.gpu_partition_gnina,
            qos=cfg.slurm.gpu_qos_gnina,
            time=defaults.time,
            cpus_per_task=defaults.cpus_per_task,
            mem=defaults.mem,
            gres="gpu:1",
            exclusive=cfg.slurm.gpu_exclusive,
            account=acct,
            output=str(logs_dir / "70_gnina_%A_%a.out"),
        )
        + "#SBATCH --array=0-0\n"
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _gpu_env_banner()
        + textwrap.dedent(
            f"""\
            out_csv="{run_dir}/gnina/chunks/chunk_${{task_id}}/summary.csv"
            if [[ -s "$out_csv" ]]; then
              echo "[resume] gnina summary exists: $out_csv"
              exit 0
            fi

            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.run_gnina_chunk \\
              --run-yaml "{run_yaml_path}" \\
              --chunk-id "$task_id"
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
            exclusive=False,
            account=acct,
            output=str(logs_dir / "80_finalize_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            conda run -n {cfg.clustering.env_name} python -m docking_pipeline.steps.finalize_rank \\
              --run-yaml "{run_yaml_path}"
            """
        )
    )

    scripts["slurm/submit_workflow.sh"] = _render_submit_script(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/submit_workflow_auto.sh"] = _render_submit_auto_script(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/submit_workflow_deps.sh"] = _render_submit_deps_script(cfg, run_yaml_path=run_yaml_path)

    # Dependency-chain submitters (small CPU jobs that only submit the next stage once inputs exist).
    scripts["slurm/01_submit_fast.sbatch"] = _render_submitter_fast(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/02_submit_balance.sbatch"] = _render_submitter_balance(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/03_submit_detail.sbatch"] = _render_submitter_detail(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/04_submit_cluster_unimol.sbatch"] = _render_submitter_cluster_unimol(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/05_submit_unimol_array.sbatch"] = _render_submitter_unimol_array(cfg, run_yaml_path=run_yaml_path)
    scripts["slurm/06_submit_gnina_finalize.sbatch"] = _render_submitter_gnina_finalize(cfg, run_yaml_path=run_yaml_path)

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

        SBATCH_TIME="${{SBATCH_TIME:-}}"
        SBATCH_ARGS=()
        if [[ -n "$SBATCH_TIME" ]]; then
          SBATCH_ARGS+=("-t" "$SBATCH_TIME")
        fi

        # Idempotent: if prepare already ran and created chunk files, do not resubmit.
        if [[ "${{SKIP_PREPARE:-0}}" == "1" ]]; then
          echo "[submit] 00_prepare_inputs (skipped; SKIP_PREPARE=1)"
        elif compgen -G "$RUN_DIR/inputs/fast/chunks/chunk*.sdf" > /dev/null; then
          echo "[submit] 00_prepare_inputs (skipped; chunk files already exist)"
        elif [[ -f "$PREP_JOBID_FILE" ]]; then
          echo "[submit] 00_prepare_inputs (skipped; already submitted jobid=$(cat "$PREP_JOBID_FILE"))"
        else
          echo "[submit] 00_prepare_inputs"
          j0=$(sbatch "${{SBATCH_ARGS[@]}}" "$RUN_DIR/slurm/00_prepare_inputs.sbatch" | awk '{{print $4}}')
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
        echo "  sbatch --array=0-\\$((n-1)) slurm/12_summarize_fast_array.sbatch"
        echo "  sbatch slurm/11_select_fast_top.sbatch"
        echo
        echo "# Uni-Dock2 balance"
        echo "  n=\\$(ls -1 inputs/balance/chunks/chunk*.sdf | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/20_unidock_balance_array.sbatch"
        echo "  sbatch --array=0-\\$((n-1)) slurm/22_summarize_balance_array.sbatch"
        echo "  sbatch slurm/21_select_balance_top.sbatch"
        echo
        echo "# Uni-Dock2 detail"
        echo "  n=\\$(ls -1 inputs/detail/chunks/chunk*.sdf | wc -l)"
        echo "  sbatch --array=0-\\$((n-1)) slurm/30_unidock_detail_array.sbatch"
        echo "  sbatch --array=0-\\$((n-1)) slurm/31_summarize_detail_array.sbatch"
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


def _render_submit_auto_script(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    """
    A best-effort "run everything" submission helper.

    This runs on the login node and sequentially:
    1) submits a job (optionally with SBATCH_TIME),
    2) waits for completion,
    3) derives next array sizes from generated files,
    4) submits the next stage.

    It is intentionally conservative to avoid "empty array" mistakes.
    """
    run_dir = cfg.run.work_dir
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        RUN_YAML="${{RUN_YAML:-{run_yaml_path}}}"
        RUN_DIR="{run_dir}"

        SBATCH_TIME="${{SBATCH_TIME:-}}"
        SBATCH_ARGS=()
        if [[ -n "$SBATCH_TIME" ]]; then
          SBATCH_ARGS+=("-t" "$SBATCH_TIME")
        fi

        submit() {{
          # Usage: submit <sbatch_path> [extra sbatch args...]
          local script="$1"
          shift
          sbatch --parsable "${{SBATCH_ARGS[@]}}" "$@" "$script"
        }}

        wait_job() {{
          # Wait until a job disappears from squeue, then check sacct state.
          local jobid="$1"
          while true; do
            local q
            q="$(squeue -h -j "$jobid" 2>/dev/null || true)"
            if [[ -z "$q" ]]; then
              break
            fi
            sleep 10
          done

          local state
          # sacct can lag right after completion; poll a bit.
          state=""
          for _ in {{1..60}}; do
            state="$(sacct -j "$jobid" -n -o State 2>/dev/null | head -n1 | awk '{{print $1}}' || true)"
            if [[ -n "$state" ]]; then
              break
            fi
            sleep 5
          done
          if [[ -z "$state" ]]; then
            echo "[error] sacct returned empty state for job $jobid for too long" >&2
            return 1
          fi
          if [[ "$state" == "COMPLETED" ]]; then
            return 0
          fi
          echo "[error] job $jobid ended with state=$state" >&2
          sacct -j "$jobid" --format=JobID,JobName%20,Partition,State,ExitCode,Elapsed,NodeList | head -n 20 >&2 || true
          return 1
        }}

        wait_for_glob() {{
          # Usage: wait_for_glob <glob> <jobid>
          local pattern="$1"
          local jobid="$2"
          while true; do
            if compgen -G "$pattern" > /dev/null; then
              return 0
            fi
            # If the job already finished, fail fast (it should have produced the files).
            if ! squeue -h -j "$jobid" >/dev/null 2>&1; then
              echo "[error] squeue failed while waiting for $pattern" >&2
              return 1
            fi
            local q
            q="$(squeue -h -j "$jobid" 2>/dev/null || true)"
            if [[ -z "$q" ]]; then
              # job no longer in queue; confirm completion state (and surface errors)
              wait_job "$jobid"
              break
            fi
            sleep 10
          done
          if compgen -G "$pattern" > /dev/null; then
            return 0
          fi
          echo "[error] expected outputs not found: $pattern" >&2
          return 2
        }}

        cd "$RUN_DIR"

        echo "[auto] 00_prepare_inputs"
        if compgen -G "inputs/fast/chunks/chunk*.sdf" > /dev/null; then
          echo "  skipped (fast chunks already exist)"
        else
          j0="$(submit "slurm/00_prepare_inputs.sbatch")"
          echo "  jobid=$j0"
          wait_for_glob "inputs/fast/chunks/chunk*.sdf" "$j0"
        fi

        echo "[auto] 10_unidock_fast_array"
        n_fast="$(ls -1 inputs/fast/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_fast" -lt 1 ]]; then
          echo "[error] no fast chunks under inputs/fast/chunks" >&2
        exit 2
        fi
        n_fast_sum="$(ls -1 unidock_fast/chunk_summaries/summary_*.csv 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_fast_sum" -ge "$n_fast" ]]; then
          echo "  skipped (fast summaries already exist: $n_fast_sum/$n_fast)"
        else
            j1="$(submit "slurm/10_unidock_fast_array.sbatch" --array=0-$((n_fast-1)))"
            echo "  jobid=$j1"
          # Allow downstream summarization to run even if some array tasks fail.
          wait_for_glob "unidock_fast/chunks/poses_*.sdf" "$j1"
          j1s="$(submit "slurm/12_summarize_fast_array.sbatch" --dependency=afterany:$j1 --array=0-$((n_fast-1)))"
          echo "  summarize_jobid=$j1s"
          wait_job "$j1s"
        fi

        echo "[auto] 11_select_fast_top"
        if compgen -G "inputs/balance/chunks/chunk*.sdf" > /dev/null; then
          echo "  skipped (balance chunks already exist)"
        else
          j2="$(submit "slurm/11_select_fast_top.sbatch")"
          echo "  jobid=$j2"
          wait_job "$j2"
        fi

        echo "[auto] 20_unidock_balance_array"
        n_bal="$(ls -1 inputs/balance/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_bal" -lt 1 ]]; then
          echo "[error] no balance chunks under inputs/balance/chunks" >&2
          exit 2
        fi
        n_bal_sum="$(ls -1 unidock_balance/chunk_summaries/summary_*.csv 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_bal_sum" -ge "$n_bal" ]]; then
          echo "  skipped (balance summaries already exist: $n_bal_sum/$n_bal)"
        else
          j3="$(submit "slurm/20_unidock_balance_array.sbatch" --array=0-$((n_bal-1)))"
          echo "  jobid=$j3"
          wait_for_glob "unidock_balance/chunks/poses_*.sdf" "$j3"
          j3s="$(submit "slurm/22_summarize_balance_array.sbatch" --dependency=afterany:$j3 --array=0-$((n_bal-1)))"
          echo "  summarize_jobid=$j3s"
          wait_job "$j3s"
        fi

        echo "[auto] 21_select_balance_top"
        if compgen -G "inputs/detail/chunks/chunk*.sdf" > /dev/null; then
          echo "  skipped (detail chunks already exist)"
        else
          j4="$(submit "slurm/21_select_balance_top.sbatch")"
          echo "  jobid=$j4"
          wait_job "$j4"
        fi

        echo "[auto] 30_unidock_detail_array"
        n_det="$(ls -1 inputs/detail/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_det" -lt 1 ]]; then
          echo "[error] no detail chunks under inputs/detail/chunks" >&2
          exit 2
        fi
        n_det_sum="$(ls -1 unidock_detail/chunk_summaries/summary_*.csv 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_det_sum" -ge "$n_det" ]]; then
          echo "  skipped (detail summaries already exist: $n_det_sum/$n_det)"
        else
          j5="$(submit "slurm/30_unidock_detail_array.sbatch" --array=0-$((n_det-1)))"
          echo "  jobid=$j5"
          wait_for_glob "unidock_detail/chunks/poses_*.sdf" "$j5"
          j5s="$(submit "slurm/31_summarize_detail_array.sbatch" --dependency=afterany:$j5 --array=0-$((n_det-1)))"
          echo "  summarize_jobid=$j5s"
          wait_job "$j5s"
        fi

        echo "[auto] 40_cluster_select"
        if [[ -s "selection/cluster_reps/ligand_ids.txt" ]]; then
          echo "  skipped (cluster reps already exist)"
        else
          j6="$(submit "slurm/40_cluster_select.sbatch")"
          echo "  jobid=$j6"
          wait_job "$j6"
        fi

        echo "[auto] 50_unimol_prepare"
        if [[ -d "unimol/chunks" ]] && compgen -G "unimol/chunks/chunk_*/batch.csv" > /dev/null; then
          echo "  skipped (unimol batch files already exist)"
        else
          j7="$(submit "slurm/50_unimol_prepare.sbatch")"
          echo "  jobid=$j7"
          wait_job "$j7"
        fi

        echo "[auto] 60_unimol_array"
        n_um="$(ls -d unimol/chunks/chunk_* 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n_um" -lt 1 ]]; then
          echo "[error] no unimol chunks under unimol/chunks" >&2
          exit 2
        fi
        j8="$(submit "slurm/60_unimol_array.sbatch" --array=0-$((n_um-1)))"
        echo "  jobid=$j8"
        wait_job "$j8"

        echo "[auto] 70_gnina_array"
        j9="$(submit "slurm/70_gnina_array.sbatch" --array=0-$((n_um-1)))"
        echo "  jobid=$j9"
        wait_job "$j9"

        echo "[auto] 80_finalize"
        j10="$(submit "slurm/80_finalize.sbatch")"
        echo "  jobid=$j10"
        wait_job "$j10"

        echo
        echo "[auto] done"
        echo "Final output: $RUN_DIR/final/final_scores.csv"
        """
    )


def _render_submit_deps_script(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    """
    Submit the whole workflow as a Slurm dependency chain.

    This submits:
      00_prepare_inputs (unless fast chunks already exist)
      01_submit_fast (depends on prepare if needed)
    and the rest is submitted by small "submitter" jobs that run after dependencies complete.
    """
    run_dir = cfg.run.work_dir
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        RUN_YAML="${{RUN_YAML:-{run_yaml_path}}}"
        RUN_DIR="{run_dir}"
        STATE_DIR="$RUN_DIR/slurm/state"
        mkdir -p "$STATE_DIR"

        SBATCH_TIME="${{SBATCH_TIME:-}}"
        SBATCH_ARGS=()
        if [[ -n "$SBATCH_TIME" ]]; then
          SBATCH_ARGS+=("-t" "$SBATCH_TIME")
        fi

        cd "$RUN_DIR"

        j0=""
        if compgen -G "inputs/fast/chunks/chunk*.sdf" > /dev/null; then
          echo "[deps] 00_prepare_inputs skipped (fast chunks already exist)"
        else
          echo "[deps] 00_prepare_inputs"
          j0=$(sbatch --parsable "${{SBATCH_ARGS[@]}}" "slurm/00_prepare_inputs.sbatch")
          echo "$j0" > "$STATE_DIR/00_prepare_jobid.txt"
          echo "  jobid=$j0"
        fi

        echo "[deps] 01_submit_fast (dependency chain start)"
        if [[ -n "$j0" ]]; then
          j1=$(sbatch --parsable --dependency=afterok:"$j0" "slurm/01_submit_fast.sbatch")
        else
          j1=$(sbatch --parsable "slurm/01_submit_fast.sbatch")
        fi
        echo "$j1" > "$STATE_DIR/01_submit_fast_jobid.txt"
        echo "  jobid=$j1"

        echo
        echo "Submitted dependency chain under $RUN_DIR."
        echo "Tip: sacct -j $j1 --format=JobID,JobName%20,State,ExitCode,Elapsed"
        """
    )


def _render_submitter_fast(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_fast",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="1G",
            gres=None,
            account=acct,
            output=str(logs_dir / "01_submit_fast_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _array_submit_helpers_snippet(max_array_tasks_per_job=cfg.slurm.max_array_tasks_per_job)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            n=$(ls -1 inputs/fast/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
              echo "[error] no fast chunks under inputs/fast/chunks" >&2
              exit 2
            fi
            MAX_PAR={cfg.slurm.max_parallel_unidock2 if cfg.slurm.max_parallel_unidock2 is not None else -1}
            echo "[deps] submit unidock fast array n=$n"
            j_fast=$(submit_array_batches "slurm/10_unidock_fast_array.sbatch" "" "" "$n" "$MAX_PAR")
            echo "$j_fast" > "$STATE_DIR/10_fast_array_jobid.txt"
            echo "  fast_array_jobid=$j_fast"

            j_sum=$(submit_array_batches "slurm/12_summarize_fast_array.sbatch" "afterany" "$j_fast" "$n" "$MAX_PAR")
            echo "$j_sum" > "$STATE_DIR/12_fast_summary_jobid.txt"
            echo "  fast_summary_jobid=$j_sum"

            j_sel=$(sbatch --parsable --dependency=afterok:"$j_sum" slurm/11_select_fast_top.sbatch)
            echo "$j_sel" > "$STATE_DIR/11_select_fast_jobid.txt"
            echo "  select_fast_jobid=$j_sel"

            j_next=$(sbatch --parsable --dependency=afterok:"$j_sel" slurm/02_submit_balance.sbatch)
            echo "$j_next" > "$STATE_DIR/02_submit_balance_jobid.txt"
            echo "  next_submitter_jobid=$j_next"
            """
        )
    )


def _render_submitter_balance(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_bal",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="1G",
            gres=None,
            account=acct,
            output=str(logs_dir / "02_submit_balance_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _array_submit_helpers_snippet(max_array_tasks_per_job=cfg.slurm.max_array_tasks_per_job)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            n=$(ls -1 inputs/balance/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
              echo "[error] no balance chunks under inputs/balance/chunks" >&2
              exit 2
            fi
            MAX_PAR={cfg.slurm.max_parallel_unidock2 if cfg.slurm.max_parallel_unidock2 is not None else -1}
            echo "[deps] submit unidock balance array n=$n"
            j_bal=$(submit_array_batches "slurm/20_unidock_balance_array.sbatch" "" "" "$n" "$MAX_PAR")
            echo "$j_bal" > "$STATE_DIR/20_balance_array_jobid.txt"
            echo "  balance_array_jobid=$j_bal"

            j_sum=$(submit_array_batches "slurm/22_summarize_balance_array.sbatch" "afterany" "$j_bal" "$n" "$MAX_PAR")
            echo "$j_sum" > "$STATE_DIR/22_balance_summary_jobid.txt"
            echo "  balance_summary_jobid=$j_sum"

            j_sel=$(sbatch --parsable --dependency=afterok:"$j_sum" slurm/21_select_balance_top.sbatch)
            echo "$j_sel" > "$STATE_DIR/21_select_balance_jobid.txt"
            echo "  select_balance_jobid=$j_sel"

            j_next=$(sbatch --parsable --dependency=afterok:"$j_sel" slurm/03_submit_detail.sbatch)
            echo "$j_next" > "$STATE_DIR/03_submit_detail_jobid.txt"
            echo "  next_submitter_jobid=$j_next"
            """
        )
    )


def _render_submitter_detail(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_det",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="1G",
            gres=None,
            account=acct,
            output=str(logs_dir / "03_submit_detail_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _array_submit_helpers_snippet(max_array_tasks_per_job=cfg.slurm.max_array_tasks_per_job)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            n=$(ls -1 inputs/detail/chunks/chunk*.sdf 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
              echo "[error] no detail chunks under inputs/detail/chunks" >&2
              exit 2
            fi
            MAX_PAR={cfg.slurm.max_parallel_unidock2 if cfg.slurm.max_parallel_unidock2 is not None else -1}
            echo "[deps] submit unidock detail array n=$n"
            j_det=$(submit_array_batches "slurm/30_unidock_detail_array.sbatch" "" "" "$n" "$MAX_PAR")
            echo "$j_det" > "$STATE_DIR/30_detail_array_jobid.txt"
            echo "  detail_array_jobid=$j_det"

            j_sum=$(submit_array_batches "slurm/31_summarize_detail_array.sbatch" "afterany" "$j_det" "$n" "$MAX_PAR")
            echo "$j_sum" > "$STATE_DIR/31_detail_summary_jobid.txt"
            echo "  detail_summary_jobid=$j_sum"

            j_next=$(sbatch --parsable --dependency=afterok:"$j_sum" slurm/04_submit_cluster_unimol.sbatch)
            echo "$j_next" > "$STATE_DIR/04_submit_cluster_unimol_jobid.txt"
            echo "  next_submitter_jobid=$j_next"
            """
        )
    )


def _render_submitter_cluster_unimol(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_um",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="2G",
            gres=None,
            account=acct,
            output=str(logs_dir / "04_submit_cluster_unimol_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _array_submit_helpers_snippet(max_array_tasks_per_job=cfg.slurm.max_array_tasks_per_job)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            echo "[deps] submit clustering select"
            j_cl=$(sbatch --parsable slurm/40_cluster_select.sbatch)
            echo "$j_cl" > "$STATE_DIR/40_cluster_jobid.txt"
            echo "  cluster_jobid=$j_cl"

            echo "[deps] submit unimol prepare"
            j_prep=$(sbatch --parsable --dependency=afterok:"$j_cl" slurm/50_unimol_prepare.sbatch)
            echo "$j_prep" > "$STATE_DIR/50_unimol_prepare_jobid.txt"
            echo "  unimol_prepare_jobid=$j_prep"

            j_next=$(sbatch --parsable --dependency=afterok:"$j_prep" slurm/05_submit_unimol_array.sbatch)
            echo "$j_next" > "$STATE_DIR/05_submit_unimol_array_jobid.txt"
            echo "  next_submitter_jobid=$j_next"
            """
        )
    )


def _render_submitter_unimol_array(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_umarr",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="1G",
            gres=None,
            account=acct,
            output=str(logs_dir / "05_submit_unimol_array_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + _array_submit_helpers_snippet(max_array_tasks_per_job=cfg.slurm.max_array_tasks_per_job)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            n=$(ls -d unimol/chunks/chunk_* 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
              echo "[error] no unimol chunks under unimol/chunks" >&2
              exit 2
            fi
            MAX_PAR={cfg.slurm.max_parallel_unimol if cfg.slurm.max_parallel_unimol is not None else -1}
            echo "[deps] submit unimol array n=$n"
            j_um=$(submit_array_batches "slurm/60_unimol_array.sbatch" "" "" "$n" "$MAX_PAR")
            echo "$j_um" > "$STATE_DIR/60_unimol_array_jobid.txt"
            echo "  unimol_array_jobid=$j_um"

            j_next=$(sbatch --parsable --dependency=afterany:"$j_um" slurm/06_submit_gnina_finalize.sbatch)
            echo "$j_next" > "$STATE_DIR/06_submit_gnina_finalize_jobid.txt"
            echo "  next_submitter_jobid=$j_next"
            """
        )
    )


def _render_submitter_gnina_finalize(cfg: DockingPipelineConfig, *, run_yaml_path: Path) -> str:
    run_dir = cfg.run.work_dir
    logs_dir = run_dir / "logs"
    defaults = cfg.slurm.defaults
    acct = cfg.slurm.account
    return (
        _sbatch_header(
            job_name=f"{cfg.run.name}_submit_gn",
            partition=cfg.slurm.cpu_partition,
            time=defaults.time,
            cpus_per_task=1,
            mem="1G",
            gres=None,
            account=acct,
            output=str(logs_dir / "06_submit_gnina_finalize_%j.out"),
        )
        + _bash_prologue()
        + _python_env_exports(cfg.run.project_dir)
        + textwrap.dedent(
            f"""\
            RUN_DIR="{run_dir}"
            STATE_DIR="$RUN_DIR/slurm/state"
            mkdir -p "$STATE_DIR"
            cd "$RUN_DIR"

            n=$(ls -d unimol/chunks/chunk_* 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$n" -lt 1 ]]; then
              echo "[error] no unimol chunks under unimol/chunks" >&2
              exit 2
            fi
            MAX_PAR={cfg.slurm.max_parallel_gnina if cfg.slurm.max_parallel_gnina is not None else -1}

            echo "[deps] submit gnina array n=$n"
            j_gn=$(submit_array_batches "slurm/70_gnina_array.sbatch" "" "" "$n" "$MAX_PAR")
            echo "$j_gn" > "$STATE_DIR/70_gnina_array_jobid.txt"
            echo "  gnina_array_jobid=$j_gn"

            echo "[deps] submit finalize"
            j_fin=$(sbatch --parsable --dependency=afterany:"$j_gn" slurm/80_finalize.sbatch)
            echo "$j_fin" > "$STATE_DIR/80_finalize_jobid.txt"
            echo "  finalize_jobid=$j_fin"
            """
        )
    )
