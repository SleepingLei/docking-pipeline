# Docking Pipeline

Slurm-oriented screening pipeline for:

```text
Uni-Dock2 fast
  -> top fraction
  -> Uni-Dock2 balance
  -> top fraction
  -> ECFP6 KMeans clustering
  -> Uni-Mol Docking V2
  -> gnina rescoring
  -> final score table
```

Remote target path used in this project:

```bash
/data/project/hanwen/software/docking-pipeline
```

## What This Pipeline Does

Given one receptor PDB and a large ligand library in SDF format, the pipeline:

1. splits the input library into Uni-Dock2 chunks
2. runs `fast -> balance -> detail`
3. clusters surviving ligands by ECFP6 fingerprints
4. runs Uni-Mol on cluster representatives
5. rescored them with gnina
6. writes a merged final score table

The implementation is designed for large Slurm screens and includes:

- per-task local `/tmp` for Uni-Dock2
- watchdog termination for stuck Uni-Dock2 tasks
- array auto-splitting for clusters with small `MaxArraySize`
- optional GPU QOS support
- resume/skip behavior for completed chunk outputs
- tolerance for failed UniDock / UniMol / gnina sub-jobs so the whole workflow can continue

## Upstream References

This pipeline wraps the following upstream tools:

- Uni-Dock2: [dptech-corp/Uni-Dock2](https://github.com/dptech-corp/Uni-Dock2)
- Uni-Mol Docking V2: [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol)
- gnina: [gnina/gnina](https://github.com/gnina/gnina)

## First Remote Step

After syncing this repository to the remote machine:

```bash
cd /data/project/hanwen/software/docking-pipeline
bash scripts/remote_env_check.sh
```

To also submit tiny Slurm smoke jobs:

```bash
bash scripts/remote_env_check.sh --submit-tests
```

## Install

Install the controller into a lightweight environment:

```bash
conda run -n dock-pipe python -m pip install -e .
```

## Main Workflow

### 1. Prepare a config

Use one of the YAML configs in `configs/`, for example:

```bash
configs/rnaseL.yaml
```

### 2. Render a run directory

```bash
conda run -n dock-pipe docking-pipeline run configs/rnaseL.yaml --dry-run
```

This creates:

```text
runs/<run_name>/
  run.yaml
  inputs/
  slurm/
  logs/
  ...
```

If you change the config and want to regenerate scripts in the same run directory:

```bash
conda run -n dock-pipe docking-pipeline run configs/rnaseL.yaml --force --dry-run
```

### 3. Submit the dependency chain

```bash
cd runs/<run_name>
bash slurm/submit_workflow_deps.sh
```

This is the normal way to launch the workflow now.  
You usually do **not** need to submit each `.sbatch` manually.

## Recommended Config Knobs

The most important settings live in these sections:

### Uni-Dock2

```yaml
unidock2:
  temp_dir: /tmp
  no_progress_timeout_minutes: 15
  progress_check_interval_minutes: 5
  stages:
    fast:    {search_mode: fast,    chunk_size: 5000}
    balance: {search_mode: balance, chunk_size: 30000}
    detail:  {search_mode: detail,  chunk_size: 8000}
```

Notes:

- `temp_dir: /tmp` is strongly recommended. Generated Slurm scripts rewrite each task to a unique local temp directory under `/tmp`.
- `no_progress_timeout_minutes` is a Uni-Dock2 watchdog. If a task's local temp directory stops changing for this long, the task is terminated.
- Smaller `fast.chunk_size` reduces the blast radius of pathological ligands.

### Slurm

```yaml
slurm:
  cpu_partition: bigmem
  bigmem_partition: bigmem

  gpu_partition_unidock2: gpu_4090
  gpu_partition_unimol: gpu_h100
  gpu_partition_gnina: gpu_4090

  gpu_qos_unidock2: normal_gpu_4090
  gpu_qos_gnina: normal_gpu_4090

  gpu_exclusive: false
  max_parallel_unidock2: 12
  max_parallel_gnina: 12
  max_array_tasks_per_job: 1000

  defaults:
    cpus_per_task: 4
    mem: 32G
```

Notes:

- Keep `gpu_partition_*` as the real partition names, such as `gpu_4090`.
- Use `gpu_qos_*` only if your cluster requires explicit GPU QOS, such as `normal_gpu_4090`.
- `gpu_exclusive: false` is important on multi-GPU nodes; it lets Slurm pack multiple 1-GPU tasks on the same node.
- `max_parallel_*` should usually match your real per-user GPU allowance, not the theoretical node total.
- `max_array_tasks_per_job` is a safety valve for clusters with low array limits. Large stages are automatically split into batches.

## Multiple SDF Inputs

`inputs.ligands_sdf` supports:

- a single SDF file path
- a directory (we take `*.sdf`)
- a glob pattern like `/data/project/hanwen/commercial/*.sdf`
- a YAML list of SDF paths

## Runtime Behavior and Failure Tolerance

The workflow is intentionally tolerant of partial failures:

- Uni-Dock2 chunk summaries are submitted as separate arrays after docking arrays finish.
- Downstream submitters use dependencies that allow the workflow to continue past failed chunk tasks.
- UniMol writes a per-chunk `unimol_summary.csv` and can continue even when some ligands fail.
- gnina/finalize can continue even if some UniMol outputs are missing.

This means a few bad ligands should not block an entire large screen.

## Key Outputs

Inside `runs/<run_name>/`:

- `inputs/ligands_manifest.csv`
  - stable `ligand_id`
  - source SDF path
  - source molecule index
  - canonical SMILES recorded during input preparation
- `unidock_fast/chunk_summaries/summary_*.csv`
- `unidock_balance/chunk_summaries/summary_*.csv`
- `unidock_detail/chunk_summaries/summary_*.csv`
- `selection/clusters.csv`
- `unimol/chunks/chunk_*/unimol_summary.csv`
- `gnina/chunks/chunk_*/summary.csv`
- `final/final_scores.csv`

## Final Score Columns

`final/final_scores.csv` merges the main downstream scores:

- `unidock_detail_score`
- `minimizedAffinity`
- `CNNscore`
- `CNNaffinity`
- `smiles`
- `cluster_id`
- `cluster_rank`

Important notes:

- `smiles` comes from `inputs/ligands_manifest.csv`.
- In the pipeline's native output, `cluster_rank` is the rank **within a cluster by `unidock_detail_score`**, not by `CNNaffinity`.
- `minimizedAffinity` is a gnina/Vina-style energy-like score; more negative is better.
- `CNNscore` is a pose-quality style score; higher is better.
- `CNNaffinity` is the main gnina CNN affinity-style score; higher is better.

## Practical Resume Pattern

When updating code or Slurm rendering logic:

```bash
cd /data/project/hanwen/software/docking-pipeline
git pull
conda run -n dock-pipe docking-pipeline run configs/rnaseL.yaml --force --dry-run

cd runs/<run_name>
bash slurm/submit_workflow_deps.sh
```

Completed chunk outputs are skipped where possible, so you usually do not need to rebuild a run from scratch unless you deliberately change chunking or workflow semantics.

## Current Useful Files

- `configs/example_pipeline.yaml`
- `configs/rnaseL.yaml`
- `scripts/remote_env_check.sh`
- `src/docking_pipeline/cli.py`
- `src/docking_pipeline/slurm_render.py`
- `src/docking_pipeline/steps/finalize_rank.py`
