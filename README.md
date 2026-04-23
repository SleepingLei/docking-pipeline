# Docking Pipeline

Slurm-oriented pipeline for:

```text
Uni-Dock2 fast
  -> top 20%
  -> Uni-Dock2 balance
  -> top 15%
  -> ECFP6 KMeans clustering
  -> Uni-Mol Docking V2
  -> gnina rescoring
  -> score + rank
```

Remote target path:

```bash
/data/project/hanwen/software/docking-pipeline
```

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

Send back the generated `remote_env_report.txt` so the installation commands can be finalized against the actual CUDA, Slurm, and conda setup.

## Local Skeleton

Current useful files:

- `docs/environment_plan.md`: remote environment and resource plan.
- `docs/pipeline_todo.md`: implementation milestones and caveats.
- `configs/example_pipeline.yaml`: planned workflow config shape.
- `scripts/remote_env_check.sh`: remote cluster probe.
- `src/docking_pipeline/cli.py`: placeholder controller CLI.

The CLI shape is intentionally small for now:

```bash
python -m docking_pipeline.cli plan configs/example_pipeline.yaml
python -m docking_pipeline.cli validate-config configs/example_pipeline.yaml
```

## Run On Slurm (Stepwise)

1. Install the controller into a lightweight env (remote):

```bash
conda run -n dock-pipe python -m pip install -e .
```

2. Generate a run directory and Slurm scripts:

```bash
conda run -n dock-pipe docking-pipeline run configs/test_0_9999.yaml --dry-run
```

This creates `runs/<run_name>/slurm/*.sbatch`.

3. Submit (prepare only, then print the next commands):

```bash
conda run -n dock-pipe docking-pipeline run configs/test_0_9999.yaml --submit
```

The printed block is the intended step-by-step workflow. This is by design: later array sizes depend on chunk files produced earlier.

Note: this toolkit intentionally does not set `#SBATCH -t` (walltime) in generated scripts; it lets the cluster default/QOS decide.

## Multiple SDF Inputs

`inputs.ligands_sdf` supports:

- a single SDF file path
- a directory (we take `*.sdf`)
- a glob pattern like `/data/project/hanwen/commercial/*.sdf`
- a YAML list of SDF paths
