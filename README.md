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
