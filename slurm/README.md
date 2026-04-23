# Slurm Scripts Live Under `runs/<run>/slurm/`

This directory is intentionally empty in the repository.

The pipeline generates per-run Slurm scripts into:

`runs/<run_name>/slurm/*.sbatch` and `runs/<run_name>/slurm/submit_workflow.sh`

Run generation:

```bash
conda run -n dock-pipe docking-pipeline run configs/example_pipeline.yaml --dry-run
```

Then submit step-by-step from inside the run directory.

