"""Pipeline step entrypoints for Slurm jobs.

These modules are invoked with:

  conda run -n dock-pipe python -m docking_pipeline.steps.<step> ...

They should be idempotent and file-oriented.
"""

