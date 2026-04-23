# Remote Report Notes: 2026-04-23

Based on the environment report from `login01`.

## Confirmed

- OS: Ubuntu 24.04.3 LTS.
- Filesystem: `/data` GPFS with plenty of free space.
- Slurm is available: `normal`, `bigmem`, `gpu_4090`, `gpu_h100`.
- Apptainer and Singularity modules are available.
- Conda and mamba are available under `/data/home/dwlei/anaconda3`.
- Build tools exist on login node: GCC 13.3, CMake 3.28, Make 4.3.
- Module system provides `cuda/13.0`.

## Expected / Not A Problem

- `nvidia-smi` and `nvcc` were not found on the login node before loading modules.
- `unidock2`, `gnina`, and Uni-Mol are not installed yet.
- The Python package probe failed because of a quoting bug in the first probe script; fixed in the repo.

## Path Correction

Actual current path:

```bash
/data/project/hanwen/software/docking-pipeline
```

The earlier path `/data/project/hanwen/software/docking-pipline` does not exist. Use `docking-pipeline` from now on.

## Next Remote Checks

The report submitted these Slurm jobs:

```text
207268 CPU smoke
207269 gpu_4090 smoke
207270 gpu_h100 smoke
```

Check them with:

```bash
squeue -j 207268,207269,207270
sacct -j 207268,207269,207270 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList
```

After they finish, inspect the output files in the report directory:

```bash
cd /data/project/hanwen/software/docking-pipeline/remote_env_check_login01_20260423_131215
ls -lh *.out
cat slurm_cpu_smoke.*.out
cat slurm_gpu_gpu_4090_smoke.*.out
cat slurm_gpu_gpu_h100_smoke.*.out
```

The important GPU smoke output is `nvidia-smi`, `CUDA_VISIBLE_DEVICES`, and whether `torch.cuda.is_available()` is true in the base env. It is fine if base env does not have torch; the real check happens after installing `unimol-v2`.
