#!/bin/bash
#SBATCH --job-name=rnasel_rescore
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/rescore_csv_h100.%j.out
#SBATCH --chdir=/data/project/hanwen/software/docking-pipeline

set -euo pipefail

## Optional:
## 如果集群要求 account，请取消下一行注释并填真实值
##SBATCH --account=your_account
#SBATCH --time=12:00:00

CONDA_ENV="${CONDA_ENV:-dock-pipe}"

mkdir -p logs

echo "[info] host=$(hostname)"
echo "[info] cwd=$(pwd)"
echo "[info] date=$(date)"
echo "[info] conda_env=${CONDA_ENV}"

nvidia-smi || true

conda run -n "${CONDA_ENV}" python scripts/rescore_csv_with_unimol_gnina.py "$@"
