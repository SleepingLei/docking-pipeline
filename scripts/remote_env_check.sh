#!/usr/bin/env bash
set -u

PROJECT_DIR="/data/project/hanwen/software/docking-pipeline"
SUBMIT_TESTS=0
GPU_PARTITIONS="${GPU_PARTITIONS:-gpu_4090 gpu_h100}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname 2>/dev/null || echo unknown_host)"
OUT_DIR="${PWD}/remote_env_check_${HOST}_${TIMESTAMP}"
REPORT="${OUT_DIR}/remote_env_report.txt"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/remote_env_check.sh [options]

Options:
  --project-dir PATH   Project path on the remote machine.
                       Default: /data/project/hanwen/software/docking-pipeline
  --submit-tests       Submit tiny Slurm CPU/GPU smoke jobs.
  -h, --help           Show this help.

Environment:
  GPU_PARTITIONS       Space-separated GPU partitions to smoke test when
                       --submit-tests is used. Default: "gpu_4090 gpu_h100".
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    --submit-tests)
      SUBMIT_TESTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${OUT_DIR}"

section() {
  {
    echo
    echo "===== $* ====="
  } | tee -a "${REPORT}"
}

run() {
  local label="$1"
  shift
  {
    echo
    echo "+ ${label}"
    "$@"
    local rc=$?
    echo "[exit=${rc}]"
    return 0
  } 2>&1 | tee -a "${REPORT}"
}

run_shell() {
  local label="$1"
  shift
  {
    echo
    echo "+ ${label}"
    bash -lc "$*"
    local rc=$?
    echo "[exit=${rc}]"
    return 0
  } 2>&1 | tee -a "${REPORT}"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

write_slurm_smoke_jobs() {
  local cpu_job="${OUT_DIR}/slurm_cpu_smoke.sbatch"
  cat > "${cpu_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dock_cpu_smoke
#SBATCH --partition=normal
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --chdir=${OUT_DIR}
#SBATCH --output=${OUT_DIR}/slurm_cpu_smoke.%j.out

set -euo pipefail
echo "host=\$(hostname)"
echo "date=\$(date)"
which python || true
python -V || true
EOF

  for partition in ${GPU_PARTITIONS}; do
    local gpu_job="${OUT_DIR}/slurm_gpu_${partition}_smoke.sbatch"
    cat > "${gpu_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dock_gpu_${partition}_smoke
#SBATCH --partition=${partition}
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --chdir=${OUT_DIR}
#SBATCH --output=${OUT_DIR}/slurm_gpu_${partition}_smoke.%j.out

set -euo pipefail
echo "host=\$(hostname)"
echo "date=\$(date)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch import failed:", repr(exc))
PY
EOF
  done
}

submit_slurm_smoke_jobs() {
  section "Slurm Smoke Job Submission"
  if ! have sbatch; then
    echo "sbatch not found; skip smoke job submission." | tee -a "${REPORT}"
    return 0
  fi

  run "submit CPU smoke job" sbatch "${OUT_DIR}/slurm_cpu_smoke.sbatch"
  for partition in ${GPU_PARTITIONS}; do
    local gpu_job="${OUT_DIR}/slurm_gpu_${partition}_smoke.sbatch"
    if [[ -f "${gpu_job}" ]]; then
      run "submit GPU smoke job (${partition})" sbatch "${gpu_job}"
    fi
  done
}

section "Basic Context"
echo "report=${REPORT}" | tee -a "${REPORT}"
echo "project_dir=${PROJECT_DIR}" | tee -a "${REPORT}"
run "date" date
run "whoami" whoami
run "hostname" hostname
run "pwd" pwd
run "uname -a" uname -a
run_shell "os release" 'test -f /etc/os-release && cat /etc/os-release || true'

section "Project Path"
run_shell "project path status" "ls -ld '${PROJECT_DIR}' || true; df -h '${PROJECT_DIR}' 2>/dev/null || df -h ."

section "Shell And Modules"
run "which bash" which bash
run "which zsh" which zsh
run_shell "module command" 'type module 2>/dev/null || true'
run_shell "module avail" 'module avail 2>&1 | head -200 || true'

section "Conda And Python"
run "which conda" which conda
run_shell "conda info" 'conda info || true'
run_shell "conda env list" 'conda env list || true'
run "which mamba" which mamba
run "which micromamba" which micromamba
run "which python" which python
run "python -V" python -V
run_shell "python package probe" 'python - <<'"'"'PY'"'"'
import importlib.util
for name in ["numpy", "pandas", "rdkit", "sklearn", "yaml", "torch"]:
    spec = importlib.util.find_spec(name)
    status = "FOUND" if spec else "missing"
    print(f"{name}: {status}")
PY'

section "Compiler And Build Tools"
run "which gcc" which gcc
run "gcc --version" gcc --version
run "which g++" which g++
run "g++ --version" g++ --version
run "which cmake" which cmake
run "cmake --version" cmake --version
run "which make" which make
run "make --version" make --version
run "which nvcc" which nvcc
run "nvcc --version" nvcc --version

section "CUDA Module Probe"
run_shell "module load cuda/13.0 and nvcc" 'module load cuda/13.0 2>/dev/null && which nvcc && nvcc --version || true'

section "CUDA And GPU"
run "which nvidia-smi" which nvidia-smi
run "nvidia-smi -L" nvidia-smi -L
run "nvidia-smi" nvidia-smi

section "Slurm"
run "which sinfo" which sinfo
run "sinfo" sinfo
run "which sbatch" which sbatch
run "which srun" which srun
run "which squeue" which squeue
run_shell "squeue current user" 'squeue -u "$(whoami)" || true'

section "Docking Tools In Current PATH"
run "which unidock2" which unidock2
run "unidock2 --version" unidock2 --version
run "which gnina" which gnina
run "gnina --version" gnina --version
run_shell "Uni-Mol repo path candidates" 'find "${HOME}" /data/project -maxdepth 5 -type d -name "Uni-Mol" 2>/dev/null | head -20 || true'

section "Network Probe"
run_shell "github probe" 'timeout 10 bash -lc "curl -I -L https://github.com >/dev/null" || true'
run_shell "conda-forge probe" 'timeout 10 bash -lc "curl -I -L https://conda.anaconda.org/conda-forge >/dev/null" || true'
run_shell "dptech quetz probe" 'timeout 10 bash -lc "curl -I -L http://quetz.dp.tech:8088/get/baymax >/dev/null" || true'
run_shell "huggingface probe" 'timeout 10 bash -lc "curl -I -L https://huggingface.co >/dev/null" || true'

section "Slurm Smoke Scripts"
write_slurm_smoke_jobs
find "${OUT_DIR}" -maxdepth 1 -name 'slurm_*_smoke.sbatch' -print | tee -a "${REPORT}"

if [[ "${SUBMIT_TESTS}" -eq 1 ]]; then
  submit_slurm_smoke_jobs
else
  echo "Smoke jobs were written but not submitted. Re-run with --submit-tests to submit them." | tee -a "${REPORT}"
fi

section "Done"
echo "Remote environment report written to: ${REPORT}" | tee -a "${REPORT}"
