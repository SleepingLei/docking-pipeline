#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/project/hanwen/software/docking-pipeline}"
TOOLS_DIR="${TOOLS_DIR:-${PROJECT_DIR}/third_party}"
BIN_DIR="${BIN_DIR:-${PROJECT_DIR}/bin}"
CONDA_EXE="${CONDA_EXE:-conda}"
MAMBA_EXE="${MAMBA_EXE:-mamba}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_remote_envs.sh <command>

Commands:
  dock-pipe        Create/update the controller environment.
  unidock2         Create/update the Uni-Dock2 environment.
  unimol-v2        Create/update a source-based Uni-Mol Docking V2 environment.
  gnina-binary     Install the latest gnina release binary under ./bin.
  verify           Print versions/import checks for installed tools.
  all-basic        Run dock-pipe, unidock2, gnina-binary, then verify.

Environment:
  PROJECT_DIR      Default: /data/project/hanwen/software/docking-pipeline
  TOOLS_DIR        Default: $PROJECT_DIR/third_party
  BIN_DIR          Default: $PROJECT_DIR/bin
  GNINA_TAG        Default: latest release from GitHub API
  GNINA_ASSET_HINT Default: cuda12.8

Notes:
  Uni-Mol Docking V2 still needs its checkpoint file. Put it somewhere stable,
  for example $PROJECT_DIR/models/unimol_docking_v2_240517.pt.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

conda_env_exists() {
  "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "$1"
}

solver() {
  if command_exists "${MAMBA_EXE}"; then
    printf "%s" "${MAMBA_EXE}"
  else
    printf "%s" "${CONDA_EXE}"
  fi
}

ensure_conda_shell() {
  local conda_base
  conda_base="$("${CONDA_EXE}" info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
}

create_dock_pipe() {
  ensure_conda_shell
  local solver_cmd
  solver_cmd="$(solver)"
  if ! conda_env_exists dock-pipe; then
    "${solver_cmd}" create -n dock-pipe -c conda-forge --override-channels \
      python=3.10 'numpy<2' pandas rdkit scikit-learn pyyaml tqdm click \
      -y
  else
    "${solver_cmd}" install -n dock-pipe -c conda-forge --override-channels \
      python=3.10 'numpy<2' pandas rdkit scikit-learn pyyaml tqdm click \
      -y
  fi
  "${CONDA_EXE}" run -n dock-pipe python - <<'PY'
import numpy, pandas, rdkit, sklearn, yaml
print("dock-pipe ok")
PY
}

create_unidock2() {
  ensure_conda_shell
  local solver_cmd
  solver_cmd="$(solver)"
  if ! conda_env_exists unidock2; then
    "${solver_cmd}" create -n unidock2 -c conda-forge --override-channels \
      python=3.10 -y
  fi
  "${solver_cmd}" install -n unidock2 \
    unidock2 cuda-version=12.0 \
    -c http://quetz.dp.tech:8088/get/baymax \
    -c conda-forge \
    --override-channels \
    --no-repodata-use-zst \
    -y
  "${CONDA_EXE}" run -n unidock2 unidock2 --version
}

create_unimol_v2() {
  ensure_conda_shell
  local solver_cmd
  solver_cmd="$(solver)"
  mkdir -p "${TOOLS_DIR}"
  if ! conda_env_exists unimol-v2; then
    "${solver_cmd}" create -n unimol-v2 python=3.10 -y
  fi

  "${solver_cmd}" install -n unimol-v2 -c pytorch -c nvidia -c conda-forge \
    pytorch pytorch-cuda=12.1 \
    'numpy<2' pandas scikit-learn tqdm pyyaml lmdb \
    -y

  "${CONDA_EXE}" run -n unimol-v2 python -m pip install \
    rdkit-pypi==2022.9.3 biopandas

  if [[ ! -d "${TOOLS_DIR}/Uni-Core/.git" ]]; then
    git clone https://github.com/dptech-corp/Uni-Core.git "${TOOLS_DIR}/Uni-Core"
  fi
  if [[ ! -d "${TOOLS_DIR}/Uni-Mol/.git" ]]; then
    git clone https://github.com/deepmodeling/Uni-Mol.git "${TOOLS_DIR}/Uni-Mol"
  fi

  (
    cd "${TOOLS_DIR}/Uni-Core"
    "${CONDA_EXE}" run -n unimol-v2 python setup.py install --disable-cuda-ext
  )

  "${CONDA_EXE}" run -n unimol-v2 python - <<'PY'
import torch
import rdkit
import lmdb
import pandas
import sklearn
import unicore
print("unimol-v2 imports ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
}

install_gnina_binary() {
  mkdir -p "${BIN_DIR}"
  python - "$BIN_DIR" <<'PY'
import json
import os
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

bin_dir = Path(sys.argv[1])
tag = os.environ.get("GNINA_TAG", "").strip()
hint = os.environ.get("GNINA_ASSET_HINT", "cuda12.8").lower()

if tag:
    api = f"https://api.github.com/repos/gnina/gnina/releases/tags/{tag}"
else:
    api = "https://api.github.com/repos/gnina/gnina/releases/latest"

with urllib.request.urlopen(api, timeout=60) as response:
    release = json.load(response)

assets = release.get("assets", [])
if not assets:
    raise SystemExit("No gnina release assets found")

print("Available gnina assets:")
for asset in assets:
    print(" -", asset["name"])

def score(asset):
    name = asset["name"].lower()
    points = 0
    if hint and hint in name:
        points += 100
    if "cuda12.8" in name:
        points += 80
    if "cuda12" in name:
        points += 60
    if "linux" in name:
        points += 20
    if "gnina" in name:
        points += 10
    if any(x in name for x in ("sha", "md5", "checksum")):
        points -= 100
    return points

asset = sorted(assets, key=score, reverse=True)[0]
name = asset["name"]
url = asset["browser_download_url"]
download_path = bin_dir / name
print(f"Downloading {name} from {url}")
urllib.request.urlretrieve(url, download_path)

gnina_path = bin_dir / "gnina"

if tarfile.is_tarfile(download_path):
    extract_dir = bin_dir / f".extract_{download_path.stem}"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(download_path) as tf:
        tf.extractall(extract_dir)
    candidates = [p for p in extract_dir.rglob("gnina") if p.is_file()]
    if not candidates:
        raise SystemExit("Could not find gnina executable in tar asset")
    gnina_path.write_bytes(candidates[0].read_bytes())
elif zipfile.is_zipfile(download_path):
    extract_dir = bin_dir / f".extract_{download_path.stem}"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(download_path) as zf:
        zf.extractall(extract_dir)
    candidates = [p for p in extract_dir.rglob("gnina") if p.is_file()]
    if not candidates:
        raise SystemExit("Could not find gnina executable in zip asset")
    gnina_path.write_bytes(candidates[0].read_bytes())
else:
    gnina_path.write_bytes(download_path.read_bytes())

mode = gnina_path.stat().st_mode
gnina_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print(f"Installed {gnina_path}")
PY
  "${BIN_DIR}/gnina" --version || true
}

verify_envs() {
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "TOOLS_DIR=${TOOLS_DIR}"
  echo "BIN_DIR=${BIN_DIR}"
  "${CONDA_EXE}" env list
  "${CONDA_EXE}" run -n dock-pipe python - <<'PY' || true
import rdkit, pandas, sklearn
print("dock-pipe imports ok")
PY
  "${CONDA_EXE}" run -n unidock2 unidock2 --version || true
  "${CONDA_EXE}" run -n unimol-v2 python - <<'PY' || true
import torch
print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
PY
  "${BIN_DIR}/gnina" --version || true
}

case "$1" in
  dock-pipe)
    create_dock_pipe
    ;;
  unidock2)
    create_unidock2
    ;;
  unimol-v2)
    create_unimol_v2
    ;;
  gnina-binary)
    install_gnina_binary
    ;;
  verify)
    verify_envs
    ;;
  all-basic)
    create_dock_pipe
    create_unidock2
    install_gnina_binary
    verify_envs
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 2
    ;;
esac
