# Remote Environment Plan

远程项目路径：

```bash
/data/project/hanwen/software/docking-pipeline
```

先不要急着一次性装完整环境。推荐顺序是：

1. 在登录节点运行环境探针，确认 CUDA、Slurm、conda、编译工具、网络访问。
2. 用 Slurm 提交最小 CPU/GPU smoke job，确认作业能拿到 GPU。
3. 分环境安装并验证 Uni-Dock2、Uni-Mol Docking V2、gnina。
4. 用 10-100 个 ligand 跑完整 smoke pipeline。
5. 再扩大到 1k、10k，最后才跑全量。

## 1. Environment Probe

把本仓库同步到远程后，在远程项目目录运行：

```bash
cd /data/project/hanwen/software/docking-pipeline
bash scripts/remote_env_check.sh
```

如果要顺手提交 Slurm smoke job：

```bash
bash scripts/remote_env_check.sh --submit-tests
```

脚本会生成一个 `remote_env_check_*` 目录，其中 `remote_env_report.txt` 是最重要的文件。把这个报告贴回来，就能决定该用 CUDA 11 还是 CUDA 12 路线、gnina 是否直接用 binary、Uni-Mol 是否需要源码/容器方案。

从 2026-04-23 的远程报告看，实际目录是 `/data/project/hanwen/software/docking-pipeline`；之前的 `docking-pipline` 是拼写偏差。后续统一用 `docking-pipeline`。

## 2. Recommended Environment Split

不要把所有工具强行塞进一个 conda 环境。依赖版本容易冲突，尤其是 Uni-Mol Docking V2 和 Uni-Dock2/gnina 的 CUDA 期望不同。

推荐拆成：

| Environment | Purpose | Main Packages |
| --- | --- | --- |
| `dock-pipe` | pipeline 控制、SDF/manifest、聚类、排序 | Python, RDKit, pandas, numpy, scikit-learn, PyYAML |
| `unidock2` | Uni-Dock2 docking | Python 3.10, CUDA >= 12, unidock2 |
| `unimol-v2` | Uni-Mol Docking V2 推理 | Uni-Core, PyTorch, RDKit 2022.9.3, biopandas |
| `gnina` | gnina rescoring | 优先预编译 binary；必要时源码编译 |

pipeline 里用 `conda run -n <env>` 或环境激活脚本调用各工具。用户入口仍然是一个 workflow，只是底层隔离依赖。

## 3. Candidate Install Commands

这些命令要等 `remote_env_check.sh` 输出确认后再执行。

### dock-pipe

```bash
conda create -n dock-pipe python=3.10 -y
conda activate dock-pipe
conda install -c conda-forge rdkit pandas numpy scikit-learn pyyaml tqdm pyarrow -y
python - <<'PY'
import rdkit, pandas, sklearn, yaml
print("dock-pipe ok")
PY
```

### Uni-Dock2

Uni-Dock2 官方 conda 方式要求 Python 3.10 和 CUDA >= 12。

```bash
conda create -n unidock2 python=3.10 -y
conda activate unidock2
conda install unidock2 cuda-version=12.0 \
  -c http://quetz.dp.tech:8088/get/baymax \
  -c conda-forge -y
unidock2 --version
```

如果远程节点的 CUDA/驱动或网络不适合 conda 包，再考虑源码构建。

### Uni-Mol Docking V2

Uni-Mol Docking V2 官方 Dockerfile 基于 `dptechnology/unicore:latest-pytorch1.12.1-cuda11.6-rdma`，Python 包依赖包括 `rdkit-pypi==2022.9.3` 和 `biopandas`。如果集群支持容器，优先考虑 Apptainer/Singularity；如果不支持，就用 conda/source 环境。

远程集群是 4090/H100。官方 Dockerfile 的 PyTorch 1.12.1 + CUDA 11.6 对这些新卡风险偏高，优先走 `unimol-v2` conda/source 环境，并用较新的 PyTorch CUDA 包；Uni-Core 先用 `--disable-cuda-ext` 安装，避免本地 CUDA toolkit 和 PyTorch CUDA 版本不一致。

最小验证目标：

```bash
cd /path/to/Uni-Mol/unimol_docking_v2/interface
python demo.py \
  --mode single \
  --conf-size 10 \
  --cluster \
  --input-protein ../example_data/protein.pdb \
  --input-ligand ../example_data/ligand.sdf \
  --input-docking-grid ../example_data/docking_grid.json \
  --output-ligand-name ligand_predict \
  --output-ligand-dir predict_sdf \
  --steric-clash-fix \
  --model-dir checkpoint_best.pt
```

注意：当前 Uni-Mol Docking V2 predictor 里推理命令写了 `CUDA_VISIBLE_DEVICES="0"`。用 Slurm 每个 job 分配 1 张 GPU 最稳，因为 Slurm 通常会把作业内可见 GPU 映射成 `0`。

### gnina

优先下载官方 release 的预编译 binary；源码编译依赖比较重，且官方说明要求 CUDA >= 12、OpenBabel3、CMake、Boost 等。

验证：

```bash
gnina --version
gnina -r protein.pdb -l pose.sdf \
  --score_only \
  --center_x X --center_y Y --center_z Z \
  --size_x SX --size_y SY --size_z SZ
```

如果需要输出带 `CNNscore`/`CNNaffinity` 的 SDF，则用：

```bash
gnina -r protein.pdb -l pose.sdf \
  --minimize \
  --center_x X --center_y Y --center_z Z \
  --size_x SX --size_y SY --size_z SZ \
  -o gnina_minimized.sdf
```

## 4. Resource Plan

| Stage | Main Resource | Suggested Partition | Notes |
| --- | --- | --- | --- |
| SDF split / manifest / score parsing | CPU + IO | `normal` | 4-16 CPU, avoid login node for huge SDF |
| Uni-Dock2 `fast` | GPU | `gpu_4090` | high throughput; use Slurm array chunks |
| Uni-Dock2 `balance` | GPU | `gpu_4090` | top 20% only |
| Uni-Dock2 `detail` | GPU | `gpu_4090` or `gpu_h100` | top 15% after balance; H100 only if queue is open |
| ECFP6 + KMeans | CPU + memory | `normal` / `bigmem` | fingerprints are CPU; huge top set may need `bigmem` |
| Uni-Mol Docking V2 | GPU + CPU preprocessing | `gpu_h100` preferred | PyTorch inference + RDKit conformer generation |
| gnina rescoring | GPU | `gpu_4090` | CNN rescoring benefits from GPU; one GPU per job |
| final ranking/report | CPU | `normal` | lightweight |

## 5. Batch Strategy

从小到大调参：

1. 100 ligands: smoke test，验证文件格式和 score parsing。
2. 1k ligands: 调 Uni-Dock2 chunk size 和 Slurm array 并发。
3. 10k ligands: 调 Uni-Mol batch size、gnina job 粒度。
4. 全量: 固定参数后运行。

建议初始 chunk：

| Stage | Initial Chunk Size |
| --- | --- |
| Uni-Dock2 fast | 2k-10k ligands/job |
| Uni-Dock2 balance | 1k-5k ligands/job |
| Uni-Dock2 detail | 500-2k ligands/job |
| Uni-Mol Docking V2 | 32-256 ligands/job, depends on GPU memory |
| gnina score/minimize | 100-1000 poses/job |

每个阶段都必须支持 resume：如果 chunk 的 output SDF 和 manifest 状态都存在，就跳过。

## 6. Immediate Decisions Still Needed

- 输入 protein 是完整 protein PDB，还是已经裁剪好的 pocket PDB？
- docking box 的 center/size 从哪里来：手动配置、参考 ligand、还是 pocket 自动计算？
- KMeans 的 `N` 是固定数量，还是按 top set 大小动态计算？
- Uni-Mol 阶段是否使用 `--use_current_ligand_conf` 复用 Uni-Dock2 pose？
- gnina 最终用 `--score_only` 还是 `--minimize`？
