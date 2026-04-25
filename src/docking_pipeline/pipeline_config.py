from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _as_path(p: str | Path) -> Path:
    if isinstance(p, Path):
        return p
    return Path(p)


def _resolve_path(p: str | Path, *, base_dir: Path) -> Path:
    p = _as_path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _expand_sdf_inputs(paths: list[Path]) -> list[Path]:
    """
    Expand inputs that can be:
    - a file path
    - a directory (we take *.sdf)
    - a glob pattern (e.g. /path/to/*.sdf)
    """
    import glob

    out: list[Path] = []
    for p in paths:
        s = str(p)
        if any(ch in s for ch in ["*", "?", "["]):
            matches = [Path(x) for x in glob.glob(s)]
            out.extend(sorted(matches))
            continue
        if p.is_dir():
            out.extend(sorted(p.glob("*.sdf")))
            continue
        out.append(p)
    # de-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


@dataclass(frozen=True)
class DockingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class RunSection:
    name: str
    work_dir: Path
    project_dir: Path
    random_seed: int = 1234567


@dataclass(frozen=True)
class InputsSection:
    receptor_pdb: Path
    ligands_sdf: list[Path]
    docking_box: DockingBox


@dataclass(frozen=True)
class SelectionSection:
    after_fast_top_fraction: float = 0.20
    after_balance_top_fraction: float = 0.15
    cluster_count: int = 100
    cluster_representatives_per_cluster: int = 1


@dataclass(frozen=True)
class UniDock2StageSection:
    search_mode: str
    chunk_size: int


@dataclass(frozen=True)
class UniDock2Section:
    env_name: str = "unidock2"
    gpu_device_id: int = 0
    num_pose: int = 10
    rmsd_limit: float = 1.0
    energy_range: float = 5.0
    temp_dir: str = "/tmp"
    stages: dict[str, UniDock2StageSection] = field(default_factory=dict)


@dataclass(frozen=True)
class FingerprintSection:
    type: str = "ecfp"
    radius: int = 3  # Morgan radius 3 == ECFP6
    n_bits: int = 2048


@dataclass(frozen=True)
class ClusteringSection:
    env_name: str = "dock-pipe"
    fingerprint: FingerprintSection = field(default_factory=FingerprintSection)
    method: str = "kmeans"


@dataclass(frozen=True)
class UniMolSection:
    env_name: str = "unimol-v2"
    repo_dir: Path = Path("/data/project/hanwen/software/docking-pipeline/third_party/Uni-Mol/unimol_docking_v2")
    model_path: Path = Path("/data/project/hanwen/software/docking-pipeline/models/unimol_docking_v2_240517.pt")
    mode: str = "batch_one2many"
    batch_size: int = 8
    conf_size: int = 10
    cluster_conformers: bool = True
    use_current_ligand_conf: bool = False
    steric_clash_fix: bool = True
    chunk_size: int = 128


@dataclass(frozen=True)
class GninaSection:
    executable: str = "gnina"
    # For pipeline usage we typically want to rescore existing poses without changing them.
    mode: str = "score_only"  # or minimize
    chunk_size: int = 500
    # Optional. If None/empty, do not pass `--cnn ...` (let gnina defaults decide).
    cnn: str | None = None


@dataclass(frozen=True)
class SlurmDefaults:
    # If None, do not set `#SBATCH -t` and let the cluster default/QOS decide.
    time: str | None = None
    cpus_per_task: int = 8
    mem: str = "32G"


@dataclass(frozen=True)
class SlurmSection:
    account: str | None = None
    cpu_partition: str = "normal"
    bigmem_partition: str = "bigmem"
    gpu_partition_unidock2: str = "gpu_4090"
    gpu_partition_unimol: str = "gpu_h100"
    gpu_partition_gnina: str = "gpu_4090"
    # If true, request node exclusivity for GPU jobs. Leave false for multi-GPU nodes
    # so Slurm can pack up to (GPUs per node) jobs on a single node.
    gpu_exclusive: bool = False
    # Optional caps on how many array tasks can run concurrently (useful to match GPU capacity).
    # Example: with 2 nodes * 8 GPUs/node, set unidock2=16 to keep GPUs saturated without flooding the queue.
    max_parallel_unidock2: int | None = None
    max_parallel_unimol: int | None = None
    max_parallel_gnina: int | None = None
    defaults: SlurmDefaults = field(default_factory=SlurmDefaults)


@dataclass(frozen=True)
class DockingPipelineConfig:
    run: RunSection
    inputs: InputsSection
    selection: SelectionSection = field(default_factory=SelectionSection)
    unidock2: UniDock2Section = field(default_factory=UniDock2Section)
    clustering: ClusteringSection = field(default_factory=ClusteringSection)
    unimol: UniMolSection = field(default_factory=UniMolSection)
    gnina: GninaSection = field(default_factory=GninaSection)
    slurm: SlurmSection = field(default_factory=SlurmSection)

    def to_yaml(self) -> str:
        def conv(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: conv(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(x) for x in obj]
            return obj

        data = conv(self)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)

    @staticmethod
    def from_yaml(text: str) -> "DockingPipelineConfig":
        data = yaml.safe_load(text)
        cfg = _parse_config_dict(data)
        return cfg


def load_config(config_path: Path) -> DockingPipelineConfig:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _parse_config_dict(data)


def normalize_config(cfg: DockingPipelineConfig, *, config_path: Path) -> DockingPipelineConfig:
    # Resolve relative paths using project_dir for run/work_dir and config file dir for inputs.
    cfg_dir = config_path.parent.resolve()
    project_dir = cfg.run.project_dir
    if not project_dir.is_absolute():
        project_dir = (cfg_dir / project_dir).resolve()

    work_dir = cfg.run.work_dir
    if not work_dir.is_absolute():
        work_dir = (project_dir / work_dir).resolve()

    receptor_pdb = _resolve_path(cfg.inputs.receptor_pdb, base_dir=cfg_dir)
    ligands_sdf = [_resolve_path(p, base_dir=cfg_dir) for p in cfg.inputs.ligands_sdf]
    ligands_sdf = _expand_sdf_inputs(ligands_sdf)

    unimol_repo_dir = _resolve_path(cfg.unimol.repo_dir, base_dir=cfg_dir)
    unimol_model_path = _resolve_path(cfg.unimol.model_path, base_dir=cfg_dir)

    run = RunSection(
        name=cfg.run.name,
        work_dir=work_dir,
        project_dir=project_dir,
        random_seed=cfg.run.random_seed,
    )
    inputs = InputsSection(
        receptor_pdb=receptor_pdb,
        ligands_sdf=ligands_sdf,
        docking_box=cfg.inputs.docking_box,
    )
    unimol = UniMolSection(
        env_name=cfg.unimol.env_name,
        repo_dir=unimol_repo_dir,
        model_path=unimol_model_path,
        mode=cfg.unimol.mode,
        batch_size=cfg.unimol.batch_size,
        conf_size=cfg.unimol.conf_size,
        cluster_conformers=cfg.unimol.cluster_conformers,
        use_current_ligand_conf=cfg.unimol.use_current_ligand_conf,
        steric_clash_fix=cfg.unimol.steric_clash_fix,
        chunk_size=cfg.unimol.chunk_size,
    )
    cfg = DockingPipelineConfig(
        run=run,
        inputs=inputs,
        selection=cfg.selection,
        unidock2=cfg.unidock2,
        clustering=cfg.clustering,
        unimol=unimol,
        gnina=cfg.gnina,
        slurm=cfg.slurm,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: DockingPipelineConfig) -> None:
    if not (0.0 < cfg.selection.after_fast_top_fraction <= 1.0):
        raise ValueError("after_fast_top_fraction must be in (0,1]")
    if not (0.0 < cfg.selection.after_balance_top_fraction <= 1.0):
        raise ValueError("after_balance_top_fraction must be in (0,1]")
    if cfg.selection.cluster_count <= 0:
        raise ValueError("cluster_count must be > 0")
    if cfg.selection.cluster_representatives_per_cluster <= 0:
        raise ValueError("cluster_representatives_per_cluster must be > 0")
    for name, stage in cfg.unidock2.stages.items():
        if stage.search_mode not in ("fast", "balance", "detail"):
            raise ValueError(f"unidock2 stage {name} has invalid search_mode={stage.search_mode!r}")
        if stage.chunk_size <= 0:
            raise ValueError(f"unidock2 stage {name} chunk_size must be > 0")


def _parse_config_dict(data: dict[str, Any]) -> DockingPipelineConfig:
    run_d = data.get("run") or {}
    inputs_d = data.get("inputs") or {}

    box_d = inputs_d.get("docking_box") or {}
    box = DockingBox(
        center=tuple(float(x) for x in box_d.get("center", [0.0, 0.0, 0.0])),
        size=tuple(float(x) for x in box_d.get("size", [30.0, 30.0, 30.0])),
    )

    run = RunSection(
        name=str(run_d.get("name", "run")),
        work_dir=_as_path(run_d.get("work_dir", "runs/run")),
        project_dir=_as_path(run_d.get("project_dir", ".")),
        random_seed=int(run_d.get("random_seed", 1234567)),
    )
    inputs = InputsSection(
        receptor_pdb=_as_path(inputs_d.get("receptor_pdb", "")),
        ligands_sdf=[
            _as_path(x) for x in (inputs_d.get("ligands_sdf", []) if isinstance(inputs_d.get("ligands_sdf", []), list) else [inputs_d.get("ligands_sdf", "")])
        ],
        docking_box=box,
    )

    sel_d = data.get("selection") or {}
    selection = SelectionSection(
        after_fast_top_fraction=float(sel_d.get("after_fast_top_fraction", 0.20)),
        after_balance_top_fraction=float(sel_d.get("after_balance_top_fraction", 0.15)),
        cluster_count=int(sel_d.get("cluster_count", 100)),
        cluster_representatives_per_cluster=int(sel_d.get("cluster_representatives_per_cluster", 1)),
    )

    ud_d = data.get("unidock2") or {}
    stages_d = (ud_d.get("stages") or {}) if isinstance(ud_d.get("stages"), dict) else {}
    stages: dict[str, UniDock2StageSection] = {}
    for stage_name, sd in stages_d.items():
        stages[str(stage_name)] = UniDock2StageSection(
            search_mode=str(sd.get("search_mode", stage_name)),
            chunk_size=int(sd.get("chunk_size", 1000)),
        )
    unidock2 = UniDock2Section(
        env_name=str(ud_d.get("env_name", "unidock2")),
        gpu_device_id=int(ud_d.get("gpu_device_id", 0)),
        num_pose=int(ud_d.get("num_pose", 10)),
        rmsd_limit=float(ud_d.get("rmsd_limit", 1.0)),
        energy_range=float(ud_d.get("energy_range", 5.0)),
        temp_dir=str(ud_d.get("temp_dir", "/tmp")),
        stages=stages,
    )

    cl_d = data.get("clustering") or {}
    fp_d = cl_d.get("fingerprint") or {}
    clustering = ClusteringSection(
        env_name=str(cl_d.get("env_name", "dock-pipe")),
        fingerprint=FingerprintSection(
            type=str(fp_d.get("type", "ecfp")),
            radius=int(fp_d.get("radius", 3)),
            n_bits=int(fp_d.get("n_bits", 2048)),
        ),
        method=str(cl_d.get("method", "kmeans")),
    )

    um_d = data.get("unimol") or {}
    unimol = UniMolSection(
        env_name=str(um_d.get("env_name", "unimol-v2")),
        repo_dir=_as_path(um_d.get("repo_dir", "/data/project/hanwen/software/docking-pipeline/third_party/Uni-Mol/unimol_docking_v2")),
        model_path=_as_path(um_d.get("model_path", "/data/project/hanwen/software/docking-pipeline/models/unimol_docking_v2_240517.pt")),
        mode=str(um_d.get("mode", "batch_one2many")),
        batch_size=int(um_d.get("batch_size", 8)),
        conf_size=int(um_d.get("conf_size", 10)),
        cluster_conformers=bool(um_d.get("cluster_conformers", True)),
        use_current_ligand_conf=bool(um_d.get("use_current_ligand_conf", False)),
        steric_clash_fix=bool(um_d.get("steric_clash_fix", True)),
        chunk_size=int(um_d.get("chunk_size", 128)),
    )

    gn_d = data.get("gnina") or {}
    cnn_raw = gn_d.get("cnn", None)
    if cnn_raw is None:
        cnn_val = None
    else:
        s = str(cnn_raw).strip()
        if not s:
            cnn_val = None
        elif s.lower() in {"default", "none", "null"}:
            # "default" is not a valid gnina model name; users often intend "use gnina's default".
            cnn_val = None
        else:
            cnn_val = s
    gnina = GninaSection(
        executable=str(gn_d.get("executable", "gnina")),
        mode=str(gn_d.get("mode", "score_only")),
        chunk_size=int(gn_d.get("chunk_size", 500)),
        cnn=cnn_val,
    )

    sl_d = data.get("slurm") or {}
    defs_d = sl_d.get("defaults") or {}
    time_raw = defs_d.get("time", None)
    if time_raw is None:
        time_val = None
    else:
        time_val = str(time_raw)
    slurm = SlurmSection(
        account=sl_d.get("account", None),
        cpu_partition=str(sl_d.get("cpu_partition", "normal")),
        bigmem_partition=str(sl_d.get("bigmem_partition", "bigmem")),
        gpu_partition_unidock2=str(sl_d.get("gpu_partition_unidock2", "gpu_4090")),
        gpu_partition_unimol=str(sl_d.get("gpu_partition_unimol", "gpu_h100")),
        gpu_partition_gnina=str(sl_d.get("gpu_partition_gnina", "gpu_4090")),
        defaults=SlurmDefaults(
            time=time_val,
            cpus_per_task=int(defs_d.get("cpus_per_task", 8)),
            mem=str(defs_d.get("mem", "32G")),
        ),
    )

    cfg = DockingPipelineConfig(
        run=run,
        inputs=inputs,
        selection=selection,
        unidock2=unidock2,
        clustering=clustering,
        unimol=unimol,
        gnina=gnina,
        slurm=slurm,
    )
    _validate(cfg)
    return cfg
