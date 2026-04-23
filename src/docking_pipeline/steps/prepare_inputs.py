from __future__ import annotations

import argparse
import math
from pathlib import Path

from rdkit import Chem

from docking_pipeline.pipeline_config import DockingPipelineConfig
from docking_pipeline.steps.common import ensure_dir, load_run_cfg, write_csv


def _iter_sdf(sdf_path: Path):
    # RDKit supplier keeps a file handle; do not convert to list for big SDFs.
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for idx, mol in enumerate(supplier):
        yield idx, mol


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare initial ligand chunks and Uni-Dock2 configs.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir

    # Prepare directories
    fast_chunks_dir = run_dir / "inputs" / "fast" / "chunks"
    ensure_dir(fast_chunks_dir)
    ensure_dir(run_dir / "inputs" / "fast")
    ensure_dir(run_dir / "unidock_fast" / "chunks")
    ensure_dir(run_dir / "unidock_fast" / "chunk_summaries")
    ensure_dir(run_dir / "unidock_balance" / "chunks")
    ensure_dir(run_dir / "unidock_balance" / "chunk_summaries")
    ensure_dir(run_dir / "unidock_detail" / "chunks")
    ensure_dir(run_dir / "unidock_detail" / "chunk_summaries")
    ensure_dir(run_dir / "unidock_detail" / "best_poses")
    ensure_dir(run_dir / "selection")

    # Chunking
    chunk_size = cfg.unidock2.stages["fast"].chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    manifest_rows: list[dict[str, object]] = []
    next_chunk_id = 0
    current_chunk_id = -1
    in_chunk = 0
    writer: Chem.SDWriter | None = None

    def ensure_writer():
        nonlocal next_chunk_id, current_chunk_id, in_chunk, writer
        if writer is None or in_chunk >= chunk_size:
            if writer is not None:
                writer.close()
            current_chunk_id = next_chunk_id
            out_sdf_path = fast_chunks_dir / f"chunk{current_chunk_id}.sdf"
            writer = Chem.SDWriter(str(out_sdf_path))
            in_chunk = 0
            next_chunk_id += 1

    ligand_counter = 0
    input_sdfs = list(cfg.inputs.ligands_sdf)
    if not input_sdfs:
        raise ValueError("inputs.ligands_sdf is empty")

    for sdf_path in input_sdfs:
        if not sdf_path.exists():
            raise FileNotFoundError(sdf_path)
        if not sdf_path.is_file():
            raise ValueError(f"Not a file: {sdf_path}")

        for src_idx, mol in _iter_sdf(sdf_path):
            ligand_id = f"LIG_{ligand_counter:012d}"
            ligand_counter += 1

            if mol is None:
                manifest_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "source_sdf": str(sdf_path),
                        "source_mol_idx": src_idx,
                        "status": "rdkit_none",
                    }
                )
                continue

            # Attach stable ID so Uni-Dock2 output SDF preserves it.
            mol.SetProp("ligand_id", ligand_id)
            if not mol.HasProp("_Name"):
                mol.SetProp("_Name", ligand_id)

            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                smiles = ""

            ensure_writer()
            writer.write(mol)
            in_chunk += 1

            manifest_rows.append(
                {
                    "ligand_id": ligand_id,
                    "source_sdf": str(sdf_path),
                    "source_mol_idx": src_idx,
                    "prepared_chunk": int(current_chunk_id),
                    "smiles": smiles,
                    "name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                    "status": "ok",
                }
            )

    if writer is None:
        raise RuntimeError("No valid ligands were written to any chunk; check input ligands SDF.")
    writer.close()

    # Write manifest
    write_csv(
        run_dir / "inputs" / "ligands_manifest.csv",
        manifest_rows,
        fieldnames=[
            "ligand_id",
            "source_sdf",
            "source_mol_idx",
            "prepared_chunk",
            "smiles",
            "name",
            "status",
        ],
    )

    # Generate Uni-Dock2 stage config YAMLs (fast/balance/detail)
    _write_unidock_config(cfg, stage="fast")
    _write_unidock_config(cfg, stage="balance")
    _write_unidock_config(cfg, stage="detail")

    return 0


def _write_unidock_config(cfg: DockingPipelineConfig, *, stage: str) -> None:
    run_dir = cfg.run.work_dir
    stage_dir = run_dir / f"unidock_{stage}"
    ensure_dir(stage_dir)

    search_mode = cfg.unidock2.stages[stage].search_mode
    cx, cy, cz = cfg.inputs.docking_box.center
    sx, sy, sz = cfg.inputs.docking_box.size

    yaml_data = {
        "Required": {
            "receptor": str(cfg.inputs.receptor_pdb),
            # Ligand is overridden per array task via CLI "-l"; keep a placeholder.
            "ligand": str(run_dir / "inputs" / stage / "chunks" / "chunk0.sdf"),
            "center": [float(cx), float(cy), float(cz)],
        },
        "Settings": {
            "task": "screen",
            "search_mode": str(search_mode),
            "box_size": [float(sx), float(sy), float(sz)],
        },
        "Hardware": {
            # Slurm makes the allocated GPU visible as device 0 in most setups.
            "gpu_device_id": 0,
        },
        "Advanced": {
            "num_pose": int(cfg.unidock2.num_pose),
            "rmsd_limit": float(cfg.unidock2.rmsd_limit),
            "energy_range": float(cfg.unidock2.energy_range),
            "seed": int(cfg.run.random_seed),
        },
        "Preprocessing": {
            "temp_dir_name": str(cfg.unidock2.temp_dir),
        },
    }

    import yaml as _yaml

    (stage_dir / "config.yaml").write_text(_yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
