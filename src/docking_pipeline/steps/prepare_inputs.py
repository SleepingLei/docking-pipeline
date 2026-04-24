from __future__ import annotations

import argparse
import math
from pathlib import Path

from rdkit import Chem

from docking_pipeline.pipeline_config import DockingPipelineConfig
from docking_pipeline.steps.common import ensure_dir, load_run_cfg, write_csv


def _iter_sdf_blocks(sdf_path: Path):
    """
    Stream SDF records as raw text blocks (ending with '$$$$').

    We intentionally avoid depending on RDKit's SDMolSupplier for input parsing because
    some vendor SDFs trigger 'Bad input file' / sanitization issues on certain builds.
    Uni-Dock2 can still often consume these records.
    """
    buf: list[str] = []
    idx = 0
    with sdf_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line)
            if line.strip() == "$$$$":
                yield idx, "".join(buf)
                buf = []
                idx += 1
    if buf:
        # Last record without terminator; still emit for debugging/manifest and to avoid silent drop.
        yield idx, "".join(buf)


def _ensure_ligand_id_props(sdf_block: str, *, ligand_id: str) -> str:
    """
    Ensure the record has:
      1) first line (mol name) set to ligand_id
      2) an SD property 'ligand_id' (so Uni-Dock2 outputs preserve it)
      3) a '$$$$' terminator
    """
    lines = sdf_block.splitlines()
    if not lines:
        return f"{ligand_id}\n\n\n$$$$\n"
    lines[0] = ligand_id
    has_prop = any(ln.strip() == ">  <ligand_id>" for ln in lines)

    # Ensure terminator exists and find insertion point.
    end_i = None
    for i, ln in enumerate(lines):
        if ln.strip() == "$$$$":
            end_i = i
            break
    if end_i is None:
        end_i = len(lines)
        lines.append("$$$$")

    if not has_prop:
        insert = [">  <ligand_id>", ligand_id, ""]
        lines = lines[:end_i] + insert + lines[end_i:]
    return "\n".join(lines) + "\n"


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
    chunk_fh = None

    def ensure_writer():
        nonlocal next_chunk_id, current_chunk_id, in_chunk, chunk_fh
        if chunk_fh is None or in_chunk >= chunk_size:
            if chunk_fh is not None:
                chunk_fh.close()
            current_chunk_id = next_chunk_id
            out_sdf_path = fast_chunks_dir / f"chunk{current_chunk_id}.sdf"
            chunk_fh = out_sdf_path.open("w", encoding="utf-8")
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

        for src_idx, block in _iter_sdf_blocks(sdf_path):
            ligand_id = f"LIG_{ligand_counter:012d}"
            ligand_counter += 1

            if not block.strip():
                manifest_rows.append(
                    {
                        "ligand_id": ligand_id,
                        "source_sdf": str(sdf_path),
                        "source_mol_idx": src_idx,
                        "status": "empty_record",
                    }
                )
                continue

            # Keep original name for manifest (first line of record).
            orig_name = block.splitlines()[0].strip() if block.splitlines() else ""

            # Attach stable ID so Uni-Dock2 output SDF preserves it.
            out_block = _ensure_ligand_id_props(block, ligand_id=ligand_id)

            smiles = ""
            try:
                mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
                if mol is not None:
                    # Try sanitize for canonical SMILES, but never fail the pipeline on it.
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass
                    try:
                        smiles = Chem.MolToSmiles(mol, canonical=True)
                    except Exception:
                        smiles = ""
            except Exception:
                smiles = ""

            ensure_writer()
            assert chunk_fh is not None
            chunk_fh.write(out_block)
            in_chunk += 1

            manifest_rows.append(
                {
                    "ligand_id": ligand_id,
                    "source_sdf": str(sdf_path),
                    "source_mol_idx": src_idx,
                    "prepared_chunk": int(current_chunk_id),
                    "smiles": smiles,
                    "name": orig_name,
                    "status": "ok",
                }
            )

    if chunk_fh is None:
        raise RuntimeError("No valid ligands were written to any chunk; check input ligands SDF.")
    chunk_fh.close()

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
