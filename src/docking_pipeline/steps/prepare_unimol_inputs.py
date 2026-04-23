from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, read_ligand_id_list, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare Uni-Mol Docking V2 batch_one2many inputs.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--ligand-ids", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir

    unimol_dir = run_dir / "unimol"
    chunks_dir = unimol_dir / "chunks"
    lig_dir = unimol_dir / "input_ligands"
    ensure_dir(chunks_dir)
    ensure_dir(lig_dir)

    # docking_grid.json for Uni-Mol
    cx, cy, cz = cfg.inputs.docking_box.center
    sx, sy, sz = cfg.inputs.docking_box.size
    docking_grid = {
        "center_x": float(cx),
        "center_y": float(cy),
        "center_z": float(cz),
        "size_x": float(sx),
        "size_y": float(sy),
        "size_z": float(sz),
    }
    grid_path = unimol_dir / "docking_grid.json"
    grid_path.write_text(json.dumps(docking_grid, indent=2), encoding="utf-8")

    selected = set(read_ligand_id_list(args.ligand_ids))
    if not selected:
        raise RuntimeError("Empty ligand id list for Uni-Mol inputs.")

    # Source conformations:
    # - If unimol.use_current_ligand_conf: use unidock_detail/best_poses/*.sdf
    # - Else: use inputs/detail/chunks/*.sdf
    if cfg.unimol.use_current_ligand_conf:
        source_glob = run_dir / "unidock_detail" / "best_poses"
        sources = sorted(source_glob.glob("best_*.sdf"))
    else:
        source_glob = run_dir / "inputs" / "detail" / "chunks"
        sources = sorted(source_glob.glob("chunk*.sdf"))

    found: set[str] = set()
    for sdf in sources:
        sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
        for mol in sup:
            if mol is None:
                continue
            lid = mol.GetProp("ligand_id") if mol.HasProp("ligand_id") else ""
            if lid not in selected:
                continue
            out_path = lig_dir / f"{lid}.sdf"
            w = Chem.SDWriter(str(out_path))
            mol.SetProp("_Name", lid)
            mol.SetProp("ligand_id", lid)
            w.write(mol)
            w.close()
            found.add(lid)

    missing = sorted(selected - found)
    if missing:
        (unimol_dir / "missing_ligands.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise RuntimeError(f"Uni-Mol input prep missing {len(missing)} ligands; see {unimol_dir}/missing_ligands.txt")

    # Chunk for Uni-Mol jobs
    lids = sorted(found)
    chunk_size = cfg.unimol.chunk_size
    n_chunks = math.ceil(len(lids) / chunk_size)
    for chunk_id in range(n_chunks):
        chunk_lids = lids[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        chunk_dir = chunks_dir / f"chunk_{chunk_id}"
        ensure_dir(chunk_dir)
        out_sdf_dir = chunk_dir / "out_sdf"
        ensure_dir(out_sdf_dir)
        rows: list[dict[str, object]] = []
        for lid in chunk_lids:
            rows.append(
                {
                    "input_ligand": str(lig_dir / f"{lid}.sdf"),
                    "input_docking_grid": str(grid_path),
                    "output_ligand_name": lid,
                }
            )
        write_csv(chunk_dir / "batch.csv", rows, fieldnames=["input_ligand", "input_docking_grid", "output_ligand_name"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

