from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, safe_float, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize a Uni-Dock2 output SDF chunk to per-ligand best scores.")
    ap.add_argument("--in-sdf", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-best-sdf", type=Path, default=None)
    args = ap.parse_args()

    ensure_dir(args.out_csv.parent)
    if args.out_best_sdf is not None:
        ensure_dir(args.out_best_sdf.parent)

    supplier = Chem.SDMolSupplier(str(args.in_sdf), removeHs=False)
    best: dict[str, dict[str, object]] = {}
    best_mol: dict[str, Chem.Mol] = {}

    for mol in supplier:
        if mol is None:
            continue
        ligand_id = mol.GetProp("ligand_id") if mol.HasProp("ligand_id") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
        if not ligand_id:
            continue
        score = safe_float(mol.GetProp("vina_binding_free_energy") if mol.HasProp("vina_binding_free_energy") else None)
        if score is None:
            continue
        cur = best.get(ligand_id)
        if cur is None or score < float(cur["vina_binding_free_energy"]):
            best[ligand_id] = {"ligand_id": ligand_id, "vina_binding_free_energy": score}
            best_mol[ligand_id] = mol

    rows = [best[k] for k in sorted(best.keys())]
    write_csv(args.out_csv, rows, fieldnames=["ligand_id", "vina_binding_free_energy"])

    if args.out_best_sdf is not None:
        w = Chem.SDWriter(str(args.out_best_sdf))
        for ligand_id in sorted(best_mol.keys()):
            m = best_mol[ligand_id]
            # Ensure name is stable for downstream gnina parsing if needed.
            m.SetProp("_Name", ligand_id)
            m.SetProp("ligand_id", ligand_id)
            w.write(m)
        w.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

