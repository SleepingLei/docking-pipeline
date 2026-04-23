from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, safe_float, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Run gnina rescoring/minimize for one Uni-Mol chunk and summarize.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--chunk-id", type=int, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir

    chunk_dir = run_dir / "unimol" / "chunks" / f"chunk_{args.chunk_id}"
    in_dir = chunk_dir / "out_sdf"
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    gnina_dir = run_dir / "gnina" / "chunks" / f"chunk_{args.chunk_id}"
    ensure_dir(gnina_dir)

    in_sdf = gnina_dir / "input.sdf"
    out_sdf = gnina_dir / "gnina_out.sdf"
    out_csv = gnina_dir / "summary.csv"

    # Merge unimol output SDFs into one multi-molecule input for gnina.
    w = Chem.SDWriter(str(in_sdf))
    count = 0
    for p in sorted(in_dir.glob("*.sdf")):
        sup = Chem.SDMolSupplier(str(p), removeHs=False)
        for mol in sup:
            if mol is None:
                continue
            lid = mol.GetProp("ligand_id") if mol.HasProp("ligand_id") else p.stem
            mol.SetProp("_Name", lid)
            mol.SetProp("ligand_id", lid)
            w.write(mol)
            count += 1
            break
    w.close()
    if count == 0:
        raise RuntimeError(f"No Uni-Mol output SDFs found under {in_dir}")

    cx, cy, cz = cfg.inputs.docking_box.center
    sx, sy, sz = cfg.inputs.docking_box.size

    cmd = [
        str((cfg.run.project_dir / "bin" / cfg.gnina.executable).resolve())
        if (cfg.run.project_dir / "bin" / cfg.gnina.executable).exists()
        else cfg.gnina.executable,
        "-r",
        str(cfg.inputs.receptor_pdb),
        "-l",
        str(in_sdf),
        "--center_x",
        str(cx),
        "--center_y",
        str(cy),
        "--center_z",
        str(cz),
        "--size_x",
        str(sx),
        "--size_y",
        str(sy),
        "--size_z",
        str(sz),
        "--cnn",
        str(cfg.gnina.cnn),
        "--cpu",
        str(cfg.slurm.defaults.cpus_per_task),
    ]
    if cfg.gnina.mode == "score_only":
        cmd.append("--score_only")
    else:
        cmd.append("--minimize")
        cmd.extend(["-o", str(out_sdf)])

    subprocess.run(cmd, check=True)

    # Summarize output SDF properties (minimize path).
    rows: list[dict[str, object]] = []
    if out_sdf.exists():
        sup = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
        for mol in sup:
            if mol is None:
                continue
            lid = ""
            if mol.HasProp("ligand_id"):
                lid = mol.GetProp("ligand_id")
            elif mol.HasProp("_Name"):
                lid = mol.GetProp("_Name")
            if not lid:
                continue
            rows.append(
                {
                    "ligand_id": lid,
                    "minimizedAffinity": safe_float(mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else None),
                    "CNNscore": safe_float(mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else None),
                    "CNNaffinity": safe_float(mol.GetProp("CNNaffinity") if mol.HasProp("CNNaffinity") else None),
                }
            )
    write_csv(out_csv, rows, fieldnames=["ligand_id", "minimizedAffinity", "CNNscore", "CNNaffinity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

