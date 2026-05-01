from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, safe_float, write_csv


def _first_mol_block_text(path: Path) -> str:
    """
    Read the first molecule block from an SDF file as text (up to and including the first '$$$$').
    This avoids RDKit sanitization/KEKULIZE issues when preparing gnina inputs.
    """
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    if not lines:
        raise ValueError(f"Empty SDF: {path}")
    # Uni-Mol occasionally writes single-molecule SDFs without the "$$$$" terminator.
    # In that case, treat the entire file as one record and append "$$$$" ourselves.
    try:
        end_i = next(i for i, ln in enumerate(lines) if ln.strip() == "$$$$")
        return "\n".join(lines[: end_i + 1]) + "\n"
    except StopIteration:
        # Always ensure "$$$$" starts on a new line.
        return "\n".join(lines) + "\n$$$$\n"


def _ensure_ligand_id_props(sdf_block: str, *, ligand_id: str) -> str:
    """
    Ensure:
      1) first line (mol name) is ligand_id
      2) SD property 'ligand_id' exists (so downstream parsing is stable)
    """
    lines = sdf_block.splitlines()
    if not lines:
        return sdf_block
    lines[0] = ligand_id
    has_prop = any(ln.strip() == ">  <ligand_id>" for ln in lines)
    if has_prop:
        return "\n".join(lines) + "\n"

    try:
        end_i = next(i for i, ln in enumerate(lines) if ln.strip() == "$$$$")
    except StopIteration:
        end_i = len(lines)
    insert = [">  <ligand_id>", ligand_id, ""]
    lines = lines[:end_i] + insert + lines[end_i:]
    return "\n".join(lines) + "\n"


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
        print(f"[warn] skip gnina chunk {args.chunk_id}: missing unimol output dir {in_dir}")
        return 0

    gnina_dir = run_dir / "gnina" / "chunks" / f"chunk_{args.chunk_id}"
    ensure_dir(gnina_dir)

    in_sdf = gnina_dir / "input.sdf"
    out_sdf = gnina_dir / "gnina_out.sdf"
    out_csv = gnina_dir / "summary.csv"

    # Merge unimol output SDFs into one multi-molecule input for gnina.
    sdf_paths = sorted(in_dir.glob("*.sdf"))
    if not sdf_paths:
        print(f"[warn] skip gnina chunk {args.chunk_id}: no Uni-Mol output SDFs under {in_dir}")
        write_csv(out_csv, [], fieldnames=["ligand_id", "minimizedAffinity", "CNNscore", "CNNaffinity"])
        return 0

    count = 0
    with in_sdf.open("w", encoding="utf-8") as f:
        for p in sdf_paths:
            lid = p.stem
            try:
                block = _first_mol_block_text(p)
                block = _ensure_ligand_id_props(block, ligand_id=lid)
                f.write(block)
                if not block.rstrip().endswith("$$$$"):
                    f.write("$$$$\n")
                count += 1
            except Exception as e:
                # Don't fail the whole chunk on a single malformed SDF.
                print(f"[warn] skip unimol sdf due to parse error: {p} ({e})")
    if count == 0:
        print(f"[warn] skip gnina chunk {args.chunk_id}: all Uni-Mol SDFs were skipped under {in_dir}")
        write_csv(out_csv, [], fieldnames=["ligand_id", "minimizedAffinity", "CNNscore", "CNNaffinity"])
        return 0

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
        "--cpu",
        str(cfg.slurm.defaults.cpus_per_task),
    ]
    if cfg.gnina.cnn:
        cmd.extend(["--cnn", str(cfg.gnina.cnn)])
    if cfg.gnina.mode == "score_only":
        cmd.append("--score_only")
    elif cfg.gnina.mode == "minimize":
        cmd.append("--minimize")
    else:
        raise ValueError(f"Unknown gnina.mode={cfg.gnina.mode!r}; expected 'score_only' or 'minimize'")
    cmd.extend(["-o", str(out_sdf)])

    subprocess.run(cmd, check=True)

    # Summarize output SDF properties.
    rows: list[dict[str, object]] = []
    if out_sdf.exists():
        sup = Chem.SDMolSupplier(str(out_sdf), removeHs=False, sanitize=False)
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
