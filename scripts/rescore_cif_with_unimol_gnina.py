from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from rdkit import Chem

from docking_pipeline.pipeline_config import DockingPipelineConfig, load_config, normalize_config
from docking_pipeline.steps.common import ensure_dir, safe_float, write_csv


def _load_cfg(config_path: Path) -> DockingPipelineConfig:
    cfg = load_config(config_path)
    return normalize_config(cfg, config_path=config_path)


def _load_input_sdf(input_sdf: Path, *, ligand_id: str) -> tuple[Chem.Mol, str, str]:
    supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=False)
    mol = next((mol for mol in supplier if mol is not None), None)
    if mol is None:
        raise RuntimeError(f"Could not read any molecule from {input_sdf}")

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise RuntimeError(f"RDKit sanitize failed for {input_sdf}: {exc}") from exc

    mol.SetProp("_Name", ligand_id)
    mol.SetProp("ligand_id", ligand_id)
    mol.SetProp("source_file", str(input_sdf))

    smiles_explicit_h = Chem.MolToSmiles(Chem.Mol(mol), canonical=True)
    smiles_key_h = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
    return mol, smiles_explicit_h, smiles_key_h


def _write_sdf(path: Path, mol: Chem.Mol) -> None:
    ensure_dir(path.parent)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _write_docking_grid_json(cfg: DockingPipelineConfig, *, out_dir: Path) -> Path:
    cx, cy, cz = cfg.inputs.docking_box.center
    sx, sy, sz = cfg.inputs.docking_box.size
    grid = {
        "center_x": float(cx),
        "center_y": float(cy),
        "center_z": float(cz),
        "size_x": float(sx),
        "size_y": float(sy),
        "size_z": float(sz),
    }
    path = out_dir / "unimol" / "docking_grid.json"
    ensure_dir(path.parent)
    path.write_text(json.dumps(grid, indent=2), encoding="utf-8")
    return path


def _write_run_log(path: Path, *, cmd: list[str], proc: subprocess.CompletedProcess[str]) -> None:
    ensure_dir(path.parent)
    text = [
        "$ " + " ".join(shlex.quote(x) for x in cmd),
        "",
        "STDOUT",
        proc.stdout or "",
        "",
        "STDERR",
        proc.stderr or "",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _tail_text(text: str, *, max_lines: int = 30) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _build_unimol_cmd(cfg: DockingPipelineConfig, *, batch_csv: Path, out_dir: Path) -> list[str]:
    demo_py = Path(cfg.unimol.repo_dir) / "interface" / "demo.py"
    if not demo_py.exists():
        raise FileNotFoundError(demo_py)

    cmd: list[str] = [
        "conda",
        "run",
        "-n",
        cfg.unimol.env_name,
        "python",
        str(demo_py),
        "--mode",
        "batch_one2many",
        "--batch-size",
        str(cfg.unimol.batch_size),
        "--nthreads",
        str(cfg.slurm.defaults.cpus_per_task),
        "--conf-size",
        str(cfg.unimol.conf_size),
    ]
    if cfg.unimol.cluster_conformers:
        cmd.append("--cluster")
    cmd.extend(
        [
            "--input-protein",
            str(cfg.inputs.receptor_pdb),
            "--input-batch-file",
            str(batch_csv),
            "--output-ligand-dir",
            str(out_dir),
        ]
    )
    if cfg.unimol.use_current_ligand_conf:
        cmd.append("--use_current_ligand_conf")
    if cfg.unimol.steric_clash_fix:
        cmd.append("--steric-clash-fix")
    cmd.extend(["--model-dir", str(cfg.unimol.model_path)])
    return cmd


def _first_mol_block_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    if not lines:
        raise ValueError(f"Empty SDF: {path}")
    try:
        end_i = next(i for i, line in enumerate(lines) if line.strip() == "$$$$")
        return "\n".join(lines[: end_i + 1]) + "\n"
    except StopIteration:
        return "\n".join(lines) + "\n$$$$\n"


def _ensure_ligand_id_props(sdf_block: str, *, ligand_id: str) -> str:
    lines = sdf_block.splitlines()
    if not lines:
        return sdf_block
    lines[0] = ligand_id
    has_prop = any(line.strip() == ">  <ligand_id>" for line in lines)
    if has_prop:
        return "\n".join(lines) + "\n"
    try:
        end_i = next(i for i, line in enumerate(lines) if line.strip() == "$$$$")
    except StopIteration:
        end_i = len(lines)
    insert = [">  <ligand_id>", ligand_id, ""]
    return "\n".join(lines[:end_i] + insert + lines[end_i:]) + "\n"


def _build_gnina_cmd(cfg: DockingPipelineConfig, *, input_sdf: Path, output_sdf: Path) -> list[str]:
    local_bin = (cfg.run.project_dir / "bin" / cfg.gnina.executable).resolve()
    executable = str(local_bin) if local_bin.exists() else cfg.gnina.executable
    cx, cy, cz = cfg.inputs.docking_box.center
    sx, sy, sz = cfg.inputs.docking_box.size

    cmd = [
        executable,
        "-r",
        str(cfg.inputs.receptor_pdb),
        "-l",
        str(input_sdf),
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
        raise ValueError(f"Unknown gnina.mode={cfg.gnina.mode!r}")
    cmd.extend(["-o", str(output_sdf)])
    return cmd


def _parse_gnina_out_sdf(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is None:
            continue
        ligand_id = ""
        if mol.HasProp("ligand_id"):
            ligand_id = mol.GetProp("ligand_id")
        elif mol.HasProp("_Name"):
            ligand_id = mol.GetProp("_Name")
        if not ligand_id:
            continue
        return {
            "ligand_id": ligand_id,
            "minimizedAffinity": safe_float(mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else None),
            "CNNscore": safe_float(mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else None),
            "CNNaffinity": safe_float(mol.GetProp("CNNaffinity") if mol.HasProp("CNNaffinity") else None),
        }
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Read a single ligand SDF, then rescore it with Uni-Mol and gnina "
            "against the pocket defined in a pipeline config."
        )
    )
    ap.add_argument("--config", type=Path, required=True, help="RNaseL pipeline config or run.yaml.")
    ap.add_argument("--input-sdf", type=Path, default=Path("dirty-task/compare_with_disney/1912054.sdf"))
    ap.add_argument("--output-dir", type=Path, default=Path("dirty-task/compare_with_disney/unimol_gnina_rescore_1912054"))
    ap.add_argument("--ligand-id", default="1912054")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    input_sdf = args.input_sdf.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")

    mol, smiles_explicit_h, smiles_key_h = _load_input_sdf(input_sdf, ligand_id=args.ligand_id)

    ligand_sdf = output_dir / "input_ligands" / f"{args.ligand_id}.sdf"
    if args.force or not ligand_sdf.exists():
        _write_sdf(ligand_sdf, mol)

    prep_rows = [
        {
            "ligand_id": args.ligand_id,
            "input_sdf_source": str(input_sdf),
            "input_sdf": str(ligand_sdf),
            "status": "ok",
            "smiles_explicit_h": smiles_explicit_h,
            "smiles_key_h": smiles_key_h,
            "error": "",
        }
    ]
    write_csv(
        output_dir / "prep_summary.csv",
        prep_rows,
        fieldnames=["ligand_id", "input_sdf_source", "input_sdf", "status", "smiles_explicit_h", "smiles_key_h", "error"],
    )

    docking_grid_json = _write_docking_grid_json(cfg, out_dir=output_dir)
    chunk_dir = output_dir / "unimol" / "chunks" / "chunk_0"
    out_sdf_dir = chunk_dir / "out_sdf"
    ensure_dir(out_sdf_dir)
    batch_csv = chunk_dir / "batch.csv"
    write_csv(
        batch_csv,
        [
            {
                "input_ligand": str(ligand_sdf),
                "input_docking_grid": str(docking_grid_json),
                "output_ligand_name": args.ligand_id,
            }
        ],
        fieldnames=["input_ligand", "input_docking_grid", "output_ligand_name"],
    )

    unimol_out = out_sdf_dir / f"{args.ligand_id}.sdf"
    if args.force and unimol_out.exists():
        unimol_out.unlink()

    unimol_cmd = _build_unimol_cmd(cfg, batch_csv=batch_csv, out_dir=out_sdf_dir)
    unimol_proc = subprocess.run(unimol_cmd, text=True, capture_output=True)
    _write_run_log(output_dir / "logs" / "unimol.log", cmd=unimol_cmd, proc=unimol_proc)

    unimol_status = "ok" if unimol_proc.returncode == 0 and unimol_out.exists() else "failed"
    unimol_error = ""
    if unimol_status != "ok":
        unimol_error = _tail_text(unimol_proc.stderr) or _tail_text(unimol_proc.stdout) or f"unimol returncode={unimol_proc.returncode}"

    write_csv(
        chunk_dir / "unimol_summary.csv",
        [
            {
                "ligand_id": args.ligand_id,
                "status": unimol_status,
                "output_sdf": str(unimol_out) if unimol_out.exists() else "",
                "error": unimol_error,
            }
        ],
        fieldnames=["ligand_id", "status", "output_sdf", "error"],
    )

    gnina_dir = output_dir / "gnina" / "chunks" / "chunk_0"
    ensure_dir(gnina_dir)
    gnina_summary_path = gnina_dir / "summary.csv"
    gnina_row: dict[str, object]
    if unimol_status != "ok":
        gnina_row = {
            "ligand_id": args.ligand_id,
            "status": "skipped",
            "minimizedAffinity": None,
            "CNNscore": None,
            "CNNaffinity": None,
            "gnina_out_sdf": "",
            "error": "unimol_failed",
        }
    else:
        gnina_input_sdf = gnina_dir / "input.sdf"
        gnina_output_sdf = gnina_dir / "gnina_out.sdf"
        block = _ensure_ligand_id_props(_first_mol_block_text(unimol_out), ligand_id=args.ligand_id)
        gnina_input_sdf.write_text(block if block.rstrip().endswith("$$$$") else block + "$$$$\n", encoding="utf-8")
        if args.force and gnina_output_sdf.exists():
            gnina_output_sdf.unlink()

        gnina_cmd = _build_gnina_cmd(cfg, input_sdf=gnina_input_sdf, output_sdf=gnina_output_sdf)
        gnina_proc = subprocess.run(gnina_cmd, text=True, capture_output=True)
        _write_run_log(output_dir / "logs" / "gnina.log", cmd=gnina_cmd, proc=gnina_proc)

        parsed = _parse_gnina_out_sdf(gnina_output_sdf) if gnina_proc.returncode == 0 else {}
        gnina_row = {
            "ligand_id": args.ligand_id,
            "status": "ok" if parsed else "failed",
            "minimizedAffinity": parsed.get("minimizedAffinity"),
            "CNNscore": parsed.get("CNNscore"),
            "CNNaffinity": parsed.get("CNNaffinity"),
            "gnina_out_sdf": str(gnina_output_sdf) if parsed else "",
            "error": "" if parsed else (_tail_text(gnina_proc.stderr) or _tail_text(gnina_proc.stdout) or f"gnina returncode={gnina_proc.returncode}"),
        }

    write_csv(
        gnina_summary_path,
        [gnina_row],
        fieldnames=["ligand_id", "status", "minimizedAffinity", "CNNscore", "CNNaffinity", "gnina_out_sdf", "error"],
    )

    final_row = {
        "ligand_id": args.ligand_id,
        "input_sdf_source": str(input_sdf),
        "input_sdf": str(ligand_sdf),
        "smiles_explicit_h": smiles_explicit_h,
        "smiles_key_h": smiles_key_h,
        "unimol_status": unimol_status,
        "unimol_output_sdf": str(unimol_out) if unimol_out.exists() else "",
        "unimol_error": unimol_error,
        "gnina_status": gnina_row.get("status", ""),
        "gnina_output_sdf": gnina_row.get("gnina_out_sdf", ""),
        "gnina_error": gnina_row.get("error", ""),
        "minimizedAffinity": gnina_row.get("minimizedAffinity"),
        "CNNscore": gnina_row.get("CNNscore"),
        "CNNaffinity": gnina_row.get("CNNaffinity"),
    }
    final_csv = output_dir / "final" / "rescored_from_sdf.csv"
    write_csv(
        final_csv,
        [final_row],
        fieldnames=[
            "ligand_id",
            "input_sdf_source",
            "input_sdf",
            "smiles_explicit_h",
            "smiles_key_h",
            "unimol_status",
            "unimol_output_sdf",
            "unimol_error",
            "gnina_status",
            "gnina_output_sdf",
            "gnina_error",
            "minimizedAffinity",
            "CNNscore",
            "CNNaffinity",
        ],
    )

    print(f"Input SDF: {input_sdf}")
    print(f"Output dir: {output_dir}")
    print(f"Normalized input SDF: {ligand_sdf}")
    print(f"Final CSV: {final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
