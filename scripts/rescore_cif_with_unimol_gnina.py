from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from docking_pipeline.pipeline_config import DockingPipelineConfig, load_config, normalize_config
from docking_pipeline.steps.common import ensure_dir, read_csv_rows, safe_float, write_csv


NUM_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\(\d+\))?$")


@dataclass(frozen=True)
class AtomSite:
    label: str
    element: str
    fract_x: float
    fract_y: float
    fract_z: float


@dataclass(frozen=True)
class BondSite:
    atom1: str
    atom2: str


@dataclass(frozen=True)
class Cell:
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float


def _load_cfg(config_path: Path) -> DockingPipelineConfig:
    cfg = load_config(config_path)
    return normalize_config(cfg, config_path=config_path)


def _parse_cif_number(text: str) -> float:
    s = text.strip()
    match = NUM_RE.match(s)
    if not match:
        raise ValueError(f"Cannot parse CIF number: {text!r}")
    return float(match.group(1))


def _tokenize_line(line: str) -> list[str]:
    return shlex.split(line, posix=False)


def _read_cif_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _extract_scalar(lines: list[str], key: str) -> str:
    for line in lines:
        if not line.startswith(key):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed CIF scalar line for {key}: {line!r}")
        return parts[1].strip()
    raise KeyError(key)


def _extract_loop_rows(lines: list[str], expected_header: str) -> tuple[list[str], list[list[str]]]:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "loop_":
            idx += 1
            continue

        idx += 1
        headers: list[str] = []
        while idx < len(lines) and lines[idx].startswith("_"):
            headers.append(lines[idx].strip())
            idx += 1

        if not headers or expected_header not in headers:
            continue

        rows: list[list[str]] = []
        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()
            if not stripped:
                idx += 1
                break
            if stripped == "loop_" or raw.startswith("_") or raw.startswith("data_"):
                break
            if stripped.startswith(";"):
                raise ValueError(f"Unexpected multi-line CIF value inside target loop: {expected_header}")
            rows.append(_tokenize_line(raw))
            idx += 1
        return headers, rows

    raise KeyError(expected_header)


def _load_cif_atoms_and_bonds(cif_path: Path) -> tuple[Cell, list[AtomSite], list[BondSite]]:
    lines = _read_cif_lines(cif_path)
    cell = Cell(
        a=_parse_cif_number(_extract_scalar(lines, "_cell_length_a")),
        b=_parse_cif_number(_extract_scalar(lines, "_cell_length_b")),
        c=_parse_cif_number(_extract_scalar(lines, "_cell_length_c")),
        alpha_deg=_parse_cif_number(_extract_scalar(lines, "_cell_angle_alpha")),
        beta_deg=_parse_cif_number(_extract_scalar(lines, "_cell_angle_beta")),
        gamma_deg=_parse_cif_number(_extract_scalar(lines, "_cell_angle_gamma")),
    )

    atom_headers, atom_rows = _extract_loop_rows(lines, "_atom_site_label")
    atom_idx = {name: i for i, name in enumerate(atom_headers)}
    atoms = [
        AtomSite(
            label=row[atom_idx["_atom_site_label"]].strip(),
            element=row[atom_idx["_atom_site_type_symbol"]].strip(),
            fract_x=_parse_cif_number(row[atom_idx["_atom_site_fract_x"]]),
            fract_y=_parse_cif_number(row[atom_idx["_atom_site_fract_y"]]),
            fract_z=_parse_cif_number(row[atom_idx["_atom_site_fract_z"]]),
        )
        for row in atom_rows
    ]

    bond_headers, bond_rows = _extract_loop_rows(lines, "_geom_bond_atom_site_label_1")
    bond_idx = {name: i for i, name in enumerate(bond_headers)}
    bonds = [
        BondSite(
            atom1=row[bond_idx["_geom_bond_atom_site_label_1"]].strip(),
            atom2=row[bond_idx["_geom_bond_atom_site_label_2"]].strip(),
        )
        for row in bond_rows
    ]
    return cell, atoms, bonds


def _largest_component_labels(atoms: list[AtomSite], bonds: list[BondSite]) -> set[str]:
    labels = {atom.label for atom in atoms}
    graph: dict[str, set[str]] = {label: set() for label in labels}
    for bond in bonds:
        if bond.atom1 not in graph or bond.atom2 not in graph:
            continue
        graph[bond.atom1].add(bond.atom2)
        graph[bond.atom2].add(bond.atom1)

    visited: set[str] = set()
    components: list[set[str]] = []
    for label in labels:
        if label in visited:
            continue
        queue: deque[str] = deque([label])
        comp: set[str] = set()
        visited.add(label)
        while queue:
            cur = queue.popleft()
            comp.add(cur)
            for nxt in graph[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        components.append(comp)

    atoms_by_label = {atom.label: atom for atom in atoms}
    components.sort(
        key=lambda comp: (
            sum(1 for label in comp if atoms_by_label[label].element.upper() != "H"),
            len(comp),
        ),
        reverse=True,
    )
    return components[0] if components else set()


def _fractional_to_cartesian(cell: Cell, x: float, y: float, z: float) -> tuple[float, float, float]:
    alpha = math.radians(cell.alpha_deg)
    beta = math.radians(cell.beta_deg)
    gamma = math.radians(cell.gamma_deg)

    ax, ay, az = cell.a, 0.0, 0.0
    bx, by, bz = cell.b * math.cos(gamma), cell.b * math.sin(gamma), 0.0
    cx = cell.c * math.cos(beta)
    cy = cell.c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz_sq = cell.c * cell.c - cx * cx - cy * cy
    if cz_sq <= 0:
        raise ValueError(f"Invalid unit cell derived cz^2={cz_sq}")
    cz = math.sqrt(cz_sq)

    cart_x = x * ax + y * bx + z * cx
    cart_y = x * ay + y * by + z * cy
    cart_z = x * az + y * bz + z * cz
    return cart_x, cart_y, cart_z


def _build_rdkit_mol_from_cif(
    cif_path: Path,
    *,
    ligand_id: str,
    charge: int,
) -> tuple[Chem.Mol, str, str]:
    cell, atoms_all, bonds_all = _load_cif_atoms_and_bonds(cif_path)
    keep_labels = _largest_component_labels(atoms_all, bonds_all)
    atoms = [atom for atom in atoms_all if atom.label in keep_labels]
    bonds = [bond for bond in bonds_all if bond.atom1 in keep_labels and bond.atom2 in keep_labels]
    if not atoms:
        raise RuntimeError(f"No atoms selected from {cif_path}")

    rw = Chem.RWMol()
    label_to_idx: dict[str, int] = {}
    conf = Chem.Conformer(len(atoms))
    for idx, atom_site in enumerate(atoms):
        atom = Chem.Atom(atom_site.element)
        atom.SetProp("cif_label", atom_site.label)
        label_to_idx[atom_site.label] = idx
        rw_idx = rw.AddAtom(atom)
        if rw_idx != idx:
            raise RuntimeError("Atom index mismatch while building RDKit molecule")
        x, y, z = _fractional_to_cartesian(cell, atom_site.fract_x, atom_site.fract_y, atom_site.fract_z)
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))

    for bond in bonds:
        a1 = label_to_idx[bond.atom1]
        a2 = label_to_idx[bond.atom2]
        rw.AddBond(a1, a2, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    conf.Set3D(True)
    mol.AddConformer(conf)

    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
    except Exception as exc:
        raise RuntimeError(f"RDKit DetermineBondOrders failed: {exc}") from exc

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise RuntimeError(f"RDKit sanitize failed after CIF conversion: {exc}") from exc

    mol.SetProp("_Name", ligand_id)
    mol.SetProp("ligand_id", ligand_id)
    mol.SetProp("source_file", str(cif_path))

    explicit_h_smiles = Chem.MolToSmiles(Chem.Mol(mol), canonical=True)
    key_h_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
    return mol, explicit_h_smiles, key_h_smiles


def _write_sdf(path: Path, mol: Chem.Mol) -> None:
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
            "Convert a small-molecule CCDC CIF into an SDF using its crystal coordinates, "
            "then rescore it with Uni-Mol and gnina against the pocket defined in a pipeline config."
        )
    )
    ap.add_argument("--config", type=Path, required=True, help="RNaseL pipeline config or run.yaml.")
    ap.add_argument("--input-cif", type=Path, default=Path("dirty-task/compare_with_disney/1912054.cif"))
    ap.add_argument("--output-dir", type=Path, default=Path("dirty-task/compare_with_disney/unimol_gnina_rescore_1912054"))
    ap.add_argument("--ligand-id", default="1912054")
    ap.add_argument("--charge", type=int, default=0, help="Formal charge passed to RDKit bond-order perception.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    input_cif = args.input_cif.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")

    mol, smiles_explicit_h, smiles_key_h = _build_rdkit_mol_from_cif(
        input_cif,
        ligand_id=args.ligand_id,
        charge=args.charge,
    )

    ligand_sdf = output_dir / "input_ligands" / f"{args.ligand_id}.sdf"
    if args.force or not ligand_sdf.exists():
        ensure_dir(ligand_sdf.parent)
        _write_sdf(ligand_sdf, mol)

    prep_rows = [
        {
            "ligand_id": args.ligand_id,
            "input_cif": str(input_cif),
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
        fieldnames=["ligand_id", "input_cif", "input_sdf", "status", "smiles_explicit_h", "smiles_key_h", "error"],
    )

    docking_grid_json = _write_docking_grid_json(cfg, out_dir=output_dir)
    chunk_dir = output_dir / "unimol" / "chunks" / "chunk_0"
    ensure_dir(chunk_dir)
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

    if args.force:
        unimol_out = out_sdf_dir / f"{args.ligand_id}.sdf"
        if unimol_out.exists():
            unimol_out.unlink()

    unimol_cmd = _build_unimol_cmd(cfg, batch_csv=batch_csv, out_dir=out_sdf_dir)
    unimol_proc = subprocess.run(unimol_cmd, text=True, capture_output=True)
    _write_run_log(output_dir / "logs" / "unimol.log", cmd=unimol_cmd, proc=unimol_proc)

    unimol_out = out_sdf_dir / f"{args.ligand_id}.sdf"
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
        input_sdf = gnina_dir / "input.sdf"
        output_sdf = gnina_dir / "gnina_out.sdf"
        block = _ensure_ligand_id_props(_first_mol_block_text(unimol_out), ligand_id=args.ligand_id)
        input_sdf.write_text(block if block.rstrip().endswith("$$$$") else block + "$$$$\n", encoding="utf-8")
        if args.force and output_sdf.exists():
            output_sdf.unlink()

        gnina_cmd = _build_gnina_cmd(cfg, input_sdf=input_sdf, output_sdf=output_sdf)
        gnina_proc = subprocess.run(gnina_cmd, text=True, capture_output=True)
        _write_run_log(output_dir / "logs" / "gnina.log", cmd=gnina_cmd, proc=gnina_proc)

        parsed = _parse_gnina_out_sdf(output_sdf) if gnina_proc.returncode == 0 else {}
        gnina_row = {
            "ligand_id": args.ligand_id,
            "status": "ok" if parsed else "failed",
            "minimizedAffinity": parsed.get("minimizedAffinity"),
            "CNNscore": parsed.get("CNNscore"),
            "CNNaffinity": parsed.get("CNNaffinity"),
            "gnina_out_sdf": str(output_sdf) if parsed else "",
            "error": "" if parsed else (_tail_text(gnina_proc.stderr) or _tail_text(gnina_proc.stdout) or f"gnina returncode={gnina_proc.returncode}"),
        }

    write_csv(
        gnina_summary_path,
        [gnina_row],
        fieldnames=["ligand_id", "status", "minimizedAffinity", "CNNscore", "CNNaffinity", "gnina_out_sdf", "error"],
    )

    final_row = {
        "ligand_id": args.ligand_id,
        "input_cif": str(input_cif),
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
    final_csv = output_dir / "final" / "rescored_from_cif.csv"
    write_csv(
        final_csv,
        [final_row],
        fieldnames=[
            "ligand_id",
            "input_cif",
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

    print(f"Input CIF: {input_cif}")
    print(f"Output dir: {output_dir}")
    print(f"Input SDF: {ligand_sdf}")
    print(f"Final CSV: {final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
