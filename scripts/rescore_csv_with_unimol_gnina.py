from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from docking_pipeline.pipeline_config import DockingPipelineConfig, load_config, normalize_config
from docking_pipeline.steps.common import ensure_dir, read_csv_rows, safe_float, write_csv


@dataclass(frozen=True)
class InputLigand:
    ligand_id: str
    source_mol_id: str
    smiles: str
    row_index: int
    row: dict[str, str]


@dataclass(frozen=True)
class BatchRow:
    input_ligand: str
    input_docking_grid: str
    output_ligand_name: str


def _load_cfg(config_path: Path) -> DockingPipelineConfig:
    cfg = load_config(config_path)
    return normalize_config(cfg, config_path=config_path)


def _sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "ligand"


def _read_input_csv(path: Path, *, mol_id_col: str, smiles_col: str) -> tuple[list[InputLigand], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if mol_id_col not in reader.fieldnames:
            raise ValueError(f"Missing mol id column {mol_id_col!r} in {path}")
        if smiles_col not in reader.fieldnames:
            raise ValueError(f"Missing SMILES column {smiles_col!r} in {path}")

        used_ids: set[str] = set()
        ligands: list[InputLigand] = []
        for row_index, row in enumerate(reader, start=1):
            source_mol_id = str(row.get(mol_id_col, "")).strip() or f"row_{row_index:04d}"
            smiles = str(row.get(smiles_col, "")).strip()
            base_id = _sanitize_id(source_mol_id)
            ligand_id = base_id
            suffix = 2
            while ligand_id in used_ids:
                ligand_id = f"{base_id}__{suffix}"
                suffix += 1
            used_ids.add(ligand_id)
            ligands.append(
                InputLigand(
                    ligand_id=ligand_id,
                    source_mol_id=source_mol_id,
                    smiles=smiles,
                    row_index=row_index,
                    row={str(k): str(v) for k, v in row.items()},
                )
            )
        return ligands, list(reader.fieldnames)


def _prepare_3d_sdf(
    ligand: InputLigand,
    *,
    out_path: Path,
    random_seed: int,
) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(ligand.smiles)
    if mol is None:
        raise ValueError("invalid_smiles")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True
    embed_code = AllChem.EmbedMolecule(mol, params)
    if embed_code != 0:
        raise RuntimeError(f"rdkit_embed_failed({embed_code})")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        AllChem.UFFOptimizeMolecule(mol)

    mol.SetProp("_Name", ligand.ligand_id)
    mol.SetProp("ligand_id", ligand.ligand_id)
    mol.SetProp("source_mol_id", ligand.source_mol_id)
    mol.SetProp("input_smiles", ligand.smiles)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return canonical_smiles, str(out_path)


def _prepare_input_ligands(
    ligands: list[InputLigand],
    *,
    out_dir: Path,
    random_seed: int,
    force: bool,
) -> list[dict[str, object]]:
    input_lig_dir = out_dir / "input_ligands"
    ensure_dir(input_lig_dir)

    prep_rows: list[dict[str, object]] = []
    for idx, ligand in enumerate(ligands, start=1):
        sdf_path = input_lig_dir / f"{ligand.ligand_id}.sdf"
        if sdf_path.exists() and not force:
            prep_rows.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "source_mol_id": ligand.source_mol_id,
                    "status": "ok_existing",
                    "input_sdf": str(sdf_path),
                    "canonical_smiles": "",
                    "error": "",
                }
            )
            continue

        try:
            canonical_smiles, sdf_out = _prepare_3d_sdf(
                ligand,
                out_path=sdf_path,
                random_seed=random_seed + idx,
            )
            prep_rows.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "source_mol_id": ligand.source_mol_id,
                    "status": "ok",
                    "input_sdf": sdf_out,
                    "canonical_smiles": canonical_smiles,
                    "error": "",
                }
            )
        except Exception as exc:
            prep_rows.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "source_mol_id": ligand.source_mol_id,
                    "status": "failed",
                    "input_sdf": "",
                    "canonical_smiles": "",
                    "error": str(exc),
                }
            )

    write_csv(
        out_dir / "prep_summary.csv",
        prep_rows,
        fieldnames=["ligand_id", "source_mol_id", "status", "input_sdf", "canonical_smiles", "error"],
    )
    return prep_rows


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


def _write_batch_csv(path: Path, rows: list[BatchRow]) -> None:
    write_csv(
        path,
        [
            {
                "input_ligand": row.input_ligand,
                "input_docking_grid": row.input_docking_grid,
                "output_ligand_name": row.output_ligand_name,
            }
            for row in rows
        ],
        fieldnames=["input_ligand", "input_docking_grid", "output_ligand_name"],
    )


def _chunk_prepared_ligands(
    prep_rows: list[dict[str, object]],
    *,
    out_dir: Path,
    docking_grid_json: Path,
    chunk_size: int,
) -> list[Path]:
    ok_rows = [row for row in prep_rows if str(row.get("status", "")).startswith("ok")]
    if not ok_rows:
        raise RuntimeError("No prepared ligands available for Uni-Mol.")

    chunks_root = out_dir / "unimol" / "chunks"
    ensure_dir(chunks_root)

    n_chunks = math.ceil(len(ok_rows) / chunk_size)
    chunk_dirs: list[Path] = []
    for chunk_id in range(n_chunks):
        chunk_dir = chunks_root / f"chunk_{chunk_id}"
        ensure_dir(chunk_dir)
        ensure_dir(chunk_dir / "out_sdf")
        chunk_rows = ok_rows[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        batch_rows = [
            BatchRow(
                input_ligand=str(row["input_sdf"]),
                input_docking_grid=str(docking_grid_json),
                output_ligand_name=str(row["ligand_id"]),
            )
            for row in chunk_rows
        ]
        _write_batch_csv(chunk_dir / "batch.csv", batch_rows)
        chunk_dirs.append(chunk_dir)
    return chunk_dirs


def _unimol_expected_out(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.sdf"


def _build_unimol_cmd(
    cfg: DockingPipelineConfig,
    *,
    batch_csv: Path,
    out_dir: Path,
) -> list[str]:
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


def _tail_text(text: str, *, max_lines: int = 30) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


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


def _read_batch_csv(path: Path) -> list[BatchRow]:
    rows: list[BatchRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                BatchRow(
                    input_ligand=str(row.get("input_ligand", "")).strip(),
                    input_docking_grid=str(row.get("input_docking_grid", "")).strip(),
                    output_ligand_name=str(row.get("output_ligand_name", "")).strip(),
                )
            )
    return rows


def _run_unimol_chunk(
    cfg: DockingPipelineConfig,
    *,
    chunk_dir: Path,
    max_subprocess_runs: int,
    force: bool,
) -> list[dict[str, object]]:
    batch_rows = _read_batch_csv(chunk_dir / "batch.csv")
    if not batch_rows:
        raise RuntimeError(f"Empty batch.csv: {chunk_dir / 'batch.csv'}")

    out_dir = chunk_dir / "out_sdf"
    tmp_dir = chunk_dir / "_runner_tmp"
    logs_dir = chunk_dir / "logs"
    ensure_dir(out_dir)
    ensure_dir(tmp_dir)
    ensure_dir(logs_dir)

    if force:
        for row in batch_rows:
            out_path = _unimol_expected_out(out_dir, row.output_ligand_name)
            if out_path.exists():
                out_path.unlink()
        summary_path = chunk_dir / "unimol_summary.csv"
        if summary_path.exists():
            summary_path.unlink()

    failures: dict[str, str] = {}
    subprocess_runs = 0

    def run_subset(rows: list[BatchRow]) -> None:
        nonlocal subprocess_runs

        rows = [row for row in rows if force or not _unimol_expected_out(out_dir, row.output_ligand_name).exists()]
        if not rows:
            return
        if subprocess_runs >= max_subprocess_runs:
            for row in rows:
                failures.setdefault(row.output_ligand_name, "too_many_subprocess_runs")
            return

        subprocess_runs += 1
        tmp_batch = tmp_dir / f"batch_try_{subprocess_runs}.csv"
        _write_batch_csv(tmp_batch, rows)
        cmd = _build_unimol_cmd(cfg, batch_csv=tmp_batch, out_dir=out_dir)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        _write_run_log(logs_dir / f"try_{subprocess_runs}.log", cmd=cmd, proc=proc)

        if proc.returncode == 0:
            return

        err_msg = _tail_text(proc.stderr) or _tail_text(proc.stdout) or f"demo.py returncode={proc.returncode}"
        if len(rows) == 1:
            failures[rows[0].output_ligand_name] = err_msg
            return

        mid = len(rows) // 2
        run_subset(rows[:mid])
        run_subset(rows[mid:])

    run_subset(batch_rows)

    summary_rows: list[dict[str, object]] = []
    for row in batch_rows:
        out_path = _unimol_expected_out(out_dir, row.output_ligand_name)
        if out_path.exists():
            status = "ok"
            error = ""
        else:
            error = failures.get(row.output_ligand_name, "")
            status = "failed" if error else "missing_output"
        summary_rows.append(
            {
                "ligand_id": row.output_ligand_name,
                "status": status,
                "output_sdf": str(out_path) if out_path.exists() else "",
                "error": error,
            }
        )

    write_csv(
        chunk_dir / "unimol_summary.csv",
        summary_rows,
        fieldnames=["ligand_id", "status", "output_sdf", "error"],
    )
    return summary_rows


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


def _build_gnina_cmd(
    cfg: DockingPipelineConfig,
    *,
    input_sdf: Path,
    output_sdf: Path,
) -> list[str]:
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


def _parse_gnina_out_sdf(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
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
        rows.append(
            {
                "ligand_id": ligand_id,
                "minimizedAffinity": safe_float(mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else None),
                "CNNscore": safe_float(mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else None),
                "CNNaffinity": safe_float(mol.GetProp("CNNaffinity") if mol.HasProp("CNNaffinity") else None),
            }
        )
    return rows


def _run_gnina_chunk(
    cfg: DockingPipelineConfig,
    *,
    chunk_dir: Path,
    force: bool,
) -> list[dict[str, object]]:
    unimol_rows = read_csv_rows(chunk_dir / "unimol_summary.csv")
    ok_rows = [row for row in unimol_rows if row.get("status", "").strip() == "ok" and row.get("output_sdf", "").strip()]

    gnina_dir = chunk_dir.parent.parent.parent / "gnina" / "chunks" / chunk_dir.name
    ensure_dir(gnina_dir)
    logs_dir = gnina_dir / "logs"
    ensure_dir(logs_dir)

    summary_csv = gnina_dir / "summary.csv"
    if summary_csv.exists() and not force:
        return read_csv_rows(summary_csv)

    if not ok_rows:
        write_csv(
            summary_csv,
            [],
            fieldnames=["ligand_id", "status", "minimizedAffinity", "CNNscore", "CNNaffinity", "gnina_out_sdf", "error"],
        )
        return []

    multi_input_sdf = gnina_dir / "input.sdf"
    multi_out_sdf = gnina_dir / "gnina_out.sdf"
    if force:
        for path in [summary_csv, multi_input_sdf, multi_out_sdf]:
            if path.exists():
                path.unlink()

    with multi_input_sdf.open("w", encoding="utf-8") as handle:
        for row in ok_rows:
            ligand_id = row["ligand_id"].strip()
            sdf_path = Path(row["output_sdf"])
            block = _first_mol_block_text(sdf_path)
            block = _ensure_ligand_id_props(block, ligand_id=ligand_id)
            handle.write(block)
            if not block.rstrip().endswith("$$$$"):
                handle.write("$$$$\n")

    cmd = _build_gnina_cmd(cfg, input_sdf=multi_input_sdf, output_sdf=multi_out_sdf)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    _write_run_log(logs_dir / "chunk_run.log", cmd=cmd, proc=proc)

    if proc.returncode == 0:
        parsed = _parse_gnina_out_sdf(multi_out_sdf)
        parsed_map = {str(row["ligand_id"]): row for row in parsed}
        summary_rows = []
        for row in ok_rows:
            ligand_id = row["ligand_id"].strip()
            scores = parsed_map.get(ligand_id, {})
            summary_rows.append(
                {
                    "ligand_id": ligand_id,
                    "status": "ok" if scores else "missing_output",
                    "minimizedAffinity": scores.get("minimizedAffinity"),
                    "CNNscore": scores.get("CNNscore"),
                    "CNNaffinity": scores.get("CNNaffinity"),
                    "gnina_out_sdf": str(multi_out_sdf) if scores else "",
                    "error": "",
                }
            )
        write_csv(
            summary_csv,
            summary_rows,
            fieldnames=["ligand_id", "status", "minimizedAffinity", "CNNscore", "CNNaffinity", "gnina_out_sdf", "error"],
        )
        return summary_rows

    summary_rows = []
    for row in ok_rows:
        ligand_id = row["ligand_id"].strip()
        single_dir = gnina_dir / "single_runs" / ligand_id
        ensure_dir(single_dir)
        input_sdf = single_dir / "input.sdf"
        out_sdf = single_dir / "gnina_out.sdf"
        if force:
            for path in [input_sdf, out_sdf]:
                if path.exists():
                    path.unlink()
        block = _ensure_ligand_id_props(_first_mol_block_text(Path(row["output_sdf"])), ligand_id=ligand_id)
        input_sdf.write_text(block if block.rstrip().endswith("$$$$") else block + "$$$$\n", encoding="utf-8")

        single_cmd = _build_gnina_cmd(cfg, input_sdf=input_sdf, output_sdf=out_sdf)
        single_proc = subprocess.run(single_cmd, text=True, capture_output=True)
        _write_run_log(logs_dir / f"{ligand_id}.log", cmd=single_cmd, proc=single_proc)

        if single_proc.returncode != 0:
            summary_rows.append(
                {
                    "ligand_id": ligand_id,
                    "status": "failed",
                    "minimizedAffinity": None,
                    "CNNscore": None,
                    "CNNaffinity": None,
                    "gnina_out_sdf": "",
                    "error": _tail_text(single_proc.stderr) or _tail_text(single_proc.stdout) or f"gnina returncode={single_proc.returncode}",
                }
            )
            continue

        parsed = _parse_gnina_out_sdf(out_sdf)
        scores = parsed[0] if parsed else {}
        summary_rows.append(
            {
                "ligand_id": ligand_id,
                "status": "ok" if scores else "missing_output",
                "minimizedAffinity": scores.get("minimizedAffinity"),
                "CNNscore": scores.get("CNNscore"),
                "CNNaffinity": scores.get("CNNaffinity"),
                "gnina_out_sdf": str(out_sdf) if scores else "",
                "error": "",
            }
        )

    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["ligand_id", "status", "minimizedAffinity", "CNNscore", "CNNaffinity", "gnina_out_sdf", "error"],
    )
    return summary_rows


def _write_final_summary(
    ligands: list[InputLigand],
    *,
    original_fieldnames: list[str],
    out_dir: Path,
) -> Path:
    prep_map = {row["ligand_id"]: row for row in read_csv_rows(out_dir / "prep_summary.csv")}

    unimol_map: dict[str, dict[str, str]] = {}
    for path in sorted((out_dir / "unimol" / "chunks").glob("chunk_*/unimol_summary.csv")):
        for row in read_csv_rows(path):
            unimol_map[row["ligand_id"]] = row

    gnina_map: dict[str, dict[str, str]] = {}
    for path in sorted((out_dir / "gnina" / "chunks").glob("chunk_*/summary.csv")):
        for row in read_csv_rows(path):
            gnina_map[row["ligand_id"]] = row

    summary_rows: list[dict[str, object]] = []
    for ligand in ligands:
        prep = prep_map.get(ligand.ligand_id, {})
        unimol = unimol_map.get(ligand.ligand_id, {})
        gnina = gnina_map.get(ligand.ligand_id, {})
        merged: dict[str, object] = dict(ligand.row)
        merged["ligand_id"] = ligand.ligand_id
        merged["source_mol_id"] = ligand.source_mol_id
        merged["input_smiles"] = ligand.smiles
        merged["prep_status"] = prep.get("status", "")
        merged["prepared_input_sdf"] = prep.get("input_sdf", "")
        merged["canonical_smiles"] = prep.get("canonical_smiles", "")
        merged["prep_error"] = prep.get("error", "")
        merged["unimol_status"] = unimol.get("status", "")
        merged["unimol_output_sdf"] = unimol.get("output_sdf", "")
        merged["unimol_error"] = unimol.get("error", "")
        merged["gnina_status"] = gnina.get("status", "")
        merged["gnina_output_sdf"] = gnina.get("gnina_out_sdf", "")
        merged["gnina_error"] = gnina.get("error", "")
        merged["minimizedAffinity"] = safe_float(gnina.get("minimizedAffinity"))
        merged["CNNscore"] = safe_float(gnina.get("CNNscore"))
        merged["CNNaffinity"] = safe_float(gnina.get("CNNaffinity"))
        summary_rows.append(merged)

    def sort_key(row: dict[str, object]) -> tuple[float, float, float, str]:
        cnn_aff = row.get("CNNaffinity")
        cnn_score = row.get("CNNscore")
        min_aff = row.get("minimizedAffinity")
        cnn_aff_val = float(cnn_aff) if isinstance(cnn_aff, (float, int)) else float("-inf")
        cnn_score_val = float(cnn_score) if isinstance(cnn_score, (float, int)) else float("-inf")
        min_aff_val = float(min_aff) if isinstance(min_aff, (float, int)) else float("inf")
        return (-cnn_aff_val, -cnn_score_val, min_aff_val, str(row.get("ligand_id", "")))

    summary_rows = sorted(summary_rows, key=sort_key)
    rank = 1
    for row in summary_rows:
        if row.get("gnina_status") == "ok":
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = ""

    extra_fields = [
        "rank",
        "ligand_id",
        "source_mol_id",
        "input_smiles",
        "prep_status",
        "prepared_input_sdf",
        "canonical_smiles",
        "prep_error",
        "unimol_status",
        "unimol_output_sdf",
        "unimol_error",
        "gnina_status",
        "gnina_output_sdf",
        "gnina_error",
        "minimizedAffinity",
        "CNNscore",
        "CNNaffinity",
    ]
    fieldnames = list(original_fieldnames)
    for name in extra_fields:
        if name not in fieldnames:
            fieldnames.append(name)

    final_dir = out_dir / "final"
    ensure_dir(final_dir)
    final_csv = final_dir / "rescored_with_unimol_gnina.csv"
    write_csv(final_csv, summary_rows, fieldnames=fieldnames)
    return final_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a ligand CSV (mol_id + smiles), build 3D SDFs with RDKit, "
            "dock with Uni-Mol, then rescore with gnina using the receptor and pocket "
            "from an existing docking-pipeline config."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config or normalized run.yaml with receptor/box/Uni-Mol/gnina settings.")
    parser.add_argument("--input-csv", type=Path, default=Path("dirty-task/top_ligands.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dirty-task/unimol_gnina_rescore"))
    parser.add_argument("--mol-id-col", default="mol_id")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--chunk-size", type=int, default=0, help="Override Uni-Mol chunk size. 0 means use config unimol.chunk_size.")
    parser.add_argument("--force", action="store_true", help="Rebuild ligand inputs and rerun Uni-Mol/gnina even if outputs already exist.")
    parser.add_argument("--max-unimol-runs", type=int, default=200, help="Hard cap for Uni-Mol retry/bisection runs per chunk.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = _load_cfg(args.config)

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    ligands, fieldnames = _read_input_csv(input_csv, mol_id_col=args.mol_id_col, smiles_col=args.smiles_col)
    prep_rows = _prepare_input_ligands(
        ligands,
        out_dir=output_dir,
        random_seed=cfg.run.random_seed,
        force=args.force,
    )
    docking_grid_json = _write_docking_grid_json(cfg, out_dir=output_dir)

    chunk_size = args.chunk_size if args.chunk_size > 0 else cfg.unimol.chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk size must be > 0")
    chunk_dirs = _chunk_prepared_ligands(
        prep_rows,
        out_dir=output_dir,
        docking_grid_json=docking_grid_json,
        chunk_size=chunk_size,
    )

    for chunk_dir in chunk_dirs:
        _run_unimol_chunk(
            cfg,
            chunk_dir=chunk_dir,
            max_subprocess_runs=args.max_unimol_runs,
            force=args.force,
        )
        _run_gnina_chunk(cfg, chunk_dir=chunk_dir, force=args.force)

    final_csv = _write_final_summary(ligands, original_fieldnames=fieldnames, out_dir=output_dir)

    ok_prep = sum(1 for row in prep_rows if str(row.get("status", "")).startswith("ok"))
    ok_gnina = 0
    for row in read_csv_rows(final_csv):
        if row.get("gnina_status", "").strip() == "ok":
            ok_gnina += 1

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Prepared ligands: {ok_prep}/{len(prep_rows)}")
    print(f"GNINA rescored ligands: {ok_gnina}/{len(ligands)}")
    print(f"Final CSV: {final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
