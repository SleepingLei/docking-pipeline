from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, read_ligand_id_list, write_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter an input chunks dir to selected ligands and re-chunk.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--in-chunks-dir", type=Path, required=True)
    ap.add_argument("--ligand-ids", type=Path, required=True)
    ap.add_argument("--out-chunks-dir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    args = ap.parse_args()

    selected = set(read_ligand_id_list(args.ligand_ids))
    ensure_dir(args.out_chunks_dir)

    out_manifest: list[dict[str, object]] = []

    next_chunk_id = 0
    in_chunk = 0
    writer: Chem.SDWriter | None = None

    def ensure_writer():
        nonlocal next_chunk_id, in_chunk, writer
        if writer is None or in_chunk >= args.chunk_size:
            if writer is not None:
                writer.close()
            out_path = args.out_chunks_dir / f"chunk{next_chunk_id}.sdf"
            writer = Chem.SDWriter(str(out_path))
            in_chunk = 0
            next_chunk_id += 1

    for in_sdf in sorted(args.in_chunks_dir.glob("chunk*.sdf")):
        supplier = Chem.SDMolSupplier(str(in_sdf), removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            lid = mol.GetProp("ligand_id") if mol.HasProp("ligand_id") else ""
            if lid not in selected:
                continue
            ensure_writer()
            writer.write(mol)
            out_manifest.append({"ligand_id": lid, "source_chunk": in_sdf.name})
            in_chunk += 1

    if writer is None:
        raise RuntimeError("No ligands selected for filtering; produced 0 output chunks.")
    writer.close()

    write_csv(args.out_chunks_dir.parent / "filtered_manifest.csv", out_manifest, fieldnames=["ligand_id", "source_chunk"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
