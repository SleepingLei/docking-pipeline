from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem

from docking_pipeline.steps.common import ensure_dir, write_csv


def _iter_sdf_blocks(path: Path):
    buf: list[str] = []
    idx = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line)
            if line.strip() == "$$$$":
                yield idx, "".join(buf), True
                buf = []
                idx += 1
    if buf:
        yield idx, "".join(buf), False


def _first_line(block: str) -> str:
    lines = block.splitlines()
    return lines[0].strip() if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate each record in an SDF and optionally split good/bad subsets.")
    ap.add_argument("--in-sdf", type=Path, required=True)
    ap.add_argument("--report-csv", type=Path, required=True)
    ap.add_argument("--good-sdf", type=Path, default=None)
    ap.add_argument("--bad-sdf", type=Path, default=None)
    ap.add_argument("--sanitize", action="store_true", help="Also run RDKit sanitization checks.")
    args = ap.parse_args()

    ensure_dir(args.report_csv.parent)
    if args.good_sdf is not None:
        ensure_dir(args.good_sdf.parent)
    if args.bad_sdf is not None:
        ensure_dir(args.bad_sdf.parent)

    good_fh = args.good_sdf.open("w", encoding="utf-8") if args.good_sdf is not None else None
    bad_fh = args.bad_sdf.open("w", encoding="utf-8") if args.bad_sdf is not None else None

    rows: list[dict[str, object]] = []
    try:
        for idx, block, has_terminator in _iter_sdf_blocks(args.in_sdf):
            name = _first_line(block)
            status = "ok"
            detail = ""

            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is None:
                status = "parse_fail"
                detail = "Chem.MolFromMolBlock returned None"
            else:
                if mol.GetNumAtoms() == 0:
                    status = "zero_atoms"
                    detail = "parsed molecule has 0 atoms"
                elif mol.GetNumConformers() == 0:
                    status = "no_conformer"
                    detail = "parsed molecule has no conformer"
                elif args.sanitize:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception as e:  # pragma: no cover - depends on vendor chemistry
                        status = "sanitize_fail"
                        detail = str(e)

            if not has_terminator and status == "ok":
                status = "missing_terminator"
                detail = "record missing $$$$ terminator"

            rows.append(
                {
                    "record_index": idx,
                    "name": name,
                    "status": status,
                    "detail": detail,
                    "has_terminator": int(has_terminator),
                    "num_lines": len(block.splitlines()),
                }
            )

            target = good_fh if status == "ok" else bad_fh
            if target is not None:
                target.write(block)
                if not block.rstrip().endswith("$$$$"):
                    target.write("\n$$$$\n")
    finally:
        if good_fh is not None:
            good_fh.close()
        if bad_fh is not None:
            bad_fh.close()

    write_csv(
        args.report_csv,
        rows,
        fieldnames=["record_index", "name", "status", "detail", "has_terminator", "num_lines"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
