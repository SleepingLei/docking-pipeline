from __future__ import annotations

import argparse
import glob
from pathlib import Path

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, read_csv_rows, write_ligand_id_list


def main() -> int:
    ap = argparse.ArgumentParser(description="Select top fraction by per-ligand best Uni-Dock2 score.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--stage", choices=["fast", "balance", "detail"], required=True)
    ap.add_argument("--top-fraction", type=float, required=True)
    ap.add_argument("--out-ligand-ids", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir
    stage_dir = run_dir / f"unidock_{args.stage}"
    chunk_dir = stage_dir / "chunk_summaries"
    ensure_dir(args.out_ligand_ids.parent)

    rows: list[dict[str, str]] = []
    for csv_path in sorted(chunk_dir.glob("summary_*.csv")):
        rows.extend(read_csv_rows(csv_path))

    # Keep best per ligand across chunks (should already be unique, but be safe).
    best: dict[str, float] = {}
    for r in rows:
        lid = r.get("ligand_id", "").strip()
        if not lid:
            continue
        try:
            score = float(r.get("vina_binding_free_energy", ""))
        except ValueError:
            continue
        cur = best.get(lid)
        if cur is None or score < cur:
            best[lid] = score

    items = sorted(best.items(), key=lambda kv: kv[1])  # lower (more negative) is better
    if not items:
        raise RuntimeError(f"No valid scores found for stage={args.stage} under {chunk_dir}")

    top_n = max(1, int(len(items) * args.top_fraction + 1e-9))
    selected = [lid for lid, _ in items[:top_n]]
    write_ligand_id_list(args.out_ligand_ids, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

