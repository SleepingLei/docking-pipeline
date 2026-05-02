from __future__ import annotations

import argparse
from pathlib import Path

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, read_csv_rows, safe_float, write_csv


def _load_stage_scores(run_dir: Path, stage: str) -> dict[str, float]:
    d: dict[str, float] = {}
    stage_dir = run_dir / f"unidock_{stage}" / "chunk_summaries"
    for p in sorted(stage_dir.glob("summary_*.csv")):
        for r in read_csv_rows(p):
            lid = r.get("ligand_id", "").strip()
            if not lid:
                continue
            try:
                s = float(r.get("vina_binding_free_energy", ""))
            except ValueError:
                continue
            cur = d.get(lid)
            if cur is None or s < cur:
                d[lid] = s
    return d


def _load_gnina(run_dir: Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    root = run_dir / "gnina" / "chunks"
    for chunk in sorted(root.glob("chunk_*")):
        p = chunk / "summary.csv"
        if not p.exists():
            continue
        for r in read_csv_rows(p):
            lid = r.get("ligand_id", "").strip()
            if not lid:
                continue
            out[lid] = {
                "minimizedAffinity": safe_float(r.get("minimizedAffinity")),
                "CNNscore": safe_float(r.get("CNNscore")),
                "CNNaffinity": safe_float(r.get("CNNaffinity")),
            }
    return out


def _load_manifest_smiles(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    manifest_csv = run_dir / "inputs" / "ligands_manifest.csv"
    if not manifest_csv.exists():
        return out
    for r in read_csv_rows(manifest_csv):
        lid = r.get("ligand_id", "").strip()
        if not lid:
            continue
        out[lid] = r.get("smiles", "").strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge scores and compute final rank.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir
    ensure_dir(run_dir / "final")

    fast = _load_stage_scores(run_dir, "fast")
    bal = _load_stage_scores(run_dir, "balance")
    det = _load_stage_scores(run_dir, "detail")
    gn = _load_gnina(run_dir)
    smiles_map = _load_manifest_smiles(run_dir)

    # clustering info (optional)
    cluster_map: dict[str, dict[str, str]] = {}
    clusters_csv = run_dir / "clustering" / "clusters.csv"
    if clusters_csv.exists():
        for r in read_csv_rows(clusters_csv):
            lid = r.get("ligand_id", "").strip()
            if not lid:
                continue
            cluster_map[lid] = r

    all_lids = sorted(set(det.keys()) | set(gn.keys()) | set(cluster_map.keys()))
    rows: list[dict[str, object]] = []
    for lid in all_lids:
        row: dict[str, object] = {"ligand_id": lid}
        if lid in smiles_map:
            row["smiles"] = smiles_map[lid]
        if lid in fast:
            row["unidock_fast_score"] = fast[lid]
        if lid in bal:
            row["unidock_balance_score"] = bal[lid]
        if lid in det:
            row["unidock_detail_score"] = det[lid]
        if lid in cluster_map:
            row["cluster_id"] = cluster_map[lid].get("cluster_id", "")
            row["cluster_rank"] = cluster_map[lid].get("cluster_rank", "")
            row["is_representative"] = cluster_map[lid].get("is_representative", "")
        if lid in gn:
            row.update({k: v for k, v in gn[lid].items()})
        rows.append(row)

    # Rank: CNNaffinity desc, CNNscore desc, unidock_detail_score asc
    def sort_key(r: dict[str, object]):
        cnn_aff = r.get("CNNaffinity")
        cnn_s = r.get("CNNscore")
        det_s = r.get("unidock_detail_score")
        # Missing values go last.
        cnn_aff_v = float(cnn_aff) if isinstance(cnn_aff, (int, float)) else float("-inf")
        cnn_s_v = float(cnn_s) if isinstance(cnn_s, (int, float)) else float("-inf")
        det_s_v = float(det_s) if isinstance(det_s, (int, float)) else float("inf")
        return (-cnn_aff_v, -cnn_s_v, det_s_v, r.get("ligand_id", ""))

    rows_sorted = sorted(rows, key=sort_key)
    for idx, r in enumerate(rows_sorted, start=1):
        r["rank"] = idx

    # Determine fieldnames union (stable order)
    fieldnames = [
        "rank",
        "ligand_id",
        "smiles",
        "cluster_id",
        "cluster_rank",
        "is_representative",
        "unidock_fast_score",
        "unidock_balance_score",
        "unidock_detail_score",
        "minimizedAffinity",
        "CNNscore",
        "CNNaffinity",
    ]
    write_csv(run_dir / "final" / "final_scores.csv", rows_sorted, fieldnames=fieldnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
