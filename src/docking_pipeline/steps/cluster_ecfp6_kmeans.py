from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

from docking_pipeline.steps.common import ensure_dir, load_run_cfg, read_csv_rows, write_csv, write_ligand_id_list


def main() -> int:
    ap = argparse.ArgumentParser(description="ECFP6 (Morgan r=3) KMeans clustering and representative selection.")
    ap.add_argument("--run-yaml", type=Path, required=True)
    ap.add_argument("--in-stage", choices=["detail"], default="detail")
    ap.add_argument("--clusters", type=int, required=True)
    ap.add_argument("--per-cluster", type=int, required=True)
    ap.add_argument("--out-ligand-ids", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_run_cfg(args.run_yaml)
    run_dir = cfg.run.work_dir

    detail_inputs = run_dir / "inputs" / "detail" / "chunks"
    detail_summaries = run_dir / "unidock_detail" / "chunk_summaries"
    ensure_dir(run_dir / "clustering")
    ensure_dir(args.out_ligand_ids.parent)

    # Load detail scores
    score_rows: list[dict[str, str]] = []
    for p in sorted(detail_summaries.glob("summary_*.csv")):
        score_rows.extend(read_csv_rows(p))
    scores: dict[str, float] = {}
    for r in score_rows:
        lid = r.get("ligand_id", "").strip()
        if not lid:
            continue
        try:
            s = float(r.get("vina_binding_free_energy", ""))
        except ValueError:
            continue
        cur = scores.get(lid)
        if cur is None or s < cur:
            scores[lid] = s

    # Build fingerprint matrix for ligands present in the detail input set (not the output poses).
    lids: list[str] = []
    fps: list[np.ndarray] = []
    for sdf in sorted(detail_inputs.glob("chunk*.sdf")):
        sup = Chem.SDMolSupplier(str(sdf), removeHs=False)
        for mol in sup:
            if mol is None:
                continue
            lid = mol.GetProp("ligand_id") if mol.HasProp("ligand_id") else ""
            if not lid:
                continue
            if lid not in scores:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            arr = np.zeros((2048,), dtype=np.uint8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            lids.append(lid)
            fps.append(arr)

    if not lids:
        raise RuntimeError("No ligands with valid fingerprints found for clustering.")

    X = np.stack(fps, axis=0)
    n_clusters = min(args.clusters, X.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=cfg.run.random_seed, n_init="auto")
    labels = km.fit_predict(X)

    # Representatives: sort per cluster by best detail score (ascending) and take top k.
    cluster_to_ligs: dict[int, list[str]] = {}
    for lid, c in zip(lids, labels, strict=True):
        cluster_to_ligs.setdefault(int(c), []).append(lid)

    rep_ids: list[str] = []
    cluster_rows: list[dict[str, object]] = []
    for cluster_id, ligs in sorted(cluster_to_ligs.items(), key=lambda kv: kv[0]):
        ligs_sorted = sorted(ligs, key=lambda lid: scores.get(lid, 1e9))
        for rank, lid in enumerate(ligs_sorted, start=1):
            is_rep = rank <= args.per_cluster
            cluster_rows.append(
                {
                    "ligand_id": lid,
                    "cluster_id": cluster_id,
                    "cluster_rank": rank,
                    "is_representative": int(is_rep),
                    "unidock_detail_score": scores.get(lid, ""),
                }
            )
            if is_rep:
                rep_ids.append(lid)

    write_csv(
        run_dir / "clustering" / "clusters.csv",
        cluster_rows,
        fieldnames=["ligand_id", "cluster_id", "cluster_rank", "is_representative", "unidock_detail_score"],
    )
    write_ligand_id_list(args.out_ligand_ids, rep_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

