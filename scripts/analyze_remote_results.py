from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty values")
    i = (len(sorted_vals) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def _load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: dict[str, object] = dict(row)
            for k in [
                "rank",
                "cluster_rank",
                "unidock_fast_score",
                "unidock_balance_score",
                "unidock_detail_score",
                "minimizedAffinity",
                "CNNscore",
                "CNNaffinity",
            ]:
                out[k] = _to_float(row.get(k))
            out["cluster_id"] = str(row.get("cluster_id", "")).strip()
            out["ligand_id"] = str(row.get("ligand_id", "")).strip()
            out["smiles"] = str(row.get("smiles", "")).strip()
            rows.append(out)
    return rows


def _assign_cluster_rank_by_cnnaffinity(rows: list[dict[str, object]]) -> None:
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("cluster_id", "")).strip()
        if cid:
            by_cluster[cid].append(row)

    for cid in sorted(by_cluster, key=lambda x: int(x)):
        ranked = sorted(by_cluster[cid], key=_sort_key)
        for i, row in enumerate(ranked, start=1):
            row["cluster_rank"] = i


def _passes_cutoffs(row: dict[str, object], *, detail_cutoff: float | None, cnn_score_cutoff: float) -> bool:
    detail = row.get("unidock_detail_score")
    cnn_score = row.get("CNNscore")
    if not isinstance(cnn_score, (int, float)):
        return False
    if detail_cutoff is not None:
        if not isinstance(detail, (int, float)):
            return False
        if detail > detail_cutoff:
            return False
    return cnn_score >= cnn_score_cutoff


def _sort_key(row: dict[str, object]):
    cnn_aff = row.get("CNNaffinity")
    cnn_score = row.get("CNNscore")
    min_aff = row.get("minimizedAffinity")
    detail = row.get("unidock_detail_score")
    cluster_rank = row.get("cluster_rank")
    return (
        -(float(cnn_aff) if isinstance(cnn_aff, (int, float)) else float("-inf")),
        -(float(cnn_score) if isinstance(cnn_score, (int, float)) else float("-inf")),
        float(min_aff) if isinstance(min_aff, (int, float)) else float("inf"),
        float(detail) if isinstance(detail, (int, float)) else float("inf"),
        float(cluster_rank) if isinstance(cluster_rank, (int, float)) else float("inf"),
        str(row.get("ligand_id", "")),
    )


def _write_selected(
    rows: list[dict[str, object]],
    out_csv: Path,
    *,
    detail_cutoff: float | None,
    cnn_score_cutoff: float,
    global_cnn_rank_max: int | None,
) -> list[dict[str, object]]:
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("cluster_id", "")).strip()
        if cid:
            by_cluster[cid].append(row)

    selected: list[dict[str, object]] = []
    fieldnames = [
        "cluster_id",
        "selection_rank_in_cluster",
        "passes_cutoffs",
        "ligand_id",
        "smiles",
        "rank",
        "cluster_rank",
        "unidock_fast_score",
        "unidock_balance_score",
        "unidock_detail_score",
        "minimizedAffinity",
        "CNNscore",
        "CNNaffinity",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid in sorted(by_cluster, key=lambda x: int(x)):
            cluster_rows = by_cluster[cid]
            passing = [r for r in cluster_rows if _passes_cutoffs(r, detail_cutoff=detail_cutoff, cnn_score_cutoff=cnn_score_cutoff)]
            passing.sort(key=_sort_key)
            if global_cnn_rank_max is not None:
                passing = [
                    r for r in passing
                    if isinstance(r.get("global_rank_CNNaffinity"), (int, float))
                    and float(r["global_rank_CNNaffinity"]) <= global_cnn_rank_max
                ]
            chosen = passing[:5]
            for i, row in enumerate(chosen, start=1):
                out = {
                    "cluster_id": cid,
                    "selection_rank_in_cluster": i,
                    "passes_cutoffs": int(_passes_cutoffs(row, detail_cutoff=detail_cutoff, cnn_score_cutoff=cnn_score_cutoff)),
                    "ligand_id": row.get("ligand_id", ""),
                    "smiles": row.get("smiles", ""),
                    "rank": row.get("rank", ""),
                    "cluster_rank": row.get("cluster_rank", ""),
                    "unidock_fast_score": row.get("unidock_fast_score", ""),
                    "unidock_balance_score": row.get("unidock_balance_score", ""),
                    "unidock_detail_score": row.get("unidock_detail_score", ""),
                    "minimizedAffinity": row.get("minimizedAffinity", ""),
                    "CNNscore": row.get("CNNscore", ""),
                    "CNNaffinity": row.get("CNNaffinity", ""),
                }
                writer.writerow(out)
                selected.append(out)
    return selected


def _assign_global_ranks(rows: list[dict[str, object]]) -> None:
    detail_ranked = sorted(
        [r for r in rows if isinstance(r.get("unidock_detail_score"), (int, float))],
        key=lambda r: (float(r["unidock_detail_score"]), str(r.get("ligand_id", ""))),
    )
    for i, row in enumerate(detail_ranked, start=1):
        row["global_rank_unidock_detail"] = i

    cnn_ranked = sorted(
        [r for r in rows if isinstance(r.get("CNNaffinity"), (int, float))],
        key=lambda r: (-float(r["CNNaffinity"]), str(r.get("ligand_id", ""))),
    )
    for i, row in enumerate(cnn_ranked, start=1):
        row["global_rank_CNNaffinity"] = i


def _write_global_top(rows: list[dict[str, object]], out_csv: Path, *, metric: str, top_n: int) -> list[dict[str, object]]:
    if metric == "unidock_detail_score":
        ranked = sorted(
            [r for r in rows if isinstance(r.get(metric), (int, float))],
            key=lambda r: (float(r[metric]), str(r.get("ligand_id", ""))),
        )[:top_n]
    elif metric == "CNNaffinity":
        ranked = sorted(
            [r for r in rows if isinstance(r.get(metric), (int, float))],
            key=lambda r: (-float(r[metric]), str(r.get("ligand_id", ""))),
        )[:top_n]
    else:
        raise ValueError(f"unsupported metric: {metric}")

    fieldnames = [
        "selection_metric",
        "selection_rank_global",
        "ligand_id",
        "smiles",
        "cluster_id",
        "cluster_rank",
        "rank",
        "global_rank_unidock_detail",
        "global_rank_CNNaffinity",
        "unidock_fast_score",
        "unidock_balance_score",
        "unidock_detail_score",
        "minimizedAffinity",
        "CNNscore",
        "CNNaffinity",
    ]
    written: list[dict[str, object]] = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(ranked, start=1):
            out = {
                "selection_metric": metric,
                "selection_rank_global": i,
                "ligand_id": row.get("ligand_id", ""),
                "smiles": row.get("smiles", ""),
                "cluster_id": row.get("cluster_id", ""),
                "cluster_rank": row.get("cluster_rank", ""),
                "rank": row.get("rank", ""),
                "global_rank_unidock_detail": row.get("global_rank_unidock_detail", ""),
                "global_rank_CNNaffinity": row.get("global_rank_CNNaffinity", ""),
                "unidock_fast_score": row.get("unidock_fast_score", ""),
                "unidock_balance_score": row.get("unidock_balance_score", ""),
                "unidock_detail_score": row.get("unidock_detail_score", ""),
                "minimizedAffinity": row.get("minimizedAffinity", ""),
                "CNNscore": row.get("CNNscore", ""),
                "CNNaffinity": row.get("CNNaffinity", ""),
            }
            writer.writerow(out)
            written.append(out)
    return written


def _merge_shortlists(
    rows: list[dict[str, object]],
    cluster_selected: list[dict[str, object]],
    detail_top: list[dict[str, object]],
    cnn_top: list[dict[str, object]],
    out_csv: Path,
) -> list[dict[str, object]]:
    row_by_ligand = {str(r.get("ligand_id", "")): r for r in rows if str(r.get("ligand_id", ""))}

    cluster_ids = {str(r.get("ligand_id", "")) for r in cluster_selected}
    detail_ids = {str(r.get("ligand_id", "")) for r in detail_top}
    cnn_ids = {str(r.get("ligand_id", "")) for r in cnn_top}
    union_ids = sorted(cluster_ids | detail_ids | cnn_ids)

    fieldnames = [
        "ligand_id",
        "smiles",
        "cluster_id",
        "cluster_rank",
        "rank",
        "global_rank_unidock_detail",
        "global_rank_CNNaffinity",
        "selected_cluster_top5",
        "selected_global_top20_unidock_detail",
        "selected_global_top20_CNNaffinity",
        "selection_sources",
        "unidock_fast_score",
        "unidock_balance_score",
        "unidock_detail_score",
        "minimizedAffinity",
        "CNNscore",
        "CNNaffinity",
    ]

    merged: list[dict[str, object]] = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ligand_id in union_ids:
            row = row_by_ligand[ligand_id]
            srcs: list[str] = []
            if ligand_id in cluster_ids:
                srcs.append("cluster_top5")
            if ligand_id in detail_ids:
                srcs.append("global_top20_unidock_detail")
            if ligand_id in cnn_ids:
                srcs.append("global_top20_CNNaffinity")
            out = {
                "ligand_id": ligand_id,
                "smiles": row.get("smiles", ""),
                "cluster_id": row.get("cluster_id", ""),
                "cluster_rank": row.get("cluster_rank", ""),
                "rank": row.get("rank", ""),
                "global_rank_unidock_detail": row.get("global_rank_unidock_detail", ""),
                "global_rank_CNNaffinity": row.get("global_rank_CNNaffinity", ""),
                "selected_cluster_top5": int(ligand_id in cluster_ids),
                "selected_global_top20_unidock_detail": int(ligand_id in detail_ids),
                "selected_global_top20_CNNaffinity": int(ligand_id in cnn_ids),
                "selection_sources": ";".join(srcs),
                "unidock_fast_score": row.get("unidock_fast_score", ""),
                "unidock_balance_score": row.get("unidock_balance_score", ""),
                "unidock_detail_score": row.get("unidock_detail_score", ""),
                "minimizedAffinity": row.get("minimizedAffinity", ""),
                "CNNscore": row.get("CNNscore", ""),
                "CNNaffinity": row.get("CNNaffinity", ""),
            }
            writer.writerow(out)
            merged.append(out)
    return merged


def _make_plots(rows: list[dict[str, object]], out_dir: Path, *, detail_cutoff: float | None, cnn_score_cutoff: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {
        "CNNaffinity": [float(r["CNNaffinity"]) for r in rows if isinstance(r.get("CNNaffinity"), (int, float))],
        "CNNscore": [float(r["CNNscore"]) for r in rows if isinstance(r.get("CNNscore"), (int, float))],
        "minimizedAffinity": [float(r["minimizedAffinity"]) for r in rows if isinstance(r.get("minimizedAffinity"), (int, float)) and float(r["minimizedAffinity"]) < 100],
        "unidock_detail_score": [float(r["unidock_detail_score"]) for r in rows if isinstance(r.get("unidock_detail_score"), (int, float))],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, vals) in zip(axes.flat, metrics.items()):
        ax.hist(vals, bins=60, color="#4c78a8", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        if name == "CNNscore":
            ax.axvline(cnn_score_cutoff, color="#e45756", linestyle="--", label=f"cutoff={cnn_score_cutoff:.2f}")
            ax.legend()
        elif name == "CNNaffinity":
            pass
        elif name == "unidock_detail_score":
            if detail_cutoff is not None:
                ax.axvline(detail_cutoff, color="#e45756", linestyle="--", label=f"cutoff={detail_cutoff:.2f}")
                ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    random.seed(1234567)
    scatter_rows = [r for r in rows if isinstance(r.get("CNNaffinity"), (int, float)) and isinstance(r.get("CNNscore"), (int, float))]
    if len(scatter_rows) > 25000:
        scatter_rows = random.sample(scatter_rows, 25000)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10
    for cid in sorted({str(r.get("cluster_id", "")) for r in scatter_rows if str(r.get("cluster_id", ""))} , key=lambda x: int(x)):
        pts = [r for r in scatter_rows if str(r.get("cluster_id", "")) == cid]
        ax.scatter(
            [float(r["CNNscore"]) for r in pts],
            [float(r["CNNaffinity"]) for r in pts],
            s=8,
            alpha=0.35,
            label=f"cluster {cid}",
            color=colors(int(cid) % 10),
        )
    ax.axvline(cnn_score_cutoff, color="#e45756", linestyle="--")
    ax.set_xlabel("CNNscore")
    ax.set_ylabel("CNNaffinity")
    ax.set_title("CNNaffinity vs CNNscore (sampled)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "cnnaffinity_vs_cnnscore.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    total_counts = Counter(str(r.get("cluster_id", "")) for r in rows if str(r.get("cluster_id", "")))
    pass_counts = Counter(
        str(r.get("cluster_id", ""))
        for r in rows
        if str(r.get("cluster_id", "")) and _passes_cutoffs(r, detail_cutoff=detail_cutoff, cnn_score_cutoff=cnn_score_cutoff)
    )
    cids = sorted(total_counts, key=lambda x: int(x))
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = list(range(len(cids)))
    ax.bar(xs, [total_counts[c] for c in cids], color="#9ecae9", label="all")
    ax.bar(xs, [pass_counts[c] for c in cids], color="#3182bd", label="pass cutoffs")
    ax.set_xticks(xs)
    ax.set_xticklabels(cids)
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("Ligand count")
    ax.set_title("Cluster sizes and candidates passing cutoffs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_pass_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10.colors
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("cluster_id", "")).strip()
        if cid and isinstance(row.get("CNNaffinity"), (int, float)) and isinstance(row.get("cluster_rank"), (int, float)):
            by_cluster[cid].append(row)
    for i, cid in enumerate(sorted(by_cluster, key=lambda x: int(x))):
        pts = sorted(by_cluster[cid], key=lambda r: float(r["cluster_rank"]))
        ranks = [int(r["cluster_rank"]) for r in pts]
        vals = [float(r["CNNaffinity"]) for r in pts]
        ax.plot(ranks, vals, label=f"cluster {cid}", color=colors[i % 10], linewidth=1.8, alpha=0.9)
    ax.set_xlabel("cluster_rank (by CNNaffinity within cluster)")
    ax.set_ylabel("CNNaffinity")
    ax.set_title("Per-cluster CNNaffinity rank distribution")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xlim(left=1)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_rank_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_union_plots(rows: list[dict[str, object]], merged_rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(1234567)
    base_rows = [
        r for r in rows
        if isinstance(r.get("CNNaffinity"), (int, float)) and isinstance(r.get("unidock_detail_score"), (int, float))
    ]
    if len(base_rows) > 25000:
        base_rows = random.sample(base_rows, 25000)

    merged_ligands = {str(r.get("ligand_id", "")) for r in merged_rows}
    selected_rows = [
        r for r in rows
        if str(r.get("ligand_id", "")) in merged_ligands
        and isinstance(r.get("CNNaffinity"), (int, float))
        and isinstance(r.get("unidock_detail_score"), (int, float))
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        [float(r["unidock_detail_score"]) for r in base_rows],
        [float(r["CNNaffinity"]) for r in base_rows],
        s=10,
        alpha=0.15,
        color="#9aa0a6",
        label="all ligands (sampled)",
    )
    ax.scatter(
        [float(r["unidock_detail_score"]) for r in selected_rows],
        [float(r["CNNaffinity"]) for r in selected_rows],
        s=44,
        alpha=0.95,
        color="#d62728",
        edgecolors="black",
        linewidths=0.3,
        label="final shortlist",
    )
    ax.set_xlabel("unidock_detail_score")
    ax.set_ylabel("CNNaffinity")
    ax.set_title("Final shortlist in score space")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "final_union_score_space.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    counts = Counter(str(r.get("cluster_id", "")) for r in merged_rows if str(r.get("cluster_id", "")))
    cids = sorted(counts, key=lambda x: int(x))
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = list(range(len(cids)))
    ax.bar(xs, [counts[c] for c in cids], color="#4c78a8")
    ax.set_xticks(xs)
    ax.set_xticklabels(cids)
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("Shortlist count")
    ax.set_title("Final shortlist counts by cluster")
    fig.tight_layout()
    fig.savefig(out_dir / "final_union_cluster_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze remote screening results and select top ligands per cluster.")
    ap.add_argument("--in-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--detail-cutoff", type=float, default=None)
    ap.add_argument("--cnnscore-cutoff", type=float, default=0.5)
    ap.add_argument("--global-topn-detail", type=int, default=5)
    ap.add_argument("--global-topn-cnnaffinity", type=int, default=20)
    ap.add_argument("--global-cnn-rank-max", type=int, default=200)
    args = ap.parse_args()

    rows = _load_rows(args.in_csv)
    _assign_cluster_rank_by_cnnaffinity(rows)
    _assign_global_ranks(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = _write_selected(
        rows,
        args.out_dir / "top5_per_cluster_by_cnnaffinity.csv",
        detail_cutoff=args.detail_cutoff,
        cnn_score_cutoff=args.cnnscore_cutoff,
        global_cnn_rank_max=args.global_cnn_rank_max,
    )
    detail_top = _write_global_top(
        rows,
        args.out_dir / f"global_top{args.global_topn_detail}_by_unidock_detail.csv",
        metric="unidock_detail_score",
        top_n=args.global_topn_detail,
    )
    cnn_top = _write_global_top(
        rows,
        args.out_dir / f"global_top{args.global_topn_cnnaffinity}_by_cnnaffinity.csv",
        metric="CNNaffinity",
        top_n=args.global_topn_cnnaffinity,
    )
    merged = _merge_shortlists(
        rows,
        selected,
        detail_top,
        cnn_top,
        args.out_dir / "final_union_shortlist.csv",
    )
    _make_plots(
        rows,
        args.out_dir,
        detail_cutoff=args.detail_cutoff,
        cnn_score_cutoff=args.cnnscore_cutoff,
    )
    _make_union_plots(rows, merged, args.out_dir)

    total_by_cluster = Counter(str(r.get("cluster_id", "")) for r in rows if str(r.get("cluster_id", "")))
    pass_by_cluster = Counter(
        str(r.get("cluster_id", ""))
        for r in rows
        if str(r.get("cluster_id", "")) and _passes_cutoffs(r, detail_cutoff=args.detail_cutoff, cnn_score_cutoff=args.cnnscore_cutoff)
    )
    summary_path = args.out_dir / "selection_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Cutoffs used:\n")
        f.write(f"  CNNaffinity: primary ranking metric (higher is better)\n")
        f.write(f"  CNNscore >= {args.cnnscore_cutoff:.3f}\n")
        f.write(f"  global CNNaffinity rank <= {args.global_cnn_rank_max}\n")
        if args.detail_cutoff is not None:
            f.write(f"  unidock_detail_score <= {args.detail_cutoff:.3f}\n")
        else:
            f.write("  unidock_detail_score: not used as cutoff\n")
        f.write("  minimizedAffinity: not used as cutoff\n\n")
        f.write("Per-cluster pass counts:\n")
        for cid in sorted(total_by_cluster, key=lambda x: int(x)):
            f.write(f"  cluster {cid}: pass={pass_by_cluster[cid]} / total={total_by_cluster[cid]}\n")
        f.write(f"\nSelected rows written: {len(selected)}\n")
        f.write(f"Global top{args.global_topn_detail} by unidock_detail_score: {len(detail_top)}\n")
        f.write(f"Global top{args.global_topn_cnnaffinity} by CNNaffinity: {len(cnn_top)}\n")
        f.write(f"Final merged shortlist (union, deduplicated): {len(merged)}\n")
        f.write("Note: per-cluster picks are limited to ligands with global CNNaffinity rank <= global_cnn_rank_max; clusters may contribute fewer than 5 ligands.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
