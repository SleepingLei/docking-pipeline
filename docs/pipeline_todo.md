# Pipeline Todo

目标流程：

```text
protein/pocket + ligands.sdf
  -> unidock2 fast
  -> top 20%
  -> unidock2 balance
  -> top 15%
  -> ECFP6 KMeans to N clusters
  -> Uni-Mol Docking V2
  -> gnina rescoring
  -> score + rank
```

## Milestone 1: Controller Skeleton

- Define YAML config schema.
- Create stable run directory layout.
- Create ligand manifest with `ligand_id`, source SDF index, molecule name, status.
- Split large SDF into chunk files.
- Merge per-stage manifests.
- Add resume checks.

## Milestone 2: Uni-Dock2 Stages

- Generate Uni-Dock2 YAML for `fast`, `balance`, `detail`.
- Use exact search mode names:
  - `fast`
  - `balance`
  - `detail`
- Parse Uni-Dock2 SDF properties:
  - `vina_binding_free_energy`
  - `vina_intra_inter`
  - `vina_inter`
- Select best pose per ligand before applying top fractions.
- Generate Slurm array scripts for each stage.

## Milestone 3: Clustering

- Generate ECFP6 fingerprints with RDKit Morgan radius 3.
- Run KMeans to `N` clusters.
- Pick cluster representatives by best Uni-Dock2 detail score.
- Persist `cluster_id`, `cluster_rank`, representative flag.

## Milestone 4: Uni-Mol Docking V2

- Generate `docking_grid.json`.
- Generate Uni-Mol `batch_one2many` CSV.
- Wrap `interface/demo.py`.
- Support two modes:
  - generated conformers: `--conf-size 10 --cluster`
  - pose reuse: `--use_current_ligand_conf`
- Parse output SDF and keep pose path in manifest.

## Milestone 5: gnina Rescoring

- Support `--score_only` for pure scoring.
- Support `--minimize -o out.sdf` for local optimization and SDF property output.
- Parse:
  - `minimizedAffinity`
  - `CNNscore`
  - `CNNaffinity`
  - `CNN_VS`

## Milestone 6: Final Ranking

- Produce `final_scores.csv`.
- Produce `ranked.sdf`.
- Default sort:
  1. `CNNaffinity` descending
  2. `CNNscore` descending
  3. `unidock_detail_score` ascending
- Keep all intermediate scores for audit.

## Important Notes

- Do not rank raw poses for top fraction; rank best pose per ligand.
- Do not run heavy work on the login node.
- Keep Slurm job arrays idempotent.
- Keep every stage restartable by checking manifest status.
- Keep failed ligands in the manifest with reasons instead of silently dropping them.
