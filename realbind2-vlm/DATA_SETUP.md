# Data and model setup

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml`.
2. Point `data.realbind2` to the RealBind2 construction root. A formal root
   must contain `frozen/families.jsonl`, `frozen/questions.jsonl`,
   `frozen/splits.json`, and `frozen/construction_stats.json`.
3. Place model weights outside this repository and fill each path under
   `models`. Do not symlink credentials or Hugging Face caches into the repo.
4. Point `historical.repair_assets_root` to the retained historical repair
   workspace if reproducing 40-case development diagnostics.
5. Run `python scripts/audit_dataset.py --data-root /path/to/RealBind_v2`.

The current construction snapshot at packaging time contains 1,400
`AUTO_VALID` families and 973 unique Visual Genome source image IDs. It has no
`HUMAN_ACCEPTED` or `FROZEN` families. Complete the existing audit/import/export
workflow before formal evaluation; do not synthesize an export from the
`AUTO_VALID` rows.

Model weights, RealBind images, masks, caches, SQLite state, and raw outputs are
ignored by Git. Git LFS is intentionally not used.

