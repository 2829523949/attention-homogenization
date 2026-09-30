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

The certified frozen release contains 1,400 families and 973 unique source
images. Copy the complete `frozen/` directory outside Git and verify it with:

```bash
python scripts/verify_frozen_export.py --dataset-root /path/to/RealBind_v2
```

Do not rerun the builder export against the already-frozen database: the
historical exporter selects only `HUMAN_ACCEPTED` rows and would produce an
empty export after those rows have become `FROZEN`.

Model weights, RealBind images, masks, caches, SQLite state, and raw outputs are
ignored by Git. Git LFS is intentionally not used.
