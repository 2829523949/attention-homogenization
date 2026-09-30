# RealBind2 VLM repair

Unified, reproducible code for the RealBind2 baseline, prerequisite-controlled
failure cohorts, visual interference, Qwen3.5 causal analysis, historical
repair retrospective, and the frozen Strong Integrated repair.

RealBind2 has seven deterministic questions per benchmark item: two I2C, one
pair relation, two REL2I, and two REL2C. The formal target is 1,400 items,
9,800 QA rows, and 2,800 composition rows. The frozen release contains
1,400 families on 973 unique source images. Explicit project-owner acceptance,
builder export, exact QA regeneration, and all image hashes are recorded in
[FREEZE_REPORT.md](FREEZE_REPORT.md).

## Experiment groups

- `experiments/qwen35_main/`: primary Qwen3.5-4B pipeline. This is the only
  full causal pipeline.
- `experiments/cross_model/`: frozen-protocol generalization. These runners do
  not reopen repair search.
- `src/repair/strong_integrated/`: final repair. Grounding is R4
  MM-GDINO-L + Qwen selector; SAM3 isolates the selected target; crop-Qwen
  reads the attribute; final reasoning is text only.
- `legacy/`: byte-preserving source snapshots and compact evidence from prior
  experiments, including failed SAM3 consolidation.

## Model matrix

| Model | Baseline | VIR | Causal | Frozen repair |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | full | full | full | full |
| Qwen3.5-9B | full | full | no | final reasoning |
| Qwen3-VL-4B | full | full | reduced depths | final reasoning |
| Qwen3-VL-8B | full | no | no | optional |
| InternVL3.5-8B | full | full | no | final reasoning |
| LLaVA-NeXT-7B | full | full | reduced depths | optional |

## Quick start

```bash
cp configs/paths.example.yaml configs/paths.yaml
python scripts/verify_hashes.py
pytest
bash experiments/qwen35_main/00_smoke.sh
```

Artifact smoke is self-contained. Server setup and live certification are in
[SERVER_HANDOFF.md](SERVER_HANDOFF.md). The dataset is certified and live GPU
certification remains pending; see
[PRE_SERVER_STATUS.md](PRE_SERVER_STATUS.md).

For two L20 GPUs, launch the paired scripts under
`experiments/server_2xl20/`. They use independent jobs and deterministic
shards; no tensor parallelism is enabled. Outputs are append-only under
`outputs/`, and rerunning a command skips completed case IDs.

Integrity checks are `scripts/verify_hashes.py`, `scripts/audit_dataset.py`,
and `scripts/audit_outputs.py`. Dataset and weight setup is in
[DATA_SETUP.md](DATA_SETUP.md).
