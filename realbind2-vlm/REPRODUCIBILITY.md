# Reproducibility

## Environments

The main VLM environment is Python 3.10, PyTorch 2.7.0 with CUDA 12.8,
torchvision 0.22.0, Transformers 5.8.1, and bf16 for Qwen models. LLaVA uses
fp16. Exact environment files are under `envs/`.

MM-GDINO-L runs in its own environment with Transformers **4.55.4**,
tokenizers 0.21.4, and huggingface_hub 0.36.2. The prior Transformers 5.8.1
run produced incorrect box semantics and is historical invalid evidence. Do
not merge this environment with the main VLM environment.

SAM3 also has a separate environment. Record the installed SAM3 package
revision and checkpoint SHA256 in every run manifest.

## Determinism and provenance

- Dataset rows and GPU shards are ordered deterministically.
- Greedy generation uses `do_sample=false`; model-specific cache behavior is
  fixed in each YAML config.
- Every result row contains model, experiment, condition, config hash, dataset
  hash, normalized and raw prediction, runtime, and status.
- Successful rows are append-only and fsynced. Resume skips completed IDs.
- Failed rows are written separately. Shard merge rejects duplicates and
  requires exactly 9,800 baseline rows.
- `data/HASHES.json` protects prompts, scorer, model configs, manifest, and
  Strong Integrated assembly code.

The Qwen3.5 historical cache divergence between `use_cache=false` and
`use_cache=true` is real; Qwen3.5 configs retain `use_cache=false`. Changing it
is a protocol change.

