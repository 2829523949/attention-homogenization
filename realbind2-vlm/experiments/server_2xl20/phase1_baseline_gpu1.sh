#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 NUM_SHARDS=2 SHARD_INDEX=1
bash "$(dirname "$0")/../qwen35_main/01_baseline.sh"

