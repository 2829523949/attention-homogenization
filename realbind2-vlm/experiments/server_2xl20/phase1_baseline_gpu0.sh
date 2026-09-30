#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=2 SHARD_INDEX=0
bash "$(dirname "$0")/../qwen35_main/01_baseline.sh"

