#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
bash "$(dirname "$0")/../cross_model/qwen35_9b.sh"
bash "$(dirname "$0")/../cross_model/qwen3vl_4b.sh"
bash "$(dirname "$0")/../cross_model/qwen3vl_8b.sh"

