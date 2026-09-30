#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
bash "$(dirname "$0")/../qwen35_main/04_causal.sh"

