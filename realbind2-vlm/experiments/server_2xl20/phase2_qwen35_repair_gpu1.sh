#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
bash "$(dirname "$0")/../qwen35_main/05_repair_retrospective.sh"
bash "$(dirname "$0")/../qwen35_main/06_strong_integrated.sh"

