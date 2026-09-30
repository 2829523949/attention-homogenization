#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
bash "$(dirname "$0")/../cross_model/internvl35_8b.sh"
bash "$(dirname "$0")/../cross_model/llava_next_7b.sh"

