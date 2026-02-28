#!/bin/bash
# BadNets end-to-end: Llama-2-7b-chat-hf | group2
# Finetune on poisoned data, then three-stage eval (clean/attacked/defended)
# Run via: tsp bash scripts/Llama-2-7b-chat-hf/group2/badnets_gsm8k.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── GPU selection (need >=24GB for 7B fp16 + LoRA training) ─────────────────
MIN_FREE_MB=24576
source "$REPO_ROOT/base_select_gpu.sh"

# ── Environment ──────────────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentrybench311

# ── Logging ─────────────────────────────────────────────────────────────────
MODEL="Llama-2-7b-chat-hf"
GROUP="group2"
RUN_NAME="badnets_gsm8k"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/logs/$MODEL/$GROUP"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

echo "[$(date)] Starting $RUN_NAME | GPU=$CUDA_VISIBLE_DEVICES | log=$LOG_FILE"

# ── Run ──────────────────────────────────────────────────────────────────────
sentrybench run -c configs/demo_badnets_llama2.yaml \
    > "$LOG_FILE" 2>&1

echo "[$(date)] Done. Log: $LOG_FILE"
