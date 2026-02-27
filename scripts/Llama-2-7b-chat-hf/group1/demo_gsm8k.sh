#!/bin/bash
# Demo: Llama-2-7b-chat-hf | group1 | BadWords attack + KeywordFilter defense + gsm8k utility
# Run via tsp: tsp bash scripts/Llama-2-7b-chat-hf/group1/demo_gsm8k.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ── GPU selection ────────────────────────────────────────────────────────────
MIN_FREE_MB=20480   # 20 GB minimum for 7B fp16
source "$REPO_ROOT/base_select_gpu.sh"

# ── Environment ──────────────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sentrybench311

# ── Logging ─────────────────────────────────────────────────────────────────
MODEL="Llama-2-7b-chat-hf"
GROUP="group1"
RUN_NAME="demo_gsm8k"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/logs/$MODEL/$GROUP"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

echo "[$(date)] Starting $RUN_NAME | GPU=$CUDA_VISIBLE_DEVICES | log=$LOG_FILE"

# ── Run ──────────────────────────────────────────────────────────────────────
sentrybench run -c configs/demo_llama2_gsm8k.yaml \
    > "$LOG_FILE" 2>&1

echo "[$(date)] Done. Log: $LOG_FILE"
