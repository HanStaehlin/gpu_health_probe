#!/bin/bash
# GPU node health check (probe -> report -> gate). Exit 0 = PASS.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
NUM_GPUS=${NUM_GPUS:-8}
OUTROOT=${OUTROOT:-./probe_run_$(date -u +%Y%m%d_%H%M%S)}
MAX_BETWEEN_CV=${MAX_BETWEEN_CV:-2.0}
MAX_WITHIN_CV=${MAX_WITHIN_CV:-1.0}

mkdir -p "$OUTROOT"
PROBE_DIR="${OUTROOT}/probe"

banner() { echo; echo "############ $* ############"; echo; }

command -v vllm >/dev/null || { echo "ERROR: vllm not on PATH (activate its venv)" >&2; exit 1; }

banner "1/3 node inventory: $(hostname) at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm \
  --format=csv,noheader
VISIBLE=$(nvidia-smi --list-gpus | wc -l)
echo "visible GPUs: ${VISIBLE} (need ${NUM_GPUS})"
if [[ "$VISIBLE" -lt "$NUM_GPUS" ]]; then
  echo "ERROR: only ${VISIBLE} GPUs visible; set NUM_GPUS or get a whole node" >&2
  exit 1
fi
echo "--- processes currently on the GPUs (should be empty) ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
  || echo "(none)"

banner "2/3 GPU probe (${PROBE_REPEATS:-3} repeats x ${NUM_GPUS} GPUs)"
MODEL="$MODEL" NUM_GPUS="$NUM_GPUS" REPEATS="${PROBE_REPEATS:-3}" \
OUTDIR="$PROBE_DIR" \
INPUT_LEN="${PROBE_INPUT_LEN:-8192}" \
OUTPUT_LEN="${PROBE_OUTPUT_LEN:-256}" \
NUM_PROMPTS="${PROBE_NUM_PROMPTS:-32}" \
CONCURRENCY="${PROBE_CONCURRENCY:-8}" \
WARMUP_PROMPTS="${PROBE_WARMUP_PROMPTS:-16}" \
MAX_MODEL_LEN="${PROBE_MAX_MODEL_LEN:-16384}" \
MAX_NUM_BATCHED_TOKENS="${PROBE_MAX_NUM_BATCHED_TOKENS:-8192}" \
SEED=0 \
  bash "${HERE}/gpu_probe.sh"
PROBE_RC=$?
if [[ "$PROBE_RC" -ne 0 ]]; then
  echo "ERROR: probe failed (rc=${PROBE_RC}); see ${PROBE_DIR}/serve_gpu*.log" >&2
  exit 1
fi

banner "3/3 probe verdict"
python3 "${HERE}/analyze_gpu_variance.py" "${PROBE_DIR}/measurements.jsonl" \
  2>&1 | sed 's/^/    /' || true
echo
PER_CARD_FLAG=""
[[ "${PER_CARD:-0}" == "1" ]] && PER_CARD_FLAG="--per-card"
python3 "${HERE}/probe_gate.py" "${PROBE_DIR}/measurements.jsonl" \
  --metric duration \
  --max-between-cv "$MAX_BETWEEN_CV" \
  --max-within-cv "$MAX_WITHIN_CV" $PER_CARD_FLAG | tee "${OUTROOT}/probe_gate.txt"
exit "${PIPESTATUS[0]}"
