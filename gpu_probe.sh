#!/bin/bash
set -uo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
NUM_GPUS=${NUM_GPUS:-8}
REPEATS=${REPEATS:-4}
OUTDIR=${OUTDIR:-./probe_out}
BASE_PORT=${BASE_PORT:-9000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-64}
CONCURRENCY=${CONCURRENCY:-8}
WARMUP_PROMPTS=${WARMUP_PROMPTS:-16}
SEED=${SEED:-0}

mkdir -p "$OUTDIR"
RESULTS="${OUTDIR}/measurements.jsonl"
: >"$RESULTS"

rm -f "${OUTDIR}"/row_g*_r*.jsonl "${OUTDIR}"/bench_g*_r*.json \
      "${OUTDIR}/node_sampler.csv"

if (( MAX_MODEL_LEN < INPUT_LEN + OUTPUT_LEN )); then
  echo "ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} < INPUT_LEN+OUTPUT_LEN=$((INPUT_LEN + OUTPUT_LEN));" \
       "every request would be rejected" >&2
  exit 1
fi

declare -a SERVER_PIDS=()
SAMPLER_PID=""
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}" "$SAMPLER_PID"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "=== host: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,power.limit \
  --format=csv,noheader
echo "=== visible GPUs: $(nvidia-smi --list-gpus | wc -l) (need ${NUM_GPUS}) ==="

gpu_state() {
  local idx=$1
  nvidia-smi --id="$idx" \
    --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
}

wait_for_server() {
  local url=$1 pid=$2
  for _ in $(seq 1 300); do
    curl -s "${url}/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "ERROR: server exited" >&2; return 1; }
    sleep 5
  done
  echo "ERROR: server not ready" >&2; return 1
}

STARTUP_PARALLEL=${STARTUP_PARALLEL:-1}
# Use the experiment's serve args, or a model that needs e.g. --max-num-seqs dies here.
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}

launch_server() {
  local g=$1 port=$((BASE_PORT + $1))
  echo "--- starting server on GPU ${g}, port ${port} ---"
  CUDA_VISIBLE_DEVICES="$g" VLLM_SERVER_DEV_MODE=1 \
  vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --no-enable-prefix-caching \
    $EXTRA_SERVE_ARGS \
    --port "$port" >"${OUTDIR}/serve_gpu${g}.log" 2>&1 &
  SERVER_PIDS[$g]=$!
}

_t0=$SECONDS
if [[ "$STARTUP_PARALLEL" == "1" ]]; then
  launch_server 0
  wait_for_server "http://127.0.0.1:${BASE_PORT}" "${SERVER_PIDS[0]}" || exit 1
  echo "--- GPU 0 ready (page cache warm); starting the other" \
       "$((NUM_GPUS - 1)) concurrently ---"
  for g in $(seq 1 $((NUM_GPUS - 1))); do launch_server "$g"; done
  for g in $(seq 1 $((NUM_GPUS - 1))); do
    wait_for_server "http://127.0.0.1:$((BASE_PORT + g))" "${SERVER_PIDS[$g]}" \
      || exit 1
  done
else
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    launch_server "$g"
    wait_for_server "http://127.0.0.1:$((BASE_PORT + g))" "${SERVER_PIDS[$g]}" \
      || exit 1
  done
fi
echo "=== all ${NUM_GPUS} servers ready in $((SECONDS - _t0))s" \
     "(STARTUP_PARALLEL=${STARTUP_PARALLEL}) ==="

_t1=$SECONDS
for g in $(seq 0 $((NUM_GPUS - 1))); do
  echo "--- warming GPU ${g} ---"
  vllm bench serve \
    --backend vllm --base-url "http://127.0.0.1:$((BASE_PORT + g))" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --ignore-eos --seed "$SEED" \
    --num-prompts "$WARMUP_PROMPTS" --max-concurrency "$CONCURRENCY" \
    --disable-tqdm >"${OUTDIR}/warmup_gpu${g}.log" 2>&1 || true
done

echo "=== warmup done in $((SECONDS - _t1))s ==="

MEASURE_PARALLEL=${MEASURE_PARALLEL:-0}

row_file() { echo "${OUTDIR}/row_g${1}_r${2}.jsonl"; }

sample_node() {
  local out="${OUTDIR}/node_sampler.csv"
  echo "ts,gpu,clocks_sm,power_w,temp_c,throttle,util_pct" >"$out"
  while true; do
    nvidia-smi --query-gpu=index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed "s/^/$(date -u +%H:%M:%S),/" | tr -d ' ' >>"$out"
    sleep 5
  done
}

measure_one() {
  local g=$1 r=$2
  local port=$((BASE_PORT + g))
  local before after rc fname
  before=$(gpu_state "$g")
  fname="bench_g${g}_r${r}.json"
  vllm bench serve \
    --backend vllm --base-url "http://127.0.0.1:${port}" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --ignore-eos --seed "$SEED" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,99" \
    --disable-tqdm \
    --save-result --result-dir "$OUTDIR" --result-filename "$fname" \
    >"${OUTDIR}/log_g${g}_r${r}.log" 2>&1
  rc=$?
  after=$(gpu_state "$g")
  python3 - "$OUTDIR" "$fname" "$g" "$r" "$rc" "$before" "$after" \
    "$MEASURE_PARALLEL" "$NUM_GPUS" "$(cut -d' ' -f1-3 /proc/loadavg)" \
    <<'PY' >"$(row_file "$g" "$r")"
import json, os, sys
outdir, fname, gpu, rep, rc = sys.argv[1:6]
before = sys.argv[6].split(",") if len(sys.argv) > 6 else []
after = sys.argv[7].split(",") if len(sys.argv) > 7 else []
measure_parallel, num_gpus, loadavg = sys.argv[8], sys.argv[9], sys.argv[10]

def field(parts, i):
    return parts[i] if len(parts) > i else None

row = {"gpu": int(gpu), "repeat": int(rep), "rc": int(rc),
       "clocks_sm_before": field(before, 0),
       "power_before": field(before, 1),
       "temp_before": field(before, 2),
       "throttle_before": field(before, 3),
       "clocks_sm_after": field(after, 0),
       "power_after": field(after, 1),
       "temp_after": field(after, 2),
       "throttle_after": field(after, 3),
       # Tagged so sequential and parallel runs are never pooled.
       "measure_parallel": measure_parallel == "1",
       "n_concurrent": int(num_gpus) if measure_parallel == "1" else 1,
       "loadavg": loadavg}
path = os.path.join(outdir, fname)
try:
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, list):
        d = d[0]
    for k in ("duration", "request_throughput", "output_throughput",
              "total_token_throughput", "mean_ttft_ms", "median_ttft_ms",
              "mean_tpot_ms", "median_tpot_ms", "mean_e2el_ms",
              "median_e2el_ms", "completed"):
        if k in d:
            row[k] = d[k]
except Exception as exc:  # noqa: BLE001
    row["error"] = repr(exc)
print(json.dumps(row))
PY
  echo "gpu=${g} repeat=${r} rc=${rc} ${before}-> ${after}"
}

_t2=$SECONDS
if [[ "$MEASURE_PARALLEL" == "1" ]]; then
  sample_node & SAMPLER_PID=$!
  echo "=== node sampler started (pid ${SAMPLER_PID}) ==="
fi

for r in $(seq 1 "$REPEATS"); do
  if [[ "$MEASURE_PARALLEL" == "1" ]]; then
    echo "--- repeat ${r}: launching ${NUM_GPUS} benchmarks CONCURRENTLY ---"
    MPIDS=()
    for g in $(seq 0 $((NUM_GPUS - 1))); do measure_one "$g" "$r" & MPIDS+=($!); done
    for p in "${MPIDS[@]}"; do wait "$p"; done
  else
    for g in $(seq 0 $((NUM_GPUS - 1))); do measure_one "$g" "$r"; done
  fi
done

[[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null

cat "${OUTDIR}"/row_g*_r*.jsonl >"$RESULTS" 2>/dev/null

ZERO=$(python3 -c "
import json,sys
n=0
for l in open('$RESULTS'):
    r=json.loads(l)
    if not r.get('completed'): n+=1
print(n)" 2>/dev/null || echo 0)
if [[ "${ZERO:-0}" -gt 0 ]]; then
  echo "ERROR: ${ZERO} row(s) completed 0 requests -- the workload never ran." >&2
  echo "       Check MAX_MODEL_LEN vs INPUT_LEN+OUTPUT_LEN and the serve logs." >&2
  exit 1
fi

echo "=== measurements done in $((SECONDS - _t2))s ==="
echo "=== phase timing: startup $((_t1 - _t0))s | warmup $((_t2 - _t1))s |" \
     "measure $((SECONDS - _t2))s | TOTAL ${SECONDS}s ==="
echo "=== done; ${RESULTS} ==="
wc -l "$RESULTS"
