# GPU node health probe

Before any benchmark runs, this probe checks whether a multi-GPU node can be trusted.
Each card is benchmarked under the same fixed vLLM workload. A PASS/FAIL gate then
checks three things:

- **Throttling.** Hardware throttling (HW slowdown, HW thermal, power brake) fails the
  node. Software power caps and software thermal limits are only reported.
- **Between-GPU spread.** The node fails if its cards differ systematically by more
  than 2% (CV of the per-card means).
- **Run-to-run noise.** The node fails if any card's repeat-to-repeat CV is above 1%.
  This CV sets the smallest effect you can resolve, which is 2x the worst CV.

On a PASS the gate also names the quietest card (`PROBE_BEST_GPU`).

Why it matters: the effects we measure are single-digit percentages. A throttling card,
or cards that differ by more than the effect itself, would turn the experiment into a
measurement of the hardware.

## Files

| file | role |
|---|---|
| `run_probe.sh` | entry point: node inventory, then probe, then report, then gate. Exits 0 on PASS |
| `gpu_probe.sh` | starts one `vllm serve` per GPU, warms each one up, and benchmarks the cards **one at a time** in round-robin order. Clocks, power, temperature and throttle reasons are sampled before and after every run |
| `analyze_gpu_variance.py` | readable report: between-GPU vs. within-GPU variance, noise floor, quietest card |
| `probe_gate.py` | PASS/FAIL verdict plus `PROBE_*=` lines you can parse from a shell |

The only dependencies are `vllm` (for `vllm serve` and `vllm bench serve`), `nvidia-smi`,
`curl` and Python 3. The Python scripts use only the standard library.

## Usage

```bash
source /path/to/vllm-venv/bin/activate
./run_probe.sh                                   # 8 GPUs, Qwen/Qwen3.5-9B
MODEL=Qwen/Qwen3.5-0.8B NUM_GPUS=4 ./run_probe.sh
python3 probe_gate.py probe_run_*/probe/measurements.jsonl   # re-judge existing data
```

The default workload is 32 prompts of 8192 input and 256 output tokens at concurrency
8, after 16 warmup prompts, repeated 3 times per GPU. Prefix caching is off, because
the probe measures the hardware, not cache policy. Everything can be overridden with
`PROBE_INPUT_LEN`, `PROBE_OUTPUT_LEN`, `PROBE_NUM_PROMPTS`, `PROBE_CONCURRENCY`,
`PROBE_REPEATS`, `PROBE_MAX_MODEL_LEN`, `MAX_BETWEEN_CV` and `MAX_WITHIN_CV`. To pass
serve arguments through to every server, set `EXTRA_SERVE_ARGS`, for example
`--max-num-seqs 64`.

Output goes to `probe_run_<timestamp>/`. It contains `probe/measurements.jsonl` (one
row per GPU per repeat), the serve, warmup and benchmark logs, and `probe_gate.txt`.

## Things to know

- **Use the whole node.** On dense H100 SXM nodes the cards share a power budget, so
  another job on GPU 7 can slow down GPU 0. Reserve the node exclusively (on LSF:
  `bsub -x -n 8 -gpu num=8:mode=exclusive_process`).
- **The probe runs with TP=1.** A model too big for one card will fail to load. Set
  `MODEL` to a small model; the gate is judging the silicon, not the model.
- **A short probe can look noisy when it isn't.** Run-to-run jitter on this hardware is
  about 100 ms whatever the probe length, so a 1% CV limit can't be met by probes under
  about 10 s. The gate detects this case: when absolute jitter is at most 250 ms, it
  reports "PROBE TOO SHORT" as an advisory instead of failing the node. Healthy nodes
  showed at most 134 ms of jitter, and genuinely bad nodes at least 501 ms.
- **`PER_CARD=1`** lets a minority of noisy cards through, as an advisory. Use it only
  when each comparison stays on a single card. The between-GPU check still applies.

## Provenance

Copied from the `paper-apc` repo (`benchmarks/apc_hybrid/`), branch `dev/hannes`,
2026-09-23. The copy is based on commit `d17f457` **plus uncommitted working-tree
changes** in `bluevela_gpu_probe.sh` (EXTRA_SERVE_ARGS forwarding) and `probe_gate.py`
(the short-probe jitter rescue). Two changes were made for sharing:
`bluevela_gpu_probe.sh` was renamed to `gpu_probe.sh`, and its default `OUTDIR` now
points to `./probe_out` instead of a cluster home directory. `run_probe.sh` is a
standalone copy of stages 1–3 of `bluevela_exclusive_experiment.sh`.
