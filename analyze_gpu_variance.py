#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decompose GPU-to-GPU vs run-to-run variance from `gpu_variance_probe.sh`.

Answers three questions, in the order they change what we do:

1. Do the GPUs in a node differ *systematically*? If between-GPU spread is
   large relative to within-GPU spread, then any A/B split across two cards is
   confounded and every pair must share one card.
2. What is the run-to-run noise floor? An APC effect smaller than this is not
   measurable at the repeat count we are using, no matter how it is analysed.
3. Which card is quietest? Use it for the headline numbers.

The between/within comparison is a one-way ANOVA in spirit: `F`-like ratio of
between-group to within-group mean squares. We report it as a ratio rather
than a p-value because with 8 groups and a handful of repeats the exact null
distribution is not the interesting part -- the effect size relative to our
measurement target is.

Usage:
    python3 analyze_gpu_variance.py /path/to/measurements.jsonl [--metric M]
"""

import argparse
import json
import statistics as stats
from collections import defaultdict

# Lower-is-better metrics; the rest are higher-is-better throughputs.
LOWER_BETTER = {
    "duration", "mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms",
    "median_tpot_ms", "mean_e2el_ms", "median_e2el_ms",
}
THROTTLE_OK = {"0x0000000000000000", "0x0000000000000001", "", "None", None}


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cv(values):
    """Coefficient of variation in percent; 0.0 when undefined."""
    if len(values) < 2:
        return 0.0
    mean = stats.fmean(values)
    if mean == 0:
        return 0.0
    return 100.0 * stats.stdev(values) / abs(mean)


def analyse(rows, metric):
    per_gpu = defaultdict(list)
    for r in rows:
        if r.get("rc") not in (0, None) or metric not in r:
            continue
        per_gpu[r["gpu"]].append(float(r[metric]))

    if not per_gpu:
        raise SystemExit(f"no usable rows for metric {metric!r}")

    gpus = sorted(per_gpu)
    print(f"metric: {metric}  "
          f"({'lower' if metric in LOWER_BETTER else 'higher'} is better)")
    print(f"{'gpu':>4} {'n':>3} {'mean':>12} {'stdev':>10} {'cv%':>7} "
          f"{'min':>12} {'max':>12}")
    for g in gpus:
        vals = per_gpu[g]
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{g:>4} {len(vals):>3} {stats.fmean(vals):>12.2f} {sd:>10.3f} "
              f"{cv(vals):>7.2f} {min(vals):>12.2f} {max(vals):>12.2f}")

    gpu_means = [stats.fmean(per_gpu[g]) for g in gpus]
    grand = stats.fmean(gpu_means)
    between_cv = cv(gpu_means)
    within_cvs = [cv(per_gpu[g]) for g in gpus]
    within_cv = stats.fmean(within_cvs)

    # Spread across cards, as a percentage of the grand mean -- the quantity
    # that directly contaminates a cross-GPU A/B comparison.
    spread_pct = 100.0 * (max(gpu_means) - min(gpu_means)) / abs(grand)

    print()
    print(f"between-GPU CV : {between_cv:6.2f}%   "
          f"(spread across cards: {spread_pct:.2f}% of mean)")
    print(f"within-GPU CV  : {within_cv:6.2f}%   (mean over cards)")
    if within_cv > 0:
        print(f"ratio          : {between_cv / within_cv:6.2f}x  "
              "(>1 means card identity matters more than run-to-run noise)")

    # Quietest card, tie-broken toward the faster one so the headline config is
    # both stable and not pathologically slow.
    def rank(g):
        speed = stats.fmean(per_gpu[g])
        return (cv(per_gpu[g]), speed if metric in LOWER_BETTER else -speed)

    best = min(gpus, key=rank)
    print()
    print(f"recommended GPU: {best}  "
          f"(cv {cv(per_gpu[best]):.2f}%, mean {stats.fmean(per_gpu[best]):.2f})")

    print()
    print("--- implications ---")
    print(f"* pair noapc/align on ONE card: a cross-card split would carry a "
          f"{spread_pct:.2f}% systematic offset.")
    print(f"* effects below ~{2 * within_cv:.2f}% (2x within-GPU CV) are not "
          "resolvable at this repeat count.")
    throttled = [r for r in rows
                 if str(r.get("throttle_after")) not in THROTTLE_OK]
    if throttled:
        by_gpu = defaultdict(int)
        for r in throttled:
            by_gpu[r.get("gpu")] += 1
        detail = ", ".join(f"gpu{g}x{n}" for g, n in sorted(by_gpu.items()))
        print(f"* WARNING: {len(throttled)} run(s) report a throttle reason "
              f"beyond idle/none ({detail}); treat those as suspect.")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="measurements.jsonl from gpu_variance_probe.sh")
    ap.add_argument("--metric", default="output_throughput",
                    help="metric to analyse (default: output_throughput)")
    ap.add_argument("--all-metrics", action="store_true",
                    help="report every metric present, not just one")
    args = ap.parse_args()

    rows = load(args.path)
    if not rows:
        raise SystemExit(f"no rows parsed from {args.path}")
    print(f"loaded {len(rows)} measurements from {args.path}\n")

    if args.all_metrics:
        keys = sorted({k for r in rows for k in r
                       if isinstance(r.get(k), (int, float))
                       and k not in ("gpu", "repeat", "rc")})
        for k in keys:
            analyse(rows, k)
            print("\n" + "=" * 66 + "\n")
    else:
        analyse(rows, args.metric)


if __name__ == "__main__":
    main()
