#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Go/no-go verdict on a GPU probe. See docs/DESIGN.md."""
"""Go/no-go gate on a GPU probe, for use before a real experiment.

`analyze_gpu_variance.py` produces the full human-readable characterisation.
This script answers the narrower question a runner needs to act on:

    is this node trustworthy right now, and which GPU should we use?

It prints a compact verdict, emits shell-parseable KEY=VALUE lines, and exits
non-zero if the node fails. The intended use is:

    python3 probe_gate.py probe/measurements.jsonl > gate.txt || exit 1
    eval "$(grep '^PROBE_' gate.txt)"       # PROBE_BEST_GPU, PROBE_VERDICT...

Why gate at all: the effects we publish are single-digit percentages. A card
that is thermally throttling, or a node whose cards differ by more than the
effect being measured, silently turns an experiment into a measurement of the
hardware. Today a contended run produced a spurious +17.6% "result" on the
0.8B Marconi job -- caught only because the slow runs drew LESS power. This
gate is the systematic version of that catch, applied before the expensive
run rather than after it.

Thresholds are deliberately loose enough to pass a healthy node (the
characterisation runs measured between-GPU CV ~0.9-1.0% and within-GPU CV
~0.07%) and tight enough that anything an experiment could mistake for a
real effect trips them.
"""

import argparse
import json
import statistics as stats
import sys

# string "Not Active". Both forms must be handled.
#
# our two existing probe datasets caught exactly that.
#
# that cost a whole scheduled run.
#
# comparison -- both arms sit under the same cap.
#
# fail outright.
THROTTLE_BITS = {
    0x0001: ("GpuIdle", False, False),                 # benign: idle when sampled
    0x0002: ("ApplicationsClocksSetting", False, False),
    0x0004: ("SwPowerCap", True, False),               # advisory (see above)
    0x0008: ("HwSlowdown", True, True),
    0x0010: ("SyncBoost", True, False),                # advisory
    0x0020: ("SwThermalSlowdown", True, False),        # advisory
    0x0040: ("HwThermalSlowdown", True, True),
    0x0080: ("HwPowerBrakeSlowdown", True, True),
    0x0100: ("DisplayClockSetting", False, False),     # benign: headless
}
SERIOUS_MASK = sum(b for b, (_n, rep, _f) in THROTTLE_BITS.items() if rep)
FATAL_NAMES = {n for _b, (n, _r, fatal) in THROTTLE_BITS.items() if fatal}


def throttle_flags(value):
    """Return the list of SERIOUS throttle reason names in a probe sample."""
    if value in (None, "", "N/A"):
        return []
    text = str(value).strip()
    if text.lower().replace(" ", "") in ("notactive", "active"):
        return []
    try:
        mask = int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return []
    return [name for bit, (name, rep, _f) in THROTTLE_BITS.items()
            if rep and (mask & bit)] if (mask & SERIOUS_MASK) else []


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def cv(values):
    if len(values) < 2:
        return 0.0
    m = stats.fmean(values)
    return 100 * stats.stdev(values) / m if m else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--metric", default="duration",
                    help="metric to judge on (default: duration)")
    ap.add_argument("--max-between-cv", type=float, default=2.0,
                    help="max between-GPU CV%% (default 2.0)")
    ap.add_argument("--per-card", action="store_true",
                    help="within-card paired designs only: a single noisy card degrades "
                         "one replicate, not the experiment. Report bad cards instead of "
                         "refusing the whole node, provided a majority of cards pass and "
                         "between-GPU CV is within tolerance.")
    # A CV limit is a RELATIVE test, but the machine's run-to-run noise is very close to
    # a FIXED ABSOLUTE quantity (~100 ms measured across probes from 3.9 s to 79 s). So the
    # CV test is really a test of probe DURATION: at ~100 ms jitter, no probe shorter than
    # ~10 s can pass a 1.0% limit however healthy the node is. Every one of the 12 runs
    # with a ~3.9 s probe failed and none passed, while ~15 s probes on the same cards
    # passed with 60-134 ms jitter. This threshold separates "probe too short" (clean band
    # tops out at 134 ms) from "node genuinely noisy" (bad band starts at 501 ms).
    ap.add_argument("--max-within-jitter-ms", type=float, default=250.0,
                    help="absolute run-to-run jitter (ms) still considered healthy when the "
                         "probe is too short for the relative CV limit to be satisfiable. "
                         "Only applied to duration-like metrics.")
    ap.add_argument("--max-within-cv", type=float, default=1.0,
                    help="max per-GPU run-to-run CV%% (default 1.0)")
    ap.add_argument("--min-repeats", type=int, default=2,
                    help="min measurements per GPU (default 2)")
    args = ap.parse_args()

    rows = load(args.path)
    if not rows:
        print("PROBE_VERDICT=FAIL")
        print("FAIL: no measurements found", file=sys.stderr)
        return 1

    problems = []
    advisories = []

    bad_rc = [r for r in rows if r.get("rc") not in (0, None)]
    if bad_rc:
        problems.append(f"{len(bad_rc)} run(s) with non-zero rc")

    throttled, throttle_reasons = [], set()
    for r in rows:
        reasons = (throttle_flags(r.get("throttle_before"))
                   + throttle_flags(r.get("throttle_after")))
        if reasons:
            throttled.append(r)
            throttle_reasons.update(reasons)
    if throttled:
        gpus = sorted({r["gpu"] for r in throttled})
        fatal = throttle_reasons & FATAL_NAMES
        msg = (f"{len(throttled)} throttled run(s) on GPU(s) {gpus} "
               f"[{', '.join(sorted(throttle_reasons))}]")
        if fatal:
            problems.append(msg + f" -- FATAL: {', '.join(sorted(fatal))}")
        else:
            # whether this matters. Surface it, don't fail on it.
            advisories.append(msg + " -- advisory (CV checks decide)")

    by_gpu = {}
    for r in rows:
        if r.get("rc") in (0, None) and args.metric in r:
            by_gpu.setdefault(r["gpu"], []).append(r[args.metric])

    if not by_gpu:
        print("PROBE_VERDICT=FAIL")
        print(f"FAIL: no usable '{args.metric}' values", file=sys.stderr)
        return 1

    thin = [g for g, v in by_gpu.items() if len(v) < args.min_repeats]
    if thin:
        problems.append(f"GPU(s) {sorted(thin)} have <{args.min_repeats} measurements")

    means = {g: stats.fmean(v) for g, v in by_gpu.items()}
    cvs = {g: cv(v) for g, v in by_gpu.items()}
    overall = stats.fmean(means.values())
    between_cv = cv(list(means.values()))
    worst_within = max(cvs.values())

    if between_cv > args.max_between_cv:
        problems.append(
            f"between-GPU CV {between_cv:.2f}% > {args.max_between_cv}%")
    # Absolute jitter, in the metric's own units (duration is in seconds -> ms here).
    jitter_ms = {g: cvs[g] * means[g] * 10.0 for g in cvs}
    worst_jitter = max(jitter_ms.values())
    is_duration = args.metric == "duration"
    # Shortest probe for which the CV limit is even reachable at the observed jitter.
    min_dur_needed = (worst_jitter / 1000.0) / (args.max_within_cv / 100.0) \
        if args.max_within_cv > 0 else float("inf")
    short_probe = is_duration and overall < min_dur_needed

    bad = sorted(g for g in cvs if cvs[g] > args.max_within_cv)
    good = sorted(g for g in cvs if g not in bad)
    if worst_within > args.max_within_cv:
        worst_gpu = max(cvs, key=cvs.get)
        msg = (f"GPU {worst_gpu} run-to-run CV {worst_within:.2f}% "
               f"> {args.max_within_cv}%")
        # In a within-card paired design each cell is compared against itself on the same
        # card, so one noisy card costs one replicate -- not the run. Ten gate refusals in
        # one session were single bad cards on otherwise-clean nodes.
        if short_probe and worst_jitter <= args.max_within_jitter_ms:
            # The node is fine; the PROBE is too short to resolve the limit. Failing here
            # blames the hardware and sends the job round the queue again, which cannot
            # help -- the outcome is set by probe duration, not by which node we land on.
            advisories.append(
                msg + f"; but absolute jitter is only {worst_jitter:.0f} ms "
                f"(<= {args.max_within_jitter_ms:.0f} ms) and this probe runs "
                f"{overall:.1f} s, under the {min_dur_needed:.1f} s needed for a "
                f"{args.max_within_cv}% CV to be reachable -- PROBE TOO SHORT, not a bad "
                "node. Lengthen the probe workload to tighten the floor.")
        elif args.per_card and len(good) > len(bad):
            advisories.append(msg + f"; --per-card: {len(bad)} of {len(cvs)} cards noisy "
                              f"(bad={','.join(map(str, bad))}), majority pass")
        else:
            problems.append(msg)
            if args.per_card:
                problems.append(
                    f"--per-card refused: {len(bad)} of {len(cvs)} cards noisy, "
                    "not a majority-good node")

    lower_is_better = "duration" in args.metric or "ttft" in args.metric \
        or "tpot" in args.metric or "e2el" in args.metric
    def rank(g):
        return (cvs[g], means[g] if lower_is_better else -means[g])
    best = min(by_gpu, key=rank)

    print(f"=== probe gate: {args.path} ===")
    print(f"metric: {args.metric}   ({len(rows)} measurements, "
          f"{len(by_gpu)} GPUs)")
    print(f"{'gpu':>4} {'n':>3} {'mean':>12} {'cv%':>8} {'vs node':>9}  flags")
    for g in sorted(by_gpu):
        dev = 100 * (means[g] - overall) / overall
        flags = []
        if any(r["gpu"] == g for r in throttled):
            flags.append("THROTTLED")
        if cvs[g] > args.max_within_cv:
            flags.append("NOISY")
        if g == best:
            flags.append("<- best")
        print(f"{g:>4} {len(by_gpu[g]):>3} {means[g]:>12.3f} {cvs[g]:>7.3f}% "
              f"{dev:>+8.2f}%  {' '.join(flags)}")
    print()
    print(f"between-GPU CV : {between_cv:6.3f}%  (limit {args.max_between_cv}%)")
    print(f"worst within-GPU CV: {worst_within:6.3f}%  "
          f"(limit {args.max_within_cv}%)")
    print(f"resolvable effect floor: ~{2 * worst_within:.2f}% "
          "(2x worst run-to-run CV) -- treat smaller effects as noise")
    # Absolute jitter is the duration-invariant number: report it so a short probe's
    # inflated percentage floor cannot be mistaken for an unstable machine.
    print(f"worst absolute jitter: {worst_jitter:7.0f} ms   "
          f"(healthy <= {args.max_within_jitter_ms:.0f} ms)")
    if is_duration:
        print(f"probe mean duration : {overall:7.2f} s    "
              f"({min_dur_needed:.1f} s needed for a {args.max_within_cv}% CV limit"
              f"{' -- PROBE TOO SHORT' if short_probe else ''})")
    print()

    verdict = "PASS" if not problems else "FAIL"
    if advisories:
        print("ADVISORIES (not failures):")
        for a in advisories:
            print(f"  - {a}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    elif advisories:
        # "cards within tolerance" would contradict the NOISY flags printed just above.
        # The node is usable, but say WHY rather than asserting something false.
        print("node usable: nothing disqualifying, but see the advisories above -- "
              "the percentage floor is inflated by probe length, not by the hardware")
    else:
        print("node looks trustworthy: no throttling, "
              "cards within tolerance, run-to-run spread small")
    print()
    print(f"PROBE_VERDICT={verdict}")
    print(f"PROBE_BEST_GPU={best}")
    print(f"PROBE_BETWEEN_CV={between_cv:.4f}")
    print(f"PROBE_WORST_WITHIN_CV={worst_within:.4f}")
    print(f"PROBE_EFFECT_FLOOR={2 * worst_within:.4f}")
    print(f"PROBE_WORST_JITTER_MS={worst_jitter:.1f}")
    print(f"PROBE_MEAN_DUR={overall:.4f}")
    print(f"PROBE_MIN_DUR_FOR_CV={min_dur_needed:.4f}")
    print(f"PROBE_SHORT={1 if short_probe else 0}")
    # Emitted unconditionally: knowing WHICH cards are noisy is useful even when the
    # verdict is unchanged, and it lets an analysis drop those cells after the fact.
    print(f"PROBE_BAD_GPUS={','.join(map(str, bad))}")
    print(f"PROBE_GOOD_GPUS={','.join(map(str, good))}")
    if good:
        floor_good = 2 * max(cvs[g] for g in good)
        print(f"PROBE_EFFECT_FLOOR_GOOD={floor_good:.4f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
