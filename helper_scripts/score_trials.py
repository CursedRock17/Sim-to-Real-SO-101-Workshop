#!/usr/bin/env python3
"""Aggregate SO-101 pick-place eval trials into fundamental results.

Reads one or more trial CSVs (from eval_sim.py, or hand-filled from real-arm runs)
and prints, per checkpoint: N, success rate with a 95% Wilson interval, the
approach->grasp->place funnel, and a failure-mode histogram. With >1 checkpoint it
also prints a side-by-side comparison so you can see whether extra training steps
helped or overfit.

CSV schema (shared by sim and manual scoring):
  checkpoint,trial,block_x,block_y,approached,grasped,placed,success,failure_mode,maxlift,notes
The 0/1 columns are approached, grasped, placed, success. Everything else is optional
context. 'success' is trusted if present; otherwise it's derived as grasped AND placed.

Usage:
  python3 helper_scripts/score_trials.py outputs/eval/*.csv
  python3 helper_scripts/score_trials.py real_trials.csv sim_trials.csv
"""
import csv
import glob
import math
import sys
from collections import Counter, defaultdict


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion (robust at small N)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def load_rows(paths):
    rows = []
    for pat in paths:
        for path in sorted(glob.glob(pat)) or [pat]:
            try:
                with open(path, newline="") as f:
                    for r in csv.DictReader(f):
                        if any((v or "").strip() for v in r.values()):
                            rows.append(r)
            except FileNotFoundError:
                print(f"[warn] no such file: {path}", file=sys.stderr)
    return rows


def summarize(name, rows):
    n = len(rows)
    appr = sum(_truthy(r.get("approached", "")) for r in rows)
    grasp = sum(_truthy(r.get("grasped", "")) for r in rows)
    place = sum(_truthy(r.get("placed", "")) for r in rows)
    succ = sum(
        _truthy(r["success"]) if (r.get("success") or "").strip() != ""
        else (_truthy(r.get("grasped", "")) and _truthy(r.get("placed", "")))
        for r in rows
    )
    lo, hi = wilson_ci(succ, n)
    fails = Counter(
        (r.get("failure_mode") or "").strip()
        for r in rows if not _truthy(r.get("success", "")) and (r.get("failure_mode") or "").strip()
    )

    print(f"\n=== {name} ===")
    print(f"  N trials         : {n}")
    print(f"  SUCCESS          : {succ}/{n} = {succ / n:.0%}   (95% CI {lo:.0%}-{hi:.0%})"
          if n else "  SUCCESS          : n/a")
    print("  funnel (of N):")
    for label, k in (("approached", appr), ("grasped", grasp), ("placed", place)):
        bar = "#" * round(20 * k / n) if n else ""
        print(f"    {label:<11}: {k:>3}/{n}  {bar}")
    # stage conversion: where does it leak?
    if n:
        print("  stage drop-off:")
        print(f"    reach->grasp : {grasp}/{appr} ({(grasp / appr) if appr else 0:.0%})")
        print(f"    grasp->place : {place}/{grasp} ({(place / grasp) if grasp else 0:.0%})")
    if fails:
        print("  failure modes:")
        for mode, k in fails.most_common():
            print(f"    {mode:<14}: {k}")
    return {"n": n, "succ": succ, "rate": (succ / n if n else 0.0), "ci": (lo, hi)}


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    rows = load_rows(argv)
    if not rows:
        print("no rows found.")
        return 1

    by_ckpt = defaultdict(list)
    for r in rows:
        by_ckpt[(r.get("checkpoint") or "unlabeled").strip()].append(r)

    summarize("ALL TRIALS", rows)
    stats = {}
    if len(by_ckpt) > 1:
        for ckpt in sorted(by_ckpt):
            stats[ckpt] = summarize(ckpt, by_ckpt[ckpt])
        print("\n=== comparison ===")
        print(f"  {'checkpoint':<24}{'N':>5}{'success':>10}{'  95% CI':>16}")
        for ckpt in sorted(stats, key=lambda c: -stats[c]["rate"]):
            s = stats[ckpt]
            print(f"  {ckpt:<24}{s['n']:>5}{s['rate']:>9.0%}"
                  f"   {s['ci'][0]:.0%}-{s['ci'][1]:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
