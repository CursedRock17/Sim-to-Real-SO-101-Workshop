# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate 0 -- generation-time hard gates for oracle sim episode curation.

One call per finished episode returns a KEEP/DROP verdict plus a scorecard of *why*.
Only KEEP episodes should be written to the dataset. Exploits the fact that sim has
ground truth (exact block pose / velocity), so curation is objective, not heuristic.

Pure-numpy, no Isaac/torch dependency -> unit-testable off-GPU and reusable by both
the FK oracle (topdown_pipeline) and a future MoveIt2 execute+record loop.

The gates, applied as a funnel (first failure = drop_reason):
    1. plan_ok        -- the planner produced a valid, executed trajectory
    2. approach_clean -- the block was not knocked/displaced before the grasp
    3. grasp_ok       -- the block was actually lifted off the surface and held
    4. success        -- the block ended settled inside the box (== the eval metric)

Signals are scalars the generator already tracks (or can with a few lines), so the
integration into the collection loop is minimal -- see ``evaluate_episode``.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class BoxTaskSpec:
    """Success geometry + gate thresholds for the box pick-place task.

    Defaults mirror the constants baked into topdown_pipeline / eval_sim, so a KEEP
    here means the same thing the evaluator scores as success. Keep these in sync with
    the env config (``table_env_cfg.BOX_POS``) if the scene changes.
    """

    box_xy: tuple = (0.10, 0.20)      # box centre (env frame), metres
    box_half: tuple = (0.06, 0.06)    # box interior half-extents (x, y)
    box_floor: float = 0.03           # settled block-centre z lower bound (inside box)
    box_rim: float = 0.12             # settled block-centre z upper bound (below rim)
    vel_settled: float = 0.05         # m/s: below this the block is at rest
    grasp_lift_m: float = 0.03        # block must rise >= this to count as grasped
    approach_disp_m: float = 0.015    # block XY may drift <= this before the grasp


@dataclass
class Gate0Result:
    """Verdict + scorecard for one episode."""

    verdict: str                 # "KEEP" or "DROP"
    drop_reason: str             # "" if KEEP, else the first failing gate
    plan_ok: bool
    approach_clean: bool
    grasp_ok: bool
    success: bool
    # measured signals (for the scorecard / threshold tuning)
    max_lift_m: float
    pre_grasp_disp_m: float
    final_xy_err_m: float
    final_speed: float

    @property
    def keep(self) -> bool:
        return self.verdict == "KEEP"


def block_in_box(pos: Sequence[float], lin_vel: Sequence[float], spec: BoxTaskSpec) -> bool:
    """True iff the block centre is settled inside the box footprint and below the rim.

    Identical predicate to topdown_pipeline.block_in_box / eval_sim -- generation and
    evaluation agree by construction.
    """
    speed = math.sqrt(sum(float(v) * float(v) for v in lin_vel))
    return (
        abs(float(pos[0]) - spec.box_xy[0]) < spec.box_half[0]
        and abs(float(pos[1]) - spec.box_xy[1]) < spec.box_half[1]
        and spec.box_floor < float(pos[2]) < spec.box_rim
        and speed < spec.vel_settled
    )


def evaluate_episode(
    *,
    plan_ok: bool,
    block_final_pos: Sequence[float],
    block_final_vel: Sequence[float],
    max_lift_m: float,
    pre_grasp_disp_m: float,
    spec: BoxTaskSpec = BoxTaskSpec(),
) -> Gate0Result:
    """Run Gate 0 on one finished episode.

    Args:
        plan_ok: the oracle produced and executed a valid plan (no planning failure,
            no early abort). The FK oracle already knows this; MoveIt returns it too.
        block_final_pos: (x, y, z) block centre at episode end, env frame, metres.
        block_final_vel: (vx, vy, vz) block linear velocity at episode end, m/s.
        max_lift_m: max height the block rose above its start z during the episode.
            The generator already tracks this (``maxlift``).
        pre_grasp_disp_m: max block XY displacement from its start, measured over the
            approach (before the gripper closes). ~0 for a clean top-down descent;
            large if the arm bumped the block. A few lines to track in the exec loop.
        spec: task geometry + thresholds.

    Returns:
        Gate0Result with verdict, first drop_reason, and the measured signals.
    """
    fx, fy, fz = (float(v) for v in block_final_pos)
    final_speed = math.sqrt(sum(float(v) * float(v) for v in block_final_vel))
    final_xy_err = math.hypot(fx - spec.box_xy[0], fy - spec.box_xy[1])

    approach_clean = float(pre_grasp_disp_m) <= spec.approach_disp_m
    grasp_ok = float(max_lift_m) >= spec.grasp_lift_m
    success = block_in_box(block_final_pos, block_final_vel, spec)

    # Funnel order: first failing gate names the drop reason.
    drop_reason = ""
    if not plan_ok:
        drop_reason = "plan_failed"
    elif not approach_clean:
        drop_reason = "knocked_block"
    elif not grasp_ok:
        drop_reason = "grasp_failed"
    elif not success:
        drop_reason = "place_failed"

    return Gate0Result(
        verdict="KEEP" if drop_reason == "" else "DROP",
        drop_reason=drop_reason,
        plan_ok=bool(plan_ok),
        approach_clean=approach_clean,
        grasp_ok=grasp_ok,
        success=success,
        max_lift_m=round(float(max_lift_m), 4),
        pre_grasp_disp_m=round(float(pre_grasp_disp_m), 4),
        final_xy_err_m=round(final_xy_err, 4),
        final_speed=round(final_speed, 4),
    )


def scorecard_fields() -> list:
    """CSV column order for the per-episode scorecard."""
    return [
        "episode_id", "verdict", "drop_reason",
        "plan_ok", "approach_clean", "grasp_ok", "success",
        "max_lift_m", "pre_grasp_disp_m", "final_xy_err_m", "final_speed",
    ]


class ScorecardWriter:
    """Append one scorecard row per episode (CSV), for audit + threshold tuning.

    Every episode is logged -- KEEP and DROP alike -- so you can see the yield and the
    failure-mode mix, and later re-tune thresholds without re-running the sim.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=scorecard_fields())
        self._w.writeheader()
        self.n_keep = 0
        self.n_total = 0

    def log(self, episode_id, result: Gate0Result) -> bool:
        row = {"episode_id": episode_id}
        row.update({k: v for k, v in asdict(result).items() if k in scorecard_fields()})
        self._w.writerow(row)
        self._fh.flush()
        self.n_total += 1
        self.n_keep += int(result.keep)
        return result.keep

    def close(self):
        summary = {"n_total": self.n_total, "n_keep": self.n_keep,
                   "yield": round(self.n_keep / self.n_total, 4) if self.n_total else 0.0}
        with open(self.path.rsplit(".", 1)[0] + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self._fh.close()
        return summary

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
