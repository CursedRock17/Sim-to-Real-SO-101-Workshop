# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate 1 -- trajectory-quality scoring for oracle sim episode curation.

Runs on episodes that PASSED Gate 0. Where Gate 0 is a hard validity filter, Gate 1 is
a *scorer*: it rates how clean/deliberate the motion is, so the kept corpus resembles
good human demos (smooth, no hesitation) rather than bang-bang oracle motion -- the
"consistency/quality" lever the LeRobot folding writeup found dominated.

Pure-numpy, no Isaac/torch. One call per episode over the recorded joint trajectory:
    evaluate_trajectory(traj, fps, spec, ref_len=None, block_to_goal=None) -> Gate1Result

Metrics (all reported; only some gate):
  * smoothness   -- log dimensionless jerk (LDLJ; higher = smoother) + peak |accel|/|jerk|
  * idle         -- leading/trailing static frames (-> trim) and interior idle fraction
  * duration     -- frame count, and ratio to a reference length if given
  * progress     -- if block->goal distance per frame is given, fraction of frames that
                    make progress and the largest backtrack (dithering detector)
Hard FLAGs are the absolute, meaningful ones (too_short, excessive_idle, jerky,
low_progress). The composite quality_score (0-100) is for ranking / batch curation
(Gate 2). Absolute smoothness caps are hard to set a priori -- log them on the first
batch and calibrate `Gate1Spec` from the KEEP distribution.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Gate1Spec:
    fps: float = 30.0
    idle_speed: float = 2.0          # ||joint vel|| (units/s) below this = static frame
    max_idle_frac: float = 0.35      # interior idle fraction above this -> "excessive_idle"
    min_len: int = 20                # episodes shorter than this -> "too_short"
    len_band: tuple = (0.5, 2.0)     # ok range for (len / ref_len) when ref given
    jerk_cap: float = 4.0e5          # peak |jerk| (units/s^3) above this -> "jerky" (tune!)
    min_progress_frac: float = 0.5   # fraction of frames making goal progress (if provided)
    max_backtrack_frac: float = 0.25 # largest backward jump / total progress (if provided)
    # quality_score weights (need not sum to 1; normalized internally)
    w_smooth: float = 0.4
    w_idle: float = 0.25
    w_duration: float = 0.15
    w_progress: float = 0.2
    pass_score: float = 55.0         # quality_score below this -> verdict "FLAG"


@dataclass
class Gate1Result:
    verdict: str                 # "PASS" or "FLAG"
    quality_score: float         # 0..100, for ranking
    flags: List[str] = field(default_factory=list)
    # smoothness
    ldlj: float = 0.0            # log dimensionless jerk (higher = smoother)
    peak_accel: float = 0.0
    peak_jerk: float = 0.0
    # idle / duration
    n_frames: int = 0
    lead_idle: int = 0
    trail_idle: int = 0
    idle_frac: float = 0.0
    len_ratio: float = 1.0
    trim_start: int = 0          # suggested [trim_start, trim_end) keep-window
    trim_end: int = 0
    # progress (0 if not provided)
    progress_frac: float = 1.0
    max_backtrack: float = 0.0

    @property
    def flagged(self) -> bool:
        return self.verdict == "FLAG"


def _ldlj(traj: np.ndarray, dt: float) -> tuple:
    """Log dimensionless jerk over all moving joints + raw peak accel/jerk.

    Amplitude-normalized so it's comparable across durations/joints. Returns
    (ldlj, peak_accel, peak_jerk). Joints that don't move are skipped.
    """
    T = traj.shape[0]
    if T < 4:
        return 0.0, 0.0, 0.0
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    peak_accel = float(np.abs(acc).max()) if acc.size else 0.0
    peak_jerk = float(np.abs(jerk).max()) if jerk.size else 0.0

    dur = (T - 1) * dt
    ldljs = []
    for j in range(traj.shape[1]):
        amp = float(traj[:, j].max() - traj[:, j].min())
        if amp < 1e-6:
            continue  # joint didn't move; smoothness undefined
        integ = float(np.sum(jerk[:, j] ** 2) * dt)
        if integ <= 0:
            continue
        dlj = (dur ** 5 / amp ** 2) * integ      # dimensionless jerk
        ldljs.append(-math.log(dlj + 1e-12))     # higher (less negative) = smoother
    ldlj = float(np.mean(ldljs)) if ldljs else 0.0
    return ldlj, peak_accel, peak_jerk


def _idle(traj: np.ndarray, dt: float, idle_speed: float) -> tuple:
    """Leading/trailing static frame counts and interior idle fraction."""
    T = traj.shape[0]
    if T < 2:
        return 0, 0, 0.0
    speed = np.linalg.norm(np.diff(traj, axis=0) / dt, axis=1)   # (T-1,)
    static = speed < idle_speed
    lead = int(np.argmax(~static)) if (~static).any() else T - 1
    trail = int(np.argmax(~static[::-1])) if (~static).any() else T - 1
    interior = static[lead:len(static) - trail]
    idle_frac = float(interior.mean()) if interior.size else 0.0
    return lead, trail, idle_frac


def _progress(block_to_goal: np.ndarray) -> tuple:
    """Fraction of frames that reduce goal distance, and the largest backtrack
    (as a fraction of total forward progress). Dithering -> low frac / high backtrack."""
    d = np.asarray(block_to_goal, dtype=float)
    if d.size < 2:
        return 1.0, 0.0
    step = np.diff(d)                       # negative = progress toward goal
    forward = float(-step[step < 0].sum())  # total distance closed
    backward = float(step[step > 0].sum())  # total distance lost
    frac = float((step < 0).mean())
    backtrack = backward / (forward + 1e-9)
    return frac, backtrack


def evaluate_trajectory(
    traj: Sequence[Sequence[float]],
    *,
    spec: Gate1Spec = Gate1Spec(),
    ref_len: Optional[float] = None,
    block_to_goal: Optional[Sequence[float]] = None,
) -> Gate1Result:
    """Score one episode's joint trajectory.

    Args:
        traj: (T, D) recorded per-frame joint positions (state or action, normalized
            units). D is typically 6 (5 arm + gripper).
        spec: thresholds + score weights.
        ref_len: reference episode length (e.g. batch median) for the duration ratio.
            If None, duration only checks ``min_len``.
        block_to_goal: optional (T,) block-centre distance to the box goal per frame,
            for the monotonic-progress metric.

    Returns:
        Gate1Result with metrics, flags, a 0..100 quality_score, and a suggested
        [trim_start, trim_end) keep-window (leading/trailing idle removed).
    """
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"traj must be (T, D); got shape {arr.shape}")
    T = arr.shape[0]
    dt = 1.0 / spec.fps
    flags: List[str] = []

    ldlj, peak_accel, peak_jerk = _ldlj(arr, dt)
    lead, trail, idle_frac = _idle(arr, dt, spec.idle_speed)
    trim_start, trim_end = lead, T - trail
    len_ratio = float(T / ref_len) if ref_len else 1.0

    if block_to_goal is not None:
        progress_frac, max_backtrack = _progress(np.asarray(block_to_goal))
    else:
        progress_frac, max_backtrack = 1.0, 0.0

    # ---- hard flags (absolute, meaningful) ----
    if T < spec.min_len:
        flags.append("too_short")
    if idle_frac > spec.max_idle_frac:
        flags.append("excessive_idle")
    if peak_jerk > spec.jerk_cap:
        flags.append("jerky")
    if ref_len and not (spec.len_band[0] <= len_ratio <= spec.len_band[1]):
        flags.append("duration_outlier")
    if block_to_goal is not None and (
        progress_frac < spec.min_progress_frac or max_backtrack > spec.max_backtrack_frac
    ):
        flags.append("low_progress")

    # ---- subscores in 0..1 (higher = better) ----
    s_smooth = float(np.clip(1.0 - peak_jerk / spec.jerk_cap, 0.0, 1.0))
    s_idle = float(np.clip(1.0 - idle_frac / max(spec.max_idle_frac, 1e-6), 0.0, 1.0))
    if ref_len:
        # 1.0 at ratio 1, decaying outside the band
        s_dur = float(np.clip(1.0 - abs(len_ratio - 1.0), 0.0, 1.0))
    else:
        s_dur = 1.0 if T >= spec.min_len else 0.0
    s_prog = float(np.clip(progress_frac, 0.0, 1.0)) if block_to_goal is not None else 1.0

    wsum = spec.w_smooth + spec.w_idle + spec.w_duration + spec.w_progress
    quality = 100.0 * (
        spec.w_smooth * s_smooth + spec.w_idle * s_idle
        + spec.w_duration * s_dur + spec.w_progress * s_prog
    ) / wsum

    # "too_short" is disqualifying regardless of score
    verdict = "PASS" if (quality >= spec.pass_score and "too_short" not in flags) else "FLAG"

    return Gate1Result(
        verdict=verdict,
        quality_score=round(quality, 1),
        flags=flags,
        ldlj=round(ldlj, 3),
        peak_accel=round(peak_accel, 1),
        peak_jerk=round(peak_jerk, 1),
        n_frames=int(T),
        lead_idle=int(lead),
        trail_idle=int(trail),
        idle_frac=round(idle_frac, 3),
        len_ratio=round(len_ratio, 3),
        trim_start=int(trim_start),
        trim_end=int(trim_end),
        progress_frac=round(progress_frac, 3),
        max_backtrack=round(max_backtrack, 3),
    )


def gate1_fields() -> list:
    """CSV columns for the Gate 1 portion of the per-episode scorecard."""
    return [
        "g1_verdict", "quality_score", "flags",
        "ldlj", "peak_accel", "peak_jerk",
        "n_frames", "lead_idle", "trail_idle", "idle_frac", "len_ratio",
        "trim_start", "trim_end", "progress_frac", "max_backtrack",
    ]


def gate1_row(result: Gate1Result) -> dict:
    """Flatten a Gate1Result to a scorecard row (flags joined for CSV)."""
    d = asdict(result)
    d["g1_verdict"] = d.pop("verdict")
    d["flags"] = "|".join(result.flags)
    return {k: d[k] for k in gate1_fields()}
