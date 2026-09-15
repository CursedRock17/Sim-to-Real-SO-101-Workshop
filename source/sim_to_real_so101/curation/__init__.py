# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-curation gates for oracle sim episode generation.

Gate 0 = generation-time hard gates (plan validity, clean approach, grasp, task
success) -- a KEEP/DROP validity filter. Gate 1 = trajectory-quality scoring on the
survivors (smoothness, idle, duration, progress) so the corpus resembles clean,
deliberate demos. Both are pure-numpy, no Isaac/torch, reusable by any generator and
unit-testable off-GPU.
"""
from .gate0 import (
    BoxTaskSpec,
    Gate0Result,
    block_in_box,
    evaluate_episode,
    scorecard_fields,
    ScorecardWriter,
)
from .gate1 import (
    Gate1Spec,
    Gate1Result,
    evaluate_trajectory,
    gate1_fields,
    gate1_row,
)


def combined_fields() -> list:
    """Unified per-episode scorecard columns (Gate 0 + Gate 1)."""
    return scorecard_fields() + gate1_fields()


__all__ = [
    # gate 0
    "BoxTaskSpec",
    "Gate0Result",
    "block_in_box",
    "evaluate_episode",
    "scorecard_fields",
    "ScorecardWriter",
    # gate 1
    "Gate1Spec",
    "Gate1Result",
    "evaluate_trajectory",
    "gate1_fields",
    "gate1_row",
    "combined_fields",
]
