#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal GR00T N1.6 inference server for cloud (Brev) deployment.

Runs the heavy VLA forward pass on a cloud GPU (e.g. an L4) and serves actions
over ZMQ, so a CPU-only laptop can drive the real SO-101 via PolicyClient.
Container-agnostic: depends only on the `gr00t` package (no sim_to_real / attn /
rerun), so it drops onto any GR00T container.

    ┌── laptop (CPU) ──┐   ZMQ tcp://*:5555   ┌── Brev L4 (GPU) ──┐
    │ so101_eval.py    │ ◀─────────────────▶  │  THIS server       │
    │ arm + cameras    │      actions          │  Gr00tPolicy       │
    └──────────────────┘                       └────────────────────┘

Usage (on the Brev instance, inside the GR00T container):
    python3 run_gr00t_server.py \\
        --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \\
        --port 5555
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model_path: str = "CursedRock17/so101_teleop_vials_sim_and_real_finetune"
    """HF model id or local path. If a repo has checkpoint-* subfolders, either
    point directly at one (.../checkpoint-XXXXX) or use --auto-checkpoint."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Must match how the model was fine-tuned (SO-101 finetunes use NEW_EMBODIMENT)."""

    port: int = 5555
    """PolicyServer binds tcp://*:<port> (all interfaces) so the SSH tunnel can reach it."""

    device: str = "cuda"
    strict: bool = True

    auto_checkpoint: bool = False
    """Download the repo and auto-select the latest checkpoint-* if config.json isn't at root."""


def _resolve_model_path(model_path: str, auto_checkpoint: bool) -> str:
    """GR00T needs the folder that holds config.json; some repos nest it under
    checkpoint-XXXXX. If asked, download the repo and pick the latest checkpoint."""
    if not auto_checkpoint:
        return model_path
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(model_path)) if "/" in model_path and not Path(model_path).exists() else Path(model_path)
    if (local / "config.json").exists():
        return str(local)
    ckpts = sorted(local.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else str(local)


def main(cfg: ServerConfig) -> None:
    # Load the policy onto the GPU, then block serving action requests over ZMQ.
    resolved = _resolve_model_path(cfg.model_path, cfg.auto_checkpoint)
    logger.info("Loading Gr00tPolicy from %s on %s", resolved, cfg.device)
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=resolved,
        device=cfg.device,
        strict=cfg.strict,
    )
    logger.info("Policy loaded; serving on tcp://*:%d (Ctrl-C to stop)", cfg.port)
    PolicyServer.start_server(policy, port=cfg.port)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
