#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal GR00T N1.6 inference server for cloud (Brev) deployment.

Runs the heavy VLA forward pass on a cloud GPU (e.g. an L4) and serves actions
over ZMQ, so a CPU-only laptop can drive the real SO-101 via PolicyClient.
Container-agnostic: depends only on the `gr00t` package (no sim_to_real / attn /
rerun), so it drops onto any GR00T container.

    ┌── laptop (CPU) ──┐   ZMQ over SSH      ┌── Brev L4 (GPU) ──┐
    │ so101_eval.py    │ ◀─────────────────▶  │  THIS server       │
    │ arm + cameras    │      actions          │  Gr00tPolicy       │
    └──────────────────┘                       └────────────────────┘

Usage (on the Brev instance, inside the GR00T container):
    python3 run_gr00t_server.py \\
        --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \\
        --checkpoint checkpoint-30000 \\
        --port 5555
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import tyro
from gr00t.data.embodiment_tags import EmbodimentTag
from huggingface_hub import list_repo_files, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model_path: str = "CursedRock17/so101_teleop_vials_sim_and_real_finetune"
    """HF model id or local model directory."""

    checkpoint: str | None = None
    """Exact subdirectory to serve, for example checkpoint-30000."""

    revision: str | None = None
    """Optional HF commit hash to make the model selection reproducible."""

    offline: bool = False
    """Resolve the model from the existing HF cache without downloading."""

    download_only: bool = False
    """Fetch the selected inference files and print their directory, then exit."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Must match how the model was fine-tuned (SO-101 finetunes use NEW_EMBODIMENT)."""

    port: int = 5555

    host: str = "127.0.0.1"
    """Bind to loopback for access through an SSH tunnel."""

    device: str = "cuda"
    strict: bool = True

    auto_checkpoint: bool = False
    """Select the latest checkpoint only if no root model exists; prefer --checkpoint."""


def _resolve_model_path(
    model_path: str,
    auto_checkpoint: bool = False,
    checkpoint: str | None = None,
    revision: str | None = None,
    offline: bool = False,
) -> str:
    """Resolve one model, excluding other checkpoints and training state."""
    if checkpoint is not None and not re.fullmatch(r"checkpoint-\d+", checkpoint):
        raise ValueError("checkpoint must have the form checkpoint-30000")
    if checkpoint and auto_checkpoint:
        raise ValueError("Choose --checkpoint or --auto-checkpoint, not both")
    local = Path(model_path).expanduser()
    if local.is_dir():
        if checkpoint:
            local /= checkpoint
        elif auto_checkpoint and not (local / "config.json").is_file():
            candidates = [
                path
                for path in local.glob("checkpoint-*")
                if re.fullmatch(r"checkpoint-\d+", path.name)
                and (path / "config.json").is_file()
            ]
            if candidates:
                local = max(candidates, key=lambda path: int(path.name.split("-")[1]))
    else:
        if auto_checkpoint:
            if offline:
                raise ValueError("Use an explicit --checkpoint or local path offline")
            files = list_repo_files(model_path, revision=revision)
            if "config.json" not in files:
                candidates = [
                    name.split("/")[0]
                    for name in files
                    if re.fullmatch(r"checkpoint-\d+/config.json", name)
                ]
                if not candidates:
                    raise ValueError("No model config found in the repository")
                checkpoint = max(candidates, key=lambda name: int(name.split("-")[1]))
        ignored = [
            "*.pt",
            "*.pth",
            "*.bin",
            "*trainer_state.json",
            "*wandb_config.json",
        ]
        if not checkpoint:
            ignored.append("checkpoint-*/*")
        local = Path(
            snapshot_download(
                model_path,
                revision=revision,
                allow_patterns=[f"{checkpoint}/*"] if checkpoint else None,
                ignore_patterns=ignored,
                local_files_only=offline,
            )
        )
        if checkpoint:
            local /= checkpoint
    if not (local / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in selected model directory: {local}")
    return str(local)


def main(cfg: ServerConfig) -> None:
    # Load the policy onto the GPU, then block serving action requests over ZMQ.
    resolved = _resolve_model_path(
        cfg.model_path, cfg.auto_checkpoint, cfg.checkpoint, cfg.revision, cfg.offline
    )
    if cfg.download_only:
        print(resolved)
        return

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    logger.info("Loading Gr00tPolicy from %s on %s", resolved, cfg.device)
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=resolved,
        device=cfg.device,
        strict=cfg.strict,
    )
    logger.info("Policy loaded; serving on tcp://%s:%d", cfg.host, cfg.port)
    PolicyServer.start_server(policy, port=cfg.port, host=cfg.host)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
