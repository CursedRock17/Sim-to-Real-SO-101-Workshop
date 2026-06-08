#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
GR00T N1.6 Inference Server with Real-Time Attention Visualization.

Drop-in replacement for run_gr00t_server.py that wraps the Gr00tPolicy with
GR00TN1d6Attention to capture attention maps and stream overlays to rerun
on every get_action() call.

Architecture:
    ┌──────────────┐     ZMQ      ┌────────────────────────────────┐
    │  eval client  │ ◀──────────▶ │  THIS server                   │
    │ (robot loop)  │   actions    │  Gr00tPolicy + GR00TN1d6Attn   │
    └──────────────┘              │  → logs overlays to rerun       │
                                  └────────────────────────────────┘

Usage:
    # Terminal 1 — start the attention server
    python3 run_gr00t_server_attn.py \\
        --model-path /tmp/so101_groot/checkpoint-XXXX \\
        --embodiment-tag NEW_EMBODIMENT \\
        --port 5556 \\
        --camera-keys external_D455 ego \\
        --probe  # first run: discover PATCH_GRID_HW, then set it

    # Terminal 2 — run eval (same as before, just point to this server's port)
    python3 eval_so101_attn.py \\
        --policy-port=5556 \\
        --rerun=false  # server handles rerun now
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer

from lerobot_attention_visualizer import GR00TN1d6Attention
from lerobot_attention_visualizer.visualizer.overlay import (
    log_attention_overlay,
    patch_heatmap_to_image,
    rollout_to_patch_heatmap,
)
from lerobot_attention_visualizer.policies.cross_attention import (
    vision_importance_to_grids,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure rerun binary is on PATH
_local_bin = str(Path.home() / ".local" / "bin")
if _local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _local_bin + os.pathsep + os.environ.get("PATH", "")


@dataclass
class AttnServerConfig:
    """Configuration for GR00T N1.6 inference server with attention visualization."""

    # --- Model ---
    model_path: str = ""
    """Path to the fine-tuned checkpoint or HF model ID."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag for the model."""

    device: str = "cuda"
    """Device to run the model on."""

    # --- Server ---
    host: str = "0.0.0.0"
    """Host address for the ZMQ server."""

    port: int = 5556
    """Port number (use a different port from the non-attn server)."""

    strict: bool = False
    """Whether to enforce strict input/output validation."""

    # --- Attention Visualization ---
    camera_keys: List[str] = field(default_factory=lambda: ["external_D455", "ego"])
    """Camera key names matching your modality config (order matters)."""

    patch_grid_hw: Optional[Tuple[int, int]] = None
    """Per-camera vision-token patch grid (h, w). Run with --probe first to discover."""

    probe: bool = False
    """First run: print per-camera token count and exit (to discover patch_grid_hw)."""

    last_layer_only: bool = False
    """Only capture attention from the last transformer layer."""

    clip_percentile: float = 99.0
    """Percentile for clipping attention values."""

    suppress_outliers: bool = True
    """Suppress outlier attention values."""

    gamma: float = 2.5
    """Gamma correction for attention overlay."""

    colormap: str = "hot"
    """Colormap for attention overlay: 'hot', 'blue-green', 'viridis'."""

    rerun_session: str = "groot_n16_attention"
    """Rerun session name."""


class AttentionPolicyWrapper:
    """
    Wraps a Gr00tPolicy + GR00TN1d6Attention to capture attention on every
    get_action() call and log overlays to rerun.

    Implements the same interface as Gr00tPolicy so PolicyServer can use it.
    """

    def __init__(
        self,
        policy,  # Gr00tPolicy
        viz,     # GR00TN1d6Attention
        camera_keys: List[str],
        probe: bool = False,
        clip_percentile: float = 99.0,
        suppress_outliers: bool = True,
        gamma: float = 2.5,
        colormap: str = "hot",
    ):
        self.policy = policy
        self.viz = viz
        self.camera_keys = camera_keys
        self.probe = probe
        self.clip_percentile = clip_percentile
        self.suppress_outliers = suppress_outliers
        self.gamma = gamma
        self.colormap = colormap
        self._probed = False
        self._step = 0

    def get_action(
        self, observation: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Forward get_action to the real policy, then log attention overlay.
        """
        # Call the real policy (attention hooks capture maps automatically)
        result = self.policy.get_action(observation=observation, options=options)

        # --- Probe mode: discover patch grid and exit ---
        if self.probe and not self._probed:
            self._probed = True
            m = getattr(self.viz, "_vision_mask", None)
            if m is not None:
                total = int(m.sum())
                per_cam = total // len(self.camera_keys)
                logger.info(
                    f"[PROBE] Vision tokens: total={total}, "
                    f"per_camera={per_cam}"
                )
                logger.info(
                    f"[PROBE] Set --patch-grid-hw to (h, w) where h*w == {per_cam}"
                )
                logger.info(
                    f"[PROBE] Hint: for 640x480 images with SigLIP2, "
                    f"common grids are (16, 22), (15, 20), (12, 16)"
                )
            else:
                logger.warning("[PROBE] No vision mask captured — "
                               "check that GR00TN1d6Attention hooks are active")
            logger.info("[PROBE] Exiting probe mode. Set --probe=false and "
                        "--patch-grid-hw=H,W to run with overlays.")
            return result

        # --- Extract raw camera frames from observation for overlay ---
        obs_images = self._extract_camera_frames(observation)

        if obs_images:
            # Skip step 0: first forward pass is a warmup that doesn't
            # populate the encoder attention hooks.
            if self._step == 0:
                # Drain any partial buffers from warmup
                try:
                    self.viz._capture.drain_rollouts(
                        last_layer_only=self.viz._last_layer_only
                    )
                except Exception:
                    pass
                if self.viz._cross is not None:
                    try:
                        self.viz._cross.drain()
                    except Exception:
                        pass
                logger.info("Skipping attention overlay on warmup step (step 0)")
                self._step += 1
                return result

            camera_keys = self.viz.camera_keys()
            n = len(camera_keys)
            prefix = "attention"
            kw = dict(
                clip_percentile=self.clip_percentile,
                suppress_outliers=self.suppress_outliers,
                gamma=self.gamma,
            )

            # --- Section 1: Encoder self-attention (best-effort) ---
            try:
                rollouts = self.viz._capture.drain_rollouts(
                    last_layer_only=self.viz._last_layer_only
                )
                if rollouts and len(rollouts) == n:
                    for cam_key, rollout in zip(camera_keys, rollouts):
                        image = obs_images.get(cam_key)
                        if image is None:
                            continue
                        if rollout.numel() == 0:
                            continue
                        try:
                            patch_heat = rollout_to_patch_heatmap(rollout)
                            heat = patch_heatmap_to_image(
                                patch_heat, target_hw=image.shape[:2], **kw
                            )
                            log_attention_overlay(
                                f"{prefix}/{cam_key}/encoder", image, heat,
                                colormap=self.colormap,
                            )
                        except (ValueError, IndexError):
                            continue  # skip encoder for this camera
            except Exception as e:
                if self._step <= 2:
                    logger.debug(f"Encoder overlay skipped (step {self._step}): {e}")

            # --- Section 2: Action cross-attention (the key signal) ---
            try:
                importance = (
                    self.viz._cross.drain()
                    if self.viz._cross is not None
                    else None
                )
                if importance is not None and self.viz._vision_mask is not None:
                    grids = vision_importance_to_grids(
                        importance, self.viz._vision_mask, n,
                        grid_hw=self.viz._patch_grid_hw,
                    )
                    if len(grids) == n:
                        for cam_key, grid in zip(camera_keys, grids):
                            image = obs_images.get(cam_key)
                            if image is None:
                                continue
                            heat = patch_heatmap_to_image(
                                grid, target_hw=image.shape[:2], **kw
                            )
                            log_attention_overlay(
                                f"{prefix}/{cam_key}/action", image, heat,
                                colormap=self.colormap,
                            )
            except Exception as e:
                if self._step <= 3:
                    logger.warning(f"Action overlay failed (step {self._step}): {e}")

            self._step += 1
            if self._step % 30 == 0 or self._step == 2:
                logger.info(f"Attention overlay: {self._step} frames streamed")

        return result

    def _extract_camera_frames(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract raw HWC uint8 camera frames from the GR00T observation dict.

        The observation comes in GR00T format:
            observation["video"][cam_key] -> shape (B, T, H, W, C) or (B, T, C, H, W)
        We need to extract the first batch/time step as HWC uint8.
        """
        obs_images = {}

        video_data = observation.get("video", {})
        if not video_data:
            return obs_images

        for cam_key in self.camera_keys:
            if cam_key not in video_data:
                continue

            frame = video_data[cam_key]

            # Unwrap batch and time dimensions
            if isinstance(frame, np.ndarray):
                while frame.ndim > 3:
                    frame = frame[0]

                # Handle CHW -> HWC
                if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                    frame = np.transpose(frame, (1, 2, 0))

                # Ensure uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                obs_images[cam_key] = frame

            elif isinstance(frame, list):
                # Nested lists from msgpack: unwrap to get the ndarray
                inner = frame
                while isinstance(inner, list) and len(inner) > 0:
                    inner = inner[0]
                if isinstance(inner, np.ndarray):
                    obs_images[cam_key] = inner

        return obs_images

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.policy.reset(options=options)

    def _log_viz_state(self):
        """Dump ALL attributes on the attention visualizer for debugging."""
        viz = self.viz
        logger.info("=== Attention Visualizer State (full dump) ===")
        for attr in sorted(dir(viz)):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(viz, attr)
            except Exception:
                continue
            if callable(val) and not hasattr(val, "shape"):
                continue  # skip methods

            if isinstance(val, np.ndarray):
                logger.info(f"  {attr}: ndarray shape={val.shape} dtype={val.dtype}")
            elif hasattr(val, "shape"):  # torch tensor
                logger.info(f"  {attr}: tensor shape={val.shape} dtype={val.dtype}")
            elif isinstance(val, (list, tuple)):
                if len(val) > 0 and hasattr(val[0], "shape"):
                    shapes = [v.shape for v in val[:5]]
                    logger.info(f"  {attr}: {type(val).__name__}[{len(val)}] shapes={shapes}")
                elif len(val) == 0:
                    logger.info(f"  {attr}: {type(val).__name__}[EMPTY]")
                else:
                    logger.info(f"  {attr}: {type(val).__name__}[{len(val)}]")
            elif isinstance(val, dict):
                summary = {}
                for k, v in list(val.items())[:10]:
                    if hasattr(v, "shape"):
                        summary[k] = f"shape={v.shape}"
                    elif isinstance(v, (list, tuple)):
                        summary[k] = f"{type(v).__name__}[{len(v)}]"
                    else:
                        summary[k] = type(v).__name__
                logger.info(f"  {attr}: dict keys={summary}")
            elif isinstance(val, (bool, int, float, str)):
                logger.info(f"  {attr}: {val!r}")
            else:
                logger.info(f"  {attr}: {type(val).__name__}")
        logger.info("=== End Visualizer State ===")

    def _log_sub_object(self, attr_name: str):
        """Dump all attributes of a sub-object on the visualizer."""
        obj = getattr(self.viz, attr_name, None)
        if obj is None:
            logger.info(f"=== viz.{attr_name}: None ===")
            return
        logger.info(f"=== viz.{attr_name} ({type(obj).__name__}) ===")
        for attr in sorted(dir(obj)):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val) and not hasattr(val, "shape"):
                continue

            if isinstance(val, np.ndarray):
                logger.info(f"  .{attr}: ndarray shape={val.shape} dtype={val.dtype}")
            elif hasattr(val, "shape"):
                logger.info(f"  .{attr}: tensor shape={val.shape} dtype={val.dtype}")
            elif isinstance(val, (list, tuple)):
                if len(val) > 0 and hasattr(val[0], "shape"):
                    shapes = [v.shape for v in val[:5]]
                    logger.info(f"  .{attr}: {type(val).__name__}[{len(val)}] shapes={shapes}")
                elif len(val) == 0:
                    logger.info(f"  .{attr}: {type(val).__name__}[EMPTY]")
                else:
                    logger.info(f"  .{attr}: {type(val).__name__}[{len(val)}]")
            elif isinstance(val, dict):
                summary = {}
                for k, v in list(val.items())[:10]:
                    if hasattr(v, "shape"):
                        summary[k] = f"shape={v.shape}"
                    elif isinstance(v, (list, tuple)):
                        summary[k] = f"{type(v).__name__}[{len(v)}]"
                    else:
                        summary[k] = type(v).__name__
                logger.info(f"  .{attr}: dict {summary}")
            elif isinstance(val, (bool, int, float, str)):
                logger.info(f"  .{attr}: {val!r}")
            else:
                logger.info(f"  .{attr}: {type(obj).__name__}")
        logger.info(f"=== End viz.{attr_name} ===")

    def get_modality_config(self):
        return self.policy.get_modality_config()


def main(config: AttnServerConfig):
    logger.info("Starting GR00T N1.6 Attention Server...")
    logger.info(f"  Model:       {config.model_path}")
    logger.info(f"  Embodiment:  {config.embodiment_tag}")
    logger.info(f"  Device:      {config.device}")
    logger.info(f"  Cameras:     {config.camera_keys}")
    logger.info(f"  Patch grid:  {config.patch_grid_hw}")
    logger.info(f"  Probe mode:  {config.probe}")
    logger.info(f"  Port:        {config.port}")

    # --- 1. Load the real Gr00tPolicy ---
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    logger.info("Gr00tPolicy loaded")

    # --- 2. Initialize rerun ---
    try:
        import rerun as rr
        import subprocess
        import uuid

        session_name = config.rerun_session or f"groot_attn_{uuid.uuid4().hex[:8]}"
        rr.init(session_name)
        port_rr = 9876
        subprocess.Popen([
            "rerun",
            "--port", str(port_rr),
            "--memory-limit", os.getenv("RERUN_MEMORY_LIMIT", "10%"),
            "--window-size", "1280x720",
            "--expect-data-soon", "true",
        ])
        rr.connect_grpc(f"rerun+http://127.0.0.1:{port_rr}/proxy")
        logger.info(f"Rerun viewer started on port {port_rr}")
    except Exception as e:
        logger.warning(f"Could not start rerun viewer: {e}")

    # --- 3. Wrap policy with attention visualizer ---
    viz = GR00TN1d6Attention(
        policy,
        camera_keys=config.camera_keys,
        last_layer_only=config.last_layer_only,
        patch_grid_hw=config.patch_grid_hw,
    )

    attn_wrapper = AttentionPolicyWrapper(
        policy=policy,
        viz=viz,
        camera_keys=config.camera_keys,
        probe=config.probe,
        clip_percentile=config.clip_percentile,
        suppress_outliers=config.suppress_outliers,
        gamma=config.gamma,
        colormap=config.colormap,
    )

    # --- 4. Start the server with attention hooks active ---
    #
    # We enter the viz context manager to install attention hooks,
    # then run the server loop inside it.
    with viz:
        logger.info("Attention hooks installed — starting ZMQ server...")

        server = PolicyServer(
            policy=attn_wrapper,
            host=config.host,
            port=config.port,
        )

        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Shutting down attention server...")

    # Cleanup rerun
    try:
        subprocess.run(["pkill", "-f", "rerun"], capture_output=True)
    except Exception:
        pass

    logger.info("Attention server stopped")


if __name__ == "__main__":
    config = tyro.cli(AttnServerConfig)
    main(config)

