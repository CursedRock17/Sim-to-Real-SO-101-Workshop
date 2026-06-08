#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SO-101 Real-Robot GR00T Policy Evaluation Script

Combines eval_so100's clean direct-LeRobot runtime with eval_so101's
quality-of-life features, with no SO101Control dependency.

Features:
    • Direct LeRobot robot control (make_robot_from_config)
    • Cosine-interpolated movement to initial/home poses
    • Thread-safe hardware access
    • Rerun real-time visualization (cameras + joints + actions)
    • Per-joint trajectory plotting on exit
    • Passive mode (torque off, observe only)
    • Graceful shutdown with KeyboardInterrupt

Usage:
    python3 eval_so101_merged.py \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras='{ \\
            "ego": {"type": "opencv", "index": 0, "width": 640, "height": 480}, \\
            "external_D455": {"type": "opencv", "index": 2, "width": 640, "height": 480}}' \\
        --policy-host=0.0.0.0 \\
        --policy-port=5555 \\
        --rerun=true \\
        --plot=true \\
        --timeout=60
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
from pprint import pformat
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import draccus
from gr00t.policy.server_client import PolicyClient

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np

# Optional rerun — not required if --rerun=false
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


# =============================================================================
# Constants
# =============================================================================

# SO-101 joint ordering — must match training dataset
JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Camera keys — must match modality.json video keys
CAMERA_KEYS = ["ego", "external_D455"]

CONTROL_HZ = 30.0

# Actual SO-101 poses from so101_control.py
INITIAL_POSE = {
    "shoulder_pan.pos": -6.3602,
    "shoulder_lift.pos": -51.9492,
    "elbow_flex.pos": 16.7276,
    "wrist_flex.pos": 89.2483,
    "wrist_roll.pos": -51.8499,
    "gripper.pos": 0.0000,
}

HOME_POSE = {
    "shoulder_pan.pos": -6.2835,
    "shoulder_lift.pos": -91.4407,
    "elbow_flex.pos": 93.1444,
    "wrist_flex.pos": 69.1434,
    "wrist_roll.pos": -51.9027,
    "gripper.pos": 0.0707,
}


# =============================================================================
# Rerun Helpers
# =============================================================================

def init_rerun(session_name: Optional[str] = None, window_size: str = "1280x720") -> None:
    """Initialize and spawn the Rerun viewer for live rollout visualization."""
    if not HAS_RERUN:
        logging.warning("rerun-sdk not installed — skipping visualization")
        return

    if session_name is None:
        session_name = f"so101_eval_{uuid.uuid4().hex[:8]}"

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    rr.init(session_name)
    memory_limit = os.getenv("RERUN_MEMORY_LIMIT", "10%")
    port = 9876
    subprocess.Popen([
        "rerun",
        "--port", str(port),
        "--memory-limit", memory_limit,
        "--window-size", window_size,
        "--expect-data-soon", "true",
    ])
    rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")


def kill_rerun() -> None:
    """Kill any running Rerun viewer processes."""
    subprocess.run(["pkill", "-f", "rerun"], capture_output=True)


def log_rerun_data(
    observation: Optional[Dict[str, Any]] = None,
    action: Optional[Dict[str, Any]] = None,
) -> None:
    """Log observation images/scalars and action scalars to the Rerun viewer."""
    if not HAS_RERUN:
        return

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = f"observation.{k}"
            if isinstance(v, (float, int, np.integer, np.floating)):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                # Transpose CHW → HWC if needed
                if (
                    v.ndim == 3
                    and v.shape[0] in (1, 3, 4)
                    and v.shape[-1] not in (1, 3, 4)
                ):
                    v = np.transpose(v, (1, 2, 0))
                if v.ndim >= 2:
                    rr.log(key, rr.Image(v), static=True)
                else:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = f"action.{k}"
            if isinstance(v, (float, int, np.integer, np.floating)):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                for i, vi in enumerate(v.flatten()):
                    rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


# =============================================================================
# Motion Helpers
# =============================================================================

def move_to_pose(
    robot: Robot,
    target_pose: Dict[str, float],
    duration: float = 3.0,
    fps: float = CONTROL_HZ,
    use_rerun: bool = False,
    hw_lock: Optional[threading.Lock] = None,
) -> None:
    """
    Smoothly move the robot to a target pose using cosine interpolation
    (ease-in-ease-out) over the given duration.
    """
    if hw_lock:
        with hw_lock:
            current_obs = robot.get_observation()
    else:
        current_obs = robot.get_observation()

    keys = list(target_pose.keys())
    start = np.array([current_obs[k] for k in keys])
    end = np.array([target_pose[k] for k in keys])

    num_steps = int(duration * fps)
    for i in range(1, num_steps + 1):
        t = i / num_steps
        alpha = (1 - np.cos(t * np.pi)) / 2  # cosine ease-in-ease-out
        interp = start + alpha * (end - start)
        action = {k: float(interp[j]) for j, k in enumerate(keys)}

        if hw_lock:
            with hw_lock:
                robot.send_action(action)
        else:
            robot.send_action(action)

        time.sleep(1.0 / fps)

        if use_rerun:
            if hw_lock:
                with hw_lock:
                    obs = robot.get_observation()
            else:
                obs = robot.get_observation()
            log_rerun_data(observation=obs, action=action)


# =============================================================================
# Adapter: Robot Obs ↔ GR00T VLA
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.
    GR00T Policy Server expects obs shaped (batch=1, time=1, ...).
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class So101Adapter:
    """
    Adapter between raw robot observations and GR00T VLA input/output format.

    Handles:
        • Camera frames → obs["video"]
        • Joint state → obs["state"] (single_arm + gripper)
        • Language instruction → obs["language"]
        • Batch/time dimensions
        • Model action chunk → per-timestep robot motor commands
    """

    def __init__(
        self,
        policy_client: PolicyClient,
        joint_keys: List[str] = JOINT_KEYS,
        camera_keys: List[str] = CAMERA_KEYS,
    ):
        self.policy = policy_client
        self.robot_state_keys = joint_keys
        self.camera_keys = camera_keys

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """Convert raw robot observation into structured GR00T VLA input."""
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state
        state = np.array(
            [obs[k] for k in self.robot_state_keys], dtype=np.float32
        )
        model_obs["state"] = {
            "single_arm": state[:5],   # (5,)
            "gripper": state[5:6],     # (1,)
        }

        # (3) Language instruction
        model_obs["language"] = {
            "annotation.human.task_description": obs["lang"],
        }

        # (4) Add (B=1, T=1) dimensions
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """
        Extract timestep t from model action chunk.

        chunk["single_arm"]: (B, T, 5)
        chunk["gripper"]:    (B, T, 1)
        """
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]         # (1,)
        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)

        return {
            name: float(full[i])
            for i, name in enumerate(self.robot_state_keys)
        }

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """Query policy and return list of motor commands (one per timestep)."""
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Plotting
# =============================================================================

def save_eval_plot(
    obs_buffer: List[Dict[str, float]],
    action_buffer: List[Dict[str, float]],
    joint_keys: List[str],
    out_dir: Path = Path("outputs/plots"),
) -> None:
    """Save per-joint trajectory plot of observations vs actions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for ax, key in zip(axes.flatten(), joint_keys):
        obs_vals = [s.get(key, 0.0) for s in obs_buffer]
        act_vals = [s.get(key, 0.0) for s in action_buffer]
        ax.plot(obs_vals, label="obs", linewidth=0.8)
        ax.plot(act_vals, label="action", linewidth=0.8, alpha=0.8)
        ax.set_title(key)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.supxlabel("Timestep")
    fig.supylabel("Joint value (deg)")
    fig.suptitle(f"Eval joint trajectory — {ts}")
    fig.tight_layout()

    png_path = out_dir / f"eval_{ts}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logging.info("Saved plot to %s", png_path)


# =============================================================================
# Config
# =============================================================================

@dataclass
class EvalConfig:
    """Command-line configuration for real-robot policy evaluation."""

    robot: Optional[RobotConfig] = None
    policy_host: str = "0.0.0.0"
    policy_port: int = 5555
    action_horizon: int = 16
    lang_instruction: str = "Pick up vial and place it in the target location"
    play_sounds: bool = False
    timeout: int = 60          # Max seconds before auto-stop (0 = unlimited)
    rerun: bool = False        # Enable rerun visualization
    passive_mode: bool = False # Torque off, observe only
    plot: bool = False         # Save trajectory plot on exit


# =============================================================================
# Main Eval Loop
# =============================================================================

@draccus.wrap()
def eval(cfg: EvalConfig):
    """Main entry point for real-robot policy evaluation."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Thread-safe lock for hardware access
    hw_lock = threading.Lock()

    # ----- 1. Initialize Robot -----
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    log_say("Robot connected", cfg.play_sounds, blocking=True)

    # ----- 2. Rerun Visualization -----
    if cfg.rerun:
        init_rerun()

    # ----- 3. Initial Pose / Passive Mode -----
    if cfg.passive_mode:
        robot.bus.disable_torque()
        logging.info("Passive mode — torque disabled. Move the arm freely.")
    else:
        logging.info("Moving to initial pose...")
        move_to_pose(robot, INITIAL_POSE, use_rerun=cfg.rerun, hw_lock=hw_lock)
        logging.info("At initial pose")

    # ----- 4. Initialize Policy -----
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So101Adapter(policy_client)
    log_say(
        f'Policy ready: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    # ----- 5. Rerun Logging Thread -----
    log_stop = threading.Event()
    log_action: Dict[str, Any] = {}
    log_thread = None

    if cfg.rerun:
        def rerun_worker():
            while not log_stop.wait(1.0 / CONTROL_HZ):
                try:
                    with hw_lock:
                        obs = robot.get_observation()
                    log_rerun_data(observation=obs, action=log_action)
                except Exception as e:
                    logging.warning("Rerun log error: %s", e)

        log_thread = threading.Thread(target=rerun_worker, daemon=True)
        log_thread.start()

    # ----- 6. Buffers for plotting -----
    obs_buffer: List[Dict[str, float]] = []
    action_buffer: List[Dict[str, float]] = []

    # ----- 7. Control Loop -----
    start_time = time.time()
    step_count = 0

    try:
        while True:
            # Timeout check
            if cfg.timeout > 0 and (time.time() - start_time) > cfg.timeout:
                logging.info("Timeout reached (%ds). Stopping.", cfg.timeout)
                break

            # Get observation (thread-safe)
            with hw_lock:
                obs = robot.get_observation()
            obs["lang"] = cfg.lang_instruction

            if cfg.passive_mode:
                if cfg.rerun:
                    log_rerun_data(observation=obs, action={})
                time.sleep(1.0 / CONTROL_HZ)
                continue

            # Query policy
            logging.debug("Querying policy server...")
            actions = policy.get_action(obs)
            logging.debug("Got %d actions from policy", len(actions))

            # Execute action chunk
            for i, action_dict in enumerate(actions[: cfg.action_horizon]):
                tic = time.time()

                # Send to robot (thread-safe)
                with hw_lock:
                    robot.send_action(action_dict)
                step_count += 1

                # Update rerun action for logging thread
                if cfg.rerun:
                    log_action.update(action_dict)

                # Record for plotting
                if cfg.plot:
                    with hw_lock:
                        step_obs = robot.get_observation()
                    obs_buffer.append(
                        {k: float(step_obs[k]) for k in JOINT_KEYS}
                    )
                    action_buffer.append(
                        {k: v for k, v in action_dict.items()}
                    )

                # Rate limiting
                elapsed = time.time() - tic
                sleep_time = (1.0 / CONTROL_HZ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — shutting down gracefully...")
    except Exception as e:
        logging.error("Control loop crashed: %s", e, exc_info=True)

    finally:
        # ----- 8. Cleanup -----

        # Stop rerun logging thread
        if cfg.rerun:
            log_stop.set()
            if log_thread is not None:
                log_thread.join(timeout=2.0)

        # Save trajectory plot
        if cfg.plot and obs_buffer:
            save_eval_plot(obs_buffer, action_buffer, JOINT_KEYS)

        # Return to home pose (skip in passive mode)
        if cfg.passive_mode:
            logging.info("Final pose:")
            obs = robot.get_observation()
            for k in JOINT_KEYS:
                logging.info("  %s: %.4f", k, obs[k])
        else:
            logging.info("Returning to home pose...")
            try:
                move_to_pose(
                    robot, HOME_POSE, use_rerun=cfg.rerun, hw_lock=hw_lock
                )
            except Exception as e:
                logging.warning("Could not return to home: %s", e)

        # Disconnect
        robot.disconnect()

        if cfg.rerun:
            kill_rerun()

        logging.info(
            "Eval complete — %d action steps in %.1fs",
            step_count,
            time.time() - start_time,
        )


if __name__ == "__main__":
    eval()
