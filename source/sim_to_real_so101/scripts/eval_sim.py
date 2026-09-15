"""Closed-loop SIM evaluation of a fine-tuned GR00T policy in the SO-101 table task.

Runs the policy in the domain-randomized ``Lerobot-So101-Teleop-Table-Task`` env and
scores task success per episode, writing a per-trial CSV that ``score_trials.py`` can
aggregate (the SAME schema used for scoring real-arm trials by hand -> sim and real
numbers live in one place).

Unit bridge: this reuses ``GR00TRemotePolicy`` from lerobot_interface, the same
sim<->real mapping used to RECORD the training data, so eval and training agree by
construction (arm -100..100, gripper 0..100 <-> sim radians).

Success funnel per episode (block-only, unambiguous in sim):
  * approached : block was disturbed or lifted at all (arm reached and made contact)
  * grasped    : block lifted > GRASP_LIFT_M off the surface (a real pick)
  * placed     : block settled inside the box footprint (the task metric)
  * success    : grasped AND placed
Failure mode is auto-tagged from the first funnel stage that failed.

TWO-CONTAINER SETUP (server needs flash_attn -> real-robot image; env needs Isaac Lab
-> teleop-docker image; both run --network=host so the client reaches the server on
localhost):

  1) serve the checkpoint in the real-robot container:
       docker exec -d real-robot bash -lc \
         'cd /Isaac-GR00T && python gr00t/eval/run_gr00t_server.py \
            --model-path /workspace/ft_out/so101-box-merged-full/checkpoint-45000 \
            --embodiment-tag NEW_EMBODIMENT --port 5555'

  2) run this client in the Isaac (teleop-docker) container:
       /workspace/isaaclab/_isaac_sim/python.sh \
         source/sim_to_real_so101/scripts/eval_sim.py \
         --checkpoint-name checkpoint-45000 --episodes 20 --port 5555
"""
import argparse
import csv
import functools
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)

# --- CLI (must be parsed before AppLauncher launches the kit app) --------------
parser = argparse.ArgumentParser(description="Closed-loop sim eval of a GR00T policy.")
parser.add_argument("--checkpoint-name", type=str, default="unknown",
                    help="Label written to the CSV 'checkpoint' column (e.g. checkpoint-45000).")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--action-horizon", type=int, default=8,
                    help="Actions consumed from each policy chunk before re-querying.")
parser.add_argument("--max-control-steps", type=int, default=220,
                    help="Env steps per episode before giving up (30 Hz -> ~7 s).")
parser.add_argument("--warmup-steps", type=int, default=45,
                    help="Steps to drive the arm to the demo-start 'ready' pose "
                         "before running the policy (starts it in-distribution).")
parser.add_argument("--lang", type=str, default="Pick up the block and place in the box",
                    help="Task instruction; keep it matching the training task string.")
parser.add_argument("--csv", type=str, default=None,
                    help="Output CSV path (default: outputs/eval/sim_eval_<ckpt>_<ts>.csv).")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--no-dr", action="store_true",
                    help="Neutralize domain randomization (fixed lighting/colors/cameras, "
                         "block in the central zone) to match a fixed real rig.")
parser.add_argument("--block-jitter", type=float, default=0.02,
                    help="With --no-dr, +/- block XY jitter (m) around the central spawn.")
parser.add_argument("--arm-color", type=str, default="orange")
parser.add_argument("--block-color", type=str, default="red")
parser.add_argument("--debug", action="store_true",
                    help="Print per-episode arm motion + image/action diagnostics.")
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim GUI (non-headless).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
args.enable_cameras = True  # cameras must be on to feed the policy

app = AppLauncher(args).app

# --- Everything below runs after the kit app is live --------------------------
import numpy as np
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.table_env_cfg import BOX_POS
from sim_to_real_so101.utils.lerobot_interface import (
    LeRobotSO101Interface,
    GR00TRemotePolicy,
)

TASK = "Lerobot-So101-Teleop-Table-Task"

# Sim camera name -> model video key (model was trained on front/wrist; see the
# merged dataset's modality.json). The exterior D455 is the "front" exterior view;
# the gripper cam is the "wrist" view.
CAMERA_RENAME = {"external_D455": "front", "ego": "wrist"}

# Success geometry. Box center comes from the env config (stay in sync); the block
# is BLOCK_SIZE=(.022,.022,.05) tall. A block resting inside the box sits with its
# center a few cm above the mat and below the rim, and it must be settled (low vel).
BOX_XY = (BOX_POS[0], BOX_POS[1])
BOX_HALF = (0.06, 0.06)          # box inner half-footprint (m)
BOX_Z_FLOOR, BOX_Z_RIM = 0.03, 0.12
VEL_SETTLED = 0.05               # m/s: block has come to rest
GRASP_LIFT_M = 0.03              # min lift off the surface to count as a real pick
TOUCH_M = 0.01                   # block displaced/lifted this much => "approached"

# Demo-start "ready" pose (sim radians), = mean first-frame state across the recorded
# sim demos (gripper OPEN). The env's raw reset pose has the gripper CLOSED and a
# different arm configuration, which the policy never saw at t=0 -> it starts OOD and
# flails. Driving to this pose before the policy loop makes step 0 in-distribution.
START_POSE_RAD = [-0.4005, -0.5254, -0.0281, 1.4835, -1.3906, 1.6997]


def block_in_box(pos, lin_vel) -> bool:
    return (
        abs(pos[0] - BOX_XY[0]) < BOX_HALF[0]
        and abs(pos[1] - BOX_XY[1]) < BOX_HALF[1]
        and BOX_Z_FLOOR < pos[2] < BOX_Z_RIM
        and float(np.linalg.norm(lin_vel)) < VEL_SETTLED
    )


def main():
    # ----- Environment -----
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.seed = args.seed

    if args.no_dr:
        ev = env_cfg.events
        j = args.block_jitter

        def _try(msg, fn):
            try:
                fn(); print(f"[no-dr] {msg}")
            except Exception as e:
                print(f"[no-dr] SKIP {msg}: {e}")

        _try(f"block position -> central +/-{j}m, no yaw",
             lambda: ev.reset_block.params.__setitem__(
                 "pose_range", {"x": (-j, j), "y": (-j, j), "yaw": (0.0, 0.0)}))
        _try(f"block color -> {args.block_color}",
             lambda: ev.reset_block_color.params.__setitem__("color_names", [args.block_color]))
        _try(f"arm color -> {args.arm_color}",
             lambda: ev.reset_set_robot_visual_material.params.__setitem__(
                 "color_names", [args.arm_color]))
        _try("light exposure -> fixed 0",
             lambda: ev.reset_lightbox_light_exposure.params.__setitem__(
                 "exposure_range", (0.0, 0.0)))
        _try("mat rotation -> fixed",
             lambda: ev.reset_mat_rotation.params.__setitem__("yaw_range", (0.0, 0.0)))
        _try("ego cam fov -> fixed 13.5",
             lambda: ev.reset_camera_ego_fov.params.__setitem__(
                 "focal_length_range", (13.5, 13.5)))
        _try("external cam pose -> fixed",
             lambda: (ev.reset_camera_external_pose.params.__setitem__(
                          "pos_range", {"x": (0, 0), "y": (0, 0), "z": (0, 0)}),
                      ev.reset_camera_external_pose.params.__setitem__(
                          "rot_range", {"roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)})))

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device

    # Discover cameras exactly like the data-collection agent does.
    cameras = {}
    for obj in u.scene.keys():
        if obj.startswith("camera_"):
            ccfg = getattr(u.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": ccfg.height, "width": ccfg.width}
    rename_map = {k: CAMERA_RENAME[k] for k in cameras if k in CAMERA_RENAME}
    print(f"[INFO] cameras={list(cameras)} -> model keys {rename_map}")

    block = u.scene["block"]

    # ----- Policy (reuses the canonical sim<->real unit bridge; no hardware) -----
    iface = LeRobotSO101Interface(
        device=dev, port="null", id="eval", cameras=cameras,
        fps=30, kind="leader", rename_map=rename_map,
    )
    policy = GR00TRemotePolicy(
        robot_iface=iface, host=args.host, port=args.port,
        action_horizon=args.action_horizon, lang_instruction=args.lang,
    )
    try:
        policy.connect()
    except Exception as e:
        print(f"[FATAL] cannot reach GR00T server at {args.host}:{args.port} "
              f"-- start run_gr00t_server.py first. ({e})")
        env.close(); app.close(); raise SystemExit(1)

    # ----- CSV output -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or os.path.join(
        "outputs", "eval", f"sim_eval_{args.checkpoint_name}_{ts}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = ["checkpoint", "trial", "block_x", "block_y",
              "approached", "grasped", "placed", "success",
              "failure_mode", "maxlift", "notes"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    n_success = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        try:
            policy.reset()
        except Exception:
            pass
        # Let the scene settle after the DR reset so the block's start is stable.
        for _ in range(3):
            u.sim.step(render=False)
            u.scene.update(u.sim.get_physics_dt())

        # Drive the arm to the demo-start "ready" pose (gripper open) so the policy's
        # first observation is in-distribution. The ready pose is above/clear of the
        # block, so this does not disturb the block's randomized start.
        ready = torch.tensor(START_POSE_RAD, device=dev, dtype=torch.float32).unsqueeze(0)
        for _ in range(args.warmup_steps):
            obs, _, _, _, _ = env.step(ready)

        start = block.data.root_pos_w[0].detach().cpu().numpy()
        x0, y0, z0 = float(start[0]), float(start[1]), float(start[2])
        maxlift = 0.0

        q_start = obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy().copy()
        max_q_delta = 0.0
        dbg_printed = False

        for _ in range(args.max_control_steps):
            joint_pos = obs["policy"]["joint_pos_obs"][0]   # sim radians (6,)
            visual = obs["visual"]

            if args.debug and not dbg_printed:
                for cam in cameras:
                    im = visual[f"rgb_{cam}"][0].detach().cpu().numpy()
                    print(f"  [dbg] rgb_{cam}: shape={im.shape} dtype={im.dtype} "
                          f"min={float(im.min()):.3f} max={float(im.max()):.3f} "
                          f"mean={float(im.mean()):.3f}")

            sim_action = policy.get_action(joint_pos, visual)   # sim radians (6,)

            if args.debug and not dbg_printed:
                a = sim_action.detach().cpu().numpy()
                print(f"  [dbg] q_start(rad)={np.round(q_start,3)}")
                print(f"  [dbg] action(rad)={np.round(a,3)}  "
                      f"delta_from_q={np.round(a - q_start,3)}")
                dbg_printed = True

            obs, _, _, _, _ = env.step(sim_action.unsqueeze(0))
            q_now = obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy()
            max_q_delta = max(max_q_delta, float(np.abs(q_now - q_start).max()))
            maxlift = max(maxlift, float(block.data.root_pos_w[0, 2]) - z0)

        if args.debug:
            print(f"  [dbg] max arm joint travel this episode = {max_q_delta:.3f} rad "
                  f"({math.degrees(max_q_delta):.1f} deg)")

        final = block.data.root_pos_w[0].detach().cpu().numpy()
        fvel = block.data.root_lin_vel_w[0].detach().cpu().numpy()
        disp = math.hypot(float(final[0]) - x0, float(final[1]) - y0)

        placed = block_in_box(final, fvel)
        grasped = maxlift > GRASP_LIFT_M
        approached = disp > TOUCH_M or maxlift > TOUCH_M
        success = grasped and placed

        if success:
            fail = ""
        elif not approached:
            fail = "no_approach"
        elif not grasped:
            fail = "grasp_miss"
        else:
            fail = "place_miss"

        n_success += int(success)
        writer.writerow({
            "checkpoint": args.checkpoint_name, "trial": ep,
            "block_x": round(x0, 4), "block_y": round(y0, 4),
            "approached": int(approached), "grasped": int(grasped),
            "placed": int(placed), "success": int(success),
            "failure_mode": fail, "maxlift": round(maxlift, 4), "notes": "",
        })
        csv_file.flush()
        print(f"ep {ep:02d}: success={int(success)} "
              f"[appr={int(approached)} grasp={int(grasped)} place={int(placed)}] "
              f"maxlift={maxlift:+.3f} final=({final[0]:.3f},{final[1]:.3f},{final[2]:.3f}) "
              f"{fail}")

    csv_file.close()
    rate = n_success / max(1, args.episodes)
    print(f"SIM_EVAL_DONE checkpoint={args.checkpoint_name} "
          f"episodes={args.episodes} success={n_success} rate={rate:.1%}")
    print(f"[INFO] wrote {csv_path}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
