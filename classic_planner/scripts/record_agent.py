"""Execute a MoveIt-planned 30 Hz trajectory in Isaac and record it.

Reads trajectory.json, drives the SO-101 through the block->box pick-place via the
env's JointPositionAction, records (action, observation.state, images) frames in
the LeRobot schema, and applies the vials-style success metric (block in box).
Run in teleop-moveit (Isaac python): isaaclab.sh -p record_agent.py
"""
import functools, json, os
print = functools.partial(print, flush=True)
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import torch, numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa
import omni.usd
from pxr import Usd, UsdGeom

S = "/root/smoke"
plan = json.load(open(f"{S}/trajectory.json"))
traj = np.array(plan["trajectory"], dtype=np.float32)   # (N,6) Isaac order
block_env = plan["block_env"]
box_env = plan["box_env"]
print(f"loaded trajectory: {traj.shape} @ {plan['fps']} fps")

TASK = "Lerobot-So101-Teleop-Table-Task"
env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped
dev = u.device
block = u.scene["block_red"]

# Fix the block at the planned pose (test run: no randomization).
bpose = torch.tensor([[block_env[0], block_env[1], 0.055, 1., 0., 0., 0.]], device=dev)
block.write_root_pose_to_sim(bpose)
block.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))

# Start the arm at the trajectory's first config so execution begins smoothly.
robot = u.scene["robot"]
q0 = torch.zeros((1, robot.num_joints), device=dev)
jn = list(robot.data.joint_names)
for k, name in enumerate(plan["joint_order"]):
    q0[0, jn.index(name)] = float(traj[0, k])
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
for _ in range(3):
    u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())

stage = omni.usd.get_context().get_stage()
cam = lambda o, k: o["visual"][k][0].detach().cpu().numpy()
actions = torch.zeros((1, 6), device=dev)
rec_action, rec_state = [], []
block_z = []
frames_png = {}

N = traj.shape[0]
for i in range(N):
    actions[0] = torch.from_numpy(traj[i]).to(dev)
    for _ in range(2):                       # 2x 1/60s env steps = 1/30s
        obs, _, _, _, _ = env.step(actions)
    rec_action.append(traj[i].copy())
    rec_state.append(obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy().copy())
    bp_now = block.data.root_pos_w[0].tolist()
    block_z.append(float(bp_now[2]))
    if i == 0:
        _b0 = np.array(bp_now[:2]); _knock = None
    elif _knock is None and np.linalg.norm(np.array(bp_now[:2]) - _b0) > 0.015:
        _knock = i
        print(f"KNOCK at frame {i}/{N} (t={i/plan['fps']:.2f}s): block moved to ({bp_now[0]:.3f},{bp_now[1]:.3f})")
    if i in (60, 68, 76, 84, 92, N - 1):
        img = cam(obs, "rgb_external_D455")
        img = (img*255).clip(0,255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0,255).astype(np.uint8)
        imageio.imwrite(f"{S}/ep_{i}.png", img[..., :3]); frames_png[i] = f"ep_{i}.png"

# ---- metrics (vials-style: object ended in the target) ----
bf = block.data.root_pos_w[0].tolist()
box = box_env
in_xy = abs(bf[0]-box[0]) < 0.06 and abs(bf[1]-box[1]) < 0.06
lifted = max(block_z) - block_z[0]
success = in_xy and bf[2] > 0.045
print(f"REC frames={len(rec_action)} action_dim={len(rec_action[0])} state_dim={len(rec_state[0])} imgs={list(frames_png.values())}")
print(f"BLOCK start_z={block_z[0]:.3f} max_lift={lifted:.3f} final=({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f})")
print(f"METRIC in_box_xy={in_xy} lifted={lifted:.3f} SUCCESS={success}")
np.savez(f"{S}/episode.npz", action=np.array(rec_action), state=np.array(rec_state))
print("RECORD_DONE")
env.close(); app.close()
