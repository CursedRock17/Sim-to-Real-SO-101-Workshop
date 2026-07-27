"""Replay a real teleop episode (decoded to Isaac radians) in our table env to
confirm the decoding + that the working grasp is top-down. Places the block at the
episode's grasp point, drives the arm through the recorded actions, tracks the block.
Run: isaaclab.sh -p replay_teleop.py
"""
import functools, json, os, math
print = functools.partial(print, flush=True)
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app
import torch, numpy as np, imageio.v2 as imageio
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa

S = os.environ.get("CALIB_DIR", "/root/smoke")
ep = json.load(open(f"{S}/ep0.json"))["ep0"]
MINS = np.array([-110, -100, -100, -95, -160.0]); MAXS = np.array([110, 100, 90, 95, 160.0])
GMIN, GMAX = -10.0, 100.0


def decode(raw):
    n = (np.asarray(raw[:5], float) + 100.0) / 200.0
    arm = (MINS + n * (MAXS - MINS)) * math.pi / 180.0
    g = (GMIN + (raw[5] / 100.0) * (GMAX - GMIN)) * math.pi / 180.0
    return np.concatenate([arm, [g]])


traj = np.array([decode(r) for r in ep])   # (N,6) Isaac radians
GRIP_ENV = (0.214, -0.046)                  # episode grasp point (from analyze_teleop)
print(f"replay {len(traj)} frames; block at {GRIP_ENV}")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset(); u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]; jn = list(robot.data.joint_names)
ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# start arm at frame 0, block standing at the grasp point
q0 = torch.zeros((1, robot.num_joints), device=dev)
for k, n in enumerate(ORDER): q0[0, jn.index(n)] = float(traj[0, k])
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
block.write_root_pose_to_sim(torch.tensor([[GRIP_ENV[0], GRIP_ENV[1], 0.05, 1.,0.,0.,0.]], device=dev))
block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
for _ in range(3): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())

z0 = float(block.data.root_pos_w[0, 2]); xy0 = np.array(block.data.root_pos_w[0, :2].tolist())
maxlift = 0.0
for i in range(len(traj)):
    act = torch.tensor(traj[i:i+1], device=dev, dtype=torch.float32)
    for _ in range(2): obs, _, _, _, _ = env.step(act)
    bz = float(block.data.root_pos_w[0, 2]); maxlift = max(maxlift, bz - z0)
    if i in (60, 90, 118, 150, 190, len(traj)-1):
        img = obs["visual"]["rgb_external_D455"][0].detach().cpu().numpy()
        img = (img*255).clip(0,255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0,255).astype(np.uint8)
        imageio.imwrite(f"{S}/replay_{i}.png", img[..., :3])
        bp = block.data.root_pos_w[0].tolist()
        print(f"  f={i:3d} block=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}) lift={bp[2]-z0:+.3f}")

bp = block.data.root_pos_w[0].tolist()
disp = float(np.linalg.norm(np.array(bp[:2]) - xy0))
print(f"REPLAY maxlift={maxlift:+.3f} final=({bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}) xy_disp={disp:.3f}")
print(f"RESULT {'PICKED_AND_MOVED' if maxlift>0.03 and disp>0.05 else ('LIFTED' if maxlift>0.03 else 'NO_LIFT')}")
print("REPLAY_DONE")
env.close(); app.close()
