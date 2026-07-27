"""Visualize the hybrid grasp: horizontal pre-grasp (camera-friendly) -> tilt to a
-TILT deg grasp -> straight descent onto the block -> close -> lift. Renders each
stage and reports the grip. Run: isaaclab.sh -p fk_hybrid_demo.py
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
import so101_fk as fk

S = os.environ.get("CALIB_DIR", "/root/smoke")
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.5, -0.5
OFFSET = 0.04
TILT = float(os.environ.get("TILT", "60"))
GRIP = np.array([0.218, 0.054, 0.05]); gb = fk.env_to_base(GRIP)
base_xy = fk.BASE_ORIGIN_ENV[:2]; out = GRIP[:2] - base_xy; out = out / np.linalg.norm(out)
seed = np.array(json.load(open(f"{S}/fk_candidates.json"))["candidates"][0]["joints"])

# horizontal establishing pose: a known reachable horizontal config backed well off
# the block (camera sees the gripper horizontal; it does not touch the block here)
hpre = np.array(json.load(open(f"{S}/fk_candidates.json"))["candidates"][0]["pre_joints"]); okh = True
# tilted grasp + diagonal straight descent + lift
te = math.radians(TILT); tapp = np.array([math.cos(te)*out[0], math.cos(te)*out[1], -math.sin(te)]); tapp/=np.linalg.norm(tapp)
gq, okg = fk.ik(gb + OFFSET*tapp, tapp, seed)
path, okp = fk.straight_approach(gb, tapp, gq, back=0.07, n=18)
lq, okl = fk.ik(gb + np.array([0,0,0.09]) + OFFSET*tapp, tapp, gq)
print(f"tilt={TILT} hpre={okh} grasp={okg} approach={okp} lift={okl}")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset(); u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]; jn = list(robot.data.joint_names)


def a6(q5, jaw):
    v = torch.zeros((1, 6), device=dev)
    for k in range(5): v[0, k] = float(q5[k])
    v[0, 5] = jaw; return v


def snap(q5, jaw):
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(q5[k])
    q[0, jn.index("Jaw")] = jaw
    robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q); robot.write_data_to_sim()


def step(q5, jaw, n):
    obs = None
    for _ in range(n): obs, _, _, _, _ = env.step(a6(q5, jaw))
    return obs


def shot(obs, name):
    img = obs["visual"]["rgb_external_D455"][0].detach().cpu().numpy()
    img = (img*255).clip(0,255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0,255).astype(np.uint8)
    imageio.imwrite(f"{S}/hyb_{name}.png", img[..., :3])


def place():
    block.write_root_pose_to_sim(torch.tensor([[GRIP[0], GRIP[1], 0.05, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))


place(); snap(hpre, JAW_OPEN); obs = step(hpre, JAW_OPEN, 8)
shot(obs, "1horizontal")                                        # camera-friendly horizontal pose
snap(path[0], JAW_OPEN); place(); obs = step(path[0], JAW_OPEN, 8)  # go to tilted pre-grasp, reset block
z0 = float(block.data.root_pos_w[0, 2])
shot(obs, "2tilted")
for wp in path[1:]: obs = step(wp, JAW_OPEN, 3)
shot(obs, "3atblock")
obs = step(gq, JAW_CLOSE, 25); shot(obs, "4closed")
obs = step(lq, JAW_CLOSE, 60); shot(obs, "5lifted")
lift = float(block.data.root_pos_w[0, 2]) - z0
print(f"HYBRID lift={lift:+.3f} -> {'GRIP' if lift>0.03 else 'FAIL'}")
print("HYBRID_DONE")
env.close(); app.close()
