"""Verify PLANNED top-down grasps (fingers placed on the block via grasp_ik) at
several demo-zone block positions: descend, close, lift. Confirms the pipeline can
plan a working grasp to an arbitrary block position. Run: isaaclab.sh -p fk_topdown_test.py
"""
import functools, json, os
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
SEED = np.array(json.load(open(f"{S}/grasp_zone.json"))["median_grasp_q"])
APPROACH = np.array([0.0, 0.0, -1.0])   # straight down: reachable across the whole zone
GRASP_Z, PRE_Z = 0.05, 0.09           # finger-centre heights (ENV): block centre, hover
POSITIONS = [(0.175, -0.005), (0.16, 0.06), (0.20, 0.02), (0.15, -0.03)]


def plan(bx, by):
    grasp_q, ok = fk.grasp_ik(fk.env_to_base((bx, by, GRASP_Z)), APPROACH, SEED)
    if not ok:
        return None
    desc, q = [], grasp_q
    for z in np.linspace(PRE_Z, GRASP_Z, 10):
        q, o = fk.grasp_ik(fk.env_to_base((bx, by, z)), APPROACH, q)
        if not o:
            return None
        desc.append(q)
    liftq, ol = fk.grasp_ik(fk.env_to_base((bx, by, GRASP_Z + 0.06)), APPROACH, grasp_q)
    if not ol:
        liftq = desc[0]                    # fall back to the hover as the lift
    return desc, grasp_q, liftq


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


ok_count = 0
for bx, by in POSITIONS:
    pl = plan(bx, by)
    if pl is None:
        print(f"pos ({bx:.2f},{by:.2f}): PLAN_FAIL"); continue
    desc, grasp_q, liftq = pl
    block.write_root_pose_to_sim(torch.tensor([[bx, by, 0.05, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
    snap(desc[0], JAW_OPEN); step(desc[0], JAW_OPEN, 6)
    block.write_root_pose_to_sim(torch.tensor([[bx, by, 0.05, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
    step(desc[0], JAW_OPEN, 4)
    z0 = float(block.data.root_pos_w[0, 2])
    for q in desc[1:]: step(q, JAW_OPEN, 3)
    step(grasp_q, JAW_CLOSE, 25)
    obs = step(liftq, JAW_CLOSE, 55)
    lift = float(block.data.root_pos_w[0, 2]) - z0
    tag = "GRIP" if lift > 0.03 else "FAIL"
    if tag == "GRIP": ok_count += 1
    print(f"pos ({bx:.2f},{by:.2f}): {tag} lift={lift:+.3f}")

print(f"TOPDOWN_PLANNED {ok_count}/{len(POSITIONS)} gripped")
print("TOPDOWN_DONE")
env.close(); app.close()
