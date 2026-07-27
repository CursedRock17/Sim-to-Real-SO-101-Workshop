"""Precisely measure the grasp-centre residual at the actual block zone: plan a
top-down grasp to the block-spawn centre, snap there, and sweep the block on a fine
grid to find where it cages. Reports the residual to fold into GRASP_CENTER_TOOL so
the grasp centres correctly across the spawn zone. Run: isaaclab.sh -p fk_center_probe.py
"""
import functools, json, os
print = functools.partial(print, flush=True)
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app
import torch, numpy as np
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa
import so101_fk as fk

ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.7, -0.5
DOWN = np.array([0.0, 0.0, -1.0])
SEED = np.array([0.114, 0.304, 0.017, 1.329, 0.521])
BLOCK_C = (0.16, 0.06, 0.05)                      # block spawn-zone centre
grasp_q, ok = fk.grasp_ik(fk.env_to_base(BLOCK_C), DOWN, SEED)
pred = fk.base_to_env(fk.grasp_center(grasp_q))    # where we THINK the fingers are
R = fk.fk(grasp_q)[1]
liftq, _ = fk.ik_pos(fk.env_to_base((BLOCK_C[0], BLOCK_C[1], 0.14)), grasp_q)
print(f"grasp_ik ok={ok} predicted centre_env={np.round(pred,3)}")

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
    for _ in range(n): env.step(a6(q5, jaw))


results = {}
for dx in np.arange(-0.02, 0.021, 0.005):
    for dy in np.arange(-0.02, 0.021, 0.005):
        bx, by = pred[0] + dx, pred[1] + dy
        snap(grasp_q, JAW_OPEN)
        block.write_root_pose_to_sim(torch.tensor([[bx, by, 0.05, 1.,0.,0.,0.]], device=dev))
        block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
        step(grasp_q, JAW_OPEN, 10)
        z0 = float(block.data.root_pos_w[0, 2])
        step(grasp_q, JAW_CLOSE, 22)
        step(liftq, JAW_CLOSE, 45)
        lift = float(block.data.root_pos_w[0, 2]) - z0
        if lift > 0.03:
            results[(round(dx, 3), round(dy, 3))] = round(lift, 3)

if results:
    ks = np.array(list(results.keys()))
    cdx, cdy = ks[:, 0].mean(), ks[:, 1].mean()      # centroid of the gripping region
    print(f"gripping cells: {len(results)}  region centroid residual dx={cdx:+.4f} dy={cdy:+.4f}")
    resid_world = np.array([cdx, cdy, 0.0])
    corrected = fk.GRASP_CENTER_TOOL + R.T @ resid_world
    print(f"CORRECTED_GRASP_CENTER_TOOL {np.round(corrected,4).tolist()}")
else:
    print("NO_GRIP on grid")
print("CENTER_DONE")
env.close(); app.close()
