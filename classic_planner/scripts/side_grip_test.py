"""Test horizontal side-grasp candidates in Isaac: teleport to the grasp pose over
a standing block, close, raise to the lift config, and report which grip."""
import functools, json
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

S = "/root/smoke"
data = json.load(open(f"{S}/side_candidates.json"))
cands = data["candidates"]; ge = data["grip_env"]
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.5, -0.5
print(f"testing {len(cands)} candidates")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]
jn = list(robot.data.joint_names)


def cmd(arm5, jaw):
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(arm5[k])
    q[0, jn.index("Jaw")] = jaw
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q); robot.write_data_to_sim()
    a6 = torch.zeros((1, 6), device=dev)
    for k in range(5): a6[0, k] = float(arm5[k])
    a6[0, 5] = jaw
    return a6


best = None
for idx, c in enumerate(cands[:30]):
    block.write_root_pose_to_sim(torch.tensor([[ge[0], ge[1], 0.06, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
    a6 = cmd(c["joints"], JAW_OPEN)
    for _ in range(12): env.step(a6)
    z0 = float(block.data.root_pos_w[0, 2])
    a6 = cmd(c["joints"], JAW_CLOSE)
    for _ in range(25): env.step(a6)
    # raise to lift config (targets, physics holds the grip)
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(c["lift_joints"][k])
    q[0, jn.index("Jaw")] = JAW_CLOSE
    robot.set_joint_position_target(q)
    a6 = torch.zeros((1,6), device=dev)
    for k in range(5): a6[0,k] = float(c["lift_joints"][k])
    a6[0,5] = JAW_CLOSE
    for _ in range(60): env.step(a6)
    lift = float(block.data.root_pos_w[0, 2]) - z0
    if lift > 0.03:
        print(f"cand {idx}: GRIP lift={lift:+.3f} approach={[round(x,2) for x in c['approach']]}")
        if best is None or lift > best[1]:
            best = (idx, lift)

if best:
    print(f"BEST cand {best[0]} lift={best[1]:.3f}")
else:
    print("NO_GRIP among candidates")
print("SIDE_TEST_DONE")
env.close(); app.close()
