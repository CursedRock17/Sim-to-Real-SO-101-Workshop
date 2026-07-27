"""Find the shallowest downward tilt whose grasp cages+lifts the tall standing block
(hybrid: as horizontal as possible for the camera, tilted just enough not to topple).

For each elevation, build the tilted grasp with the FK-IK, a straight diagonal
approach (gripper descends in along -approach so it cages the block from above), then
close and lift the STANDING 5cm block. Report the min tilt that grips.
Run: isaaclab.sh -p fk_tilt_sweep.py
"""
import functools, json, os, math
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

S = os.environ.get("CALIB_DIR", "/root/smoke")
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.5, -0.5
OFFSET = 0.04                       # grasp center ~4cm back from frame (offset probe)
GRIP_ENV = np.array([0.218, 0.054, 0.05])   # recommended placement, block center
grip_base = fk.env_to_base(GRIP_ENV)
base_xy = fk.BASE_ORIGIN_ENV[:2]
outward = GRIP_ENV[:2] - base_xy; outward = outward / np.linalg.norm(outward)

# seed IK from a near-horizontal candidate, then homotopy through increasing tilt
cands = json.load(open(f"{S}/fk_candidates.json"))["candidates"]
seed = np.array(cands[0]["joints"])

ELEVS = [20, 35, 45, 55, 65, 75]
plans = {}
for e in ELEVS:
    er = math.radians(e)
    approach = np.array([math.cos(er) * outward[0], math.cos(er) * outward[1], -math.sin(er)])
    approach = approach / np.linalg.norm(approach)
    frame = grip_base + OFFSET * approach
    gq, ok = fk.ik(frame, approach, seed)
    if not ok:
        print(f"elev -{e}: IK fail"); continue
    seed = gq                                   # homotopy seed for next tilt
    # straight diagonal approach (position-straight, tool free while backing off)
    path, ok2 = fk.straight_approach(grip_base, approach, gq, back=0.07, n=18)
    # lift: raise grip point 8cm, same approach, seeded from grasp
    lq, okl = fk.ik(grip_base + np.array([0, 0, 0.08]) + OFFSET * approach, approach, gq)
    if ok2 and okl:
        plans[e] = (path, gq, lq)
    else:
        print(f"elev -{e}: approach ok={ok2} lift ok={okl}")
print(f"built plans for elevs: {sorted(plans)}")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]
jn = list(robot.data.joint_names)


def a6(arm5, jaw):
    v = torch.zeros((1, 6), device=dev)
    for k in range(5): v[0, k] = float(arm5[k])
    v[0, 5] = jaw
    return v


def snap(arm5, jaw):
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(arm5[k])
    q[0, jn.index("Jaw")] = jaw
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q); robot.write_data_to_sim()


def step(arm5, jaw, n):
    cmd = a6(arm5, jaw)
    for _ in range(n): env.step(cmd)


best = None
for e in sorted(plans):
    path, gq, lq = plans[e]
    block.write_root_pose_to_sim(torch.tensor([[GRIP_ENV[0], GRIP_ENV[1], 0.05, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
    snap(path[0], JAW_OPEN); step(path[0], JAW_OPEN, 6)
    block.write_root_pose_to_sim(torch.tensor([[GRIP_ENV[0], GRIP_ENV[1], 0.05, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
    step(path[0], JAW_OPEN, 4)
    z0 = float(block.data.root_pos_w[0, 2]); xy0 = np.array(block.data.root_pos_w[0, :2].tolist())
    for wp in path[1:]: step(wp, JAW_OPEN, 3)
    step(gq, JAW_CLOSE, 25)
    step(lq, JAW_CLOSE, 55)
    bp = block.data.root_pos_w[0].tolist()
    lift = bp[2] - z0; disp = float(np.linalg.norm(np.array(bp[:2]) - xy0))
    tag = "GRIP" if lift > 0.03 else ("topple" if disp > 0.03 else "miss")
    print(f"elev -{e:2d} deg: {tag:6s} lift={lift:+.3f} disp={disp:.3f}")
    if lift > 0.03 and best is None:
        best = e                      # shallowest tilt that grips

print(f"SHALLOWEST_GRIP elev=-{best} deg" if best is not None else "NO_GRIP at any tilt")
print("TILT_DONE")
env.close(); app.close()
