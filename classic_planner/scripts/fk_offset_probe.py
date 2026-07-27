"""Isolate grasp geometry from the approach: for one FK grasp config, sweep where
the block sits along the tool axis and do a STATIC close (snap arm around the
pre-placed block, no approach), to find the offset that actually cages+lifts it.
If some offset grips, the horizontal grasp geometry works and only the approach
(toppling the tall block) needs solving. Run: isaaclab.sh -p fk_offset_probe.py
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

S = os.environ.get("CALIB_DIR", "/root/smoke")
cands = json.load(open(f"{S}/fk_candidates.json"))["candidates"]
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.5, -0.5
IDX = int(os.environ.get("CAND_IDX", "0"))
c = cands[IDX]
grasp_q = np.array(c["joints"])
fkpos, fkR = fk.fk(grasp_q); approach = fkR[:, 2]
frame_env = fk.base_to_env(fkpos)
print(f"cand {IDX}: frame_env={np.round(frame_env,3)} approach={np.round(approach,3)}")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]
jn = list(robot.data.joint_names)


def a6(jaw):
    v = torch.zeros((1, 6), device=dev)
    for k in range(5): v[0, k] = float(grasp_q[k])
    v[0, 5] = jaw
    return v


def snap(jaw):
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(grasp_q[k])
    q[0, jn.index("Jaw")] = jaw
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q); robot.write_data_to_sim()


for d in [0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
    bc = frame_env - d * approach          # block center along the tool axis
    for standing in (True, False):
        # standing: upright 5cm block; flat: laid along approach (short, stable)
        if standing:
            quat = [1., 0., 0., 0.]; z = 0.05
        else:
            quat = [0.7071, 0., 0.7071, 0.]; z = 0.036  # tipped onto its side
        snap(JAW_OPEN)
        block.write_root_pose_to_sim(torch.tensor([[bc[0], bc[1], z, *quat]], device=dev))
        block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))
        for _ in range(10): env.step(a6(JAW_OPEN))
        z0 = float(block.data.root_pos_w[0, 2]); xy0 = np.array(block.data.root_pos_w[0, :2].tolist())
        for _ in range(25): env.step(a6(JAW_CLOSE))
        # lift by pitching the whole arm up a touch via a raised config from candidate
        for _ in range(40): env.step(torch.tensor([[*c["lift_joints"], JAW_CLOSE]], device=dev, dtype=torch.float32))
        bp = block.data.root_pos_w[0].tolist()
        lift = bp[2] - z0; disp = float(np.linalg.norm(np.array(bp[:2]) - xy0))
        tag = "GRIP" if lift > 0.03 else ("push" if disp > 0.02 else "open")
        print(f"  d={d:.2f} {'stand' if standing else 'flat '}: {tag:4s} lift={lift:+.3f} disp={disp:.3f}")

print("PROBE_DONE")
env.close(); app.close()
