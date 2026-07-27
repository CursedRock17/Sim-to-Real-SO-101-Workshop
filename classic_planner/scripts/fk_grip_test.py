"""Physics-verify the FK horizontal grasps in Isaac with a STRAIGHT-LINE approach.

For each candidate: stand the block at the grip point, snap to the pre-grasp (jaws
open, ~6 cm back along -approach), then step through the so101_fk straight-line
approach configs (position-straight, tool tilting into horizontal), close the jaws,
and lift. A candidate that raises the block is a real, IK-free horizontal grasp.
Diagnostic snapshots are saved for one candidate. Run: isaaclab.sh -p fk_grip_test.py
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
data = json.load(open(f"{S}/fk_candidates.json"))
cands = data["candidates"]
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.5, -0.5
BLOCK_STAND_Z = 0.05
N_TEST = min(15, len(cands))
DIAG = int(os.environ.get("DIAG_IDX", "0"))
print(f"testing {N_TEST} FK candidates (straight-line approach), diag idx={DIAG}")

env = gym.make("Lerobot-So101-Teleop-Table-Task",
               cfg=parse_env_cfg("Lerobot-So101-Teleop-Table-Task", device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]
jn = list(robot.data.joint_names)
bn = list(robot.data.body_names)
print("body_names:", bn)
GFRAME = bn.index("gripper_frame_link") if "gripper_frame_link" in bn else None
DIAG_ONLY = os.environ.get("DIAG_ONLY") == "1"


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
    cmd = a6(arm5, jaw); obs = None
    for _ in range(n): obs, _, _, _, _ = env.step(cmd)
    return obs


def shot(obs, name):
    img = obs["visual"]["rgb_external_D455"][0].detach().cpu().numpy()
    img = (img*255).clip(0,255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0,255).astype(np.uint8)
    imageio.imwrite(f"{S}/fk_{name}.png", img[..., :3])


def place(g):
    block.write_root_pose_to_sim(torch.tensor([[g[0], g[1], BLOCK_STAND_Z, 1.,0.,0.,0.]], device=dev))
    block.write_root_velocity_to_sim(torch.zeros((1,6), device=dev))


best = None
for idx in range(N_TEST):
    c = cands[idx]; g = c["grip_env"]
    grip_base = fk.env_to_base((g[0], g[1], g[2]))
    approach = np.array(c["approach"]); grasp_q = np.array(c["joints"])
    path, ok = fk.straight_approach(grip_base, approach, grasp_q, back=0.06, n=16)
    if not ok:
        print(f"cand {idx:2d}: no-approach"); continue
    place(g)
    snap(path[0], JAW_OPEN)                    # pre-grasp, clear of block
    obs = step(path[0], JAW_OPEN, 6)
    place(g); obs = step(path[0], JAW_OPEN, 4)  # re-settle block
    z0 = float(block.data.root_pos_w[0, 2]); xy0 = np.array(block.data.root_pos_w[0, :2].tolist())
    if idx == DIAG: shot(obs, "1pre")
    for wp in path[1:]:                         # straight-line approach
        obs = step(wp, JAW_OPEN, 3)
    if idx == DIAG: shot(obs, "2grasp")
    if idx == DIAG:
        blk = block.data.root_pos_w[0].tolist()
        fkpos, fkR = fk.fk(grasp_q); pred_frame = fk.base_to_env(fkpos)
        pred_grip = fk.base_to_env(fkpos - fk.GRIP_OFFSET * fkR[:, 2])
        print(f"  DIAG block_now={np.round(blk,3)} (placed {np.round(g,3)})")
        print(f"  DIAG FK pred grip={np.round(pred_grip,3)} frame={np.round(pred_frame,3)} approach={np.round(fkR[:,2],3)}")
        if GFRAME is not None:
            act = robot.data.body_pos_w[0, GFRAME].tolist()
            print(f"  DIAG isaac gripper_frame_link world={np.round(act,3)}  (FK frame - isaac = {np.round(np.array(pred_frame)-np.array(act),3)})")
    obs = step(grasp_q, JAW_CLOSE, 25)          # close
    if idx == DIAG: shot(obs, "3closed")
    obs = step(c["lift_joints"], JAW_CLOSE, 60)  # lift
    if idx == DIAG: shot(obs, "4lifted")
    bp = block.data.root_pos_w[0].tolist()
    lift = bp[2] - z0; disp = float(np.linalg.norm(np.array(bp[:2]) - xy0))
    tag = "GRIP" if lift > 0.03 else ("knock" if disp > 0.03 else "miss")
    print(f"cand {idx:2d}: {tag:5s} lift={lift:+.3f} xy_disp={disp:.3f} elev={c['elev_deg']:+5.1f} "
          f"grip={[round(x,2) for x in g]}")
    if lift > 0.03 and (best is None or lift > best[1]):
        best = (idx, lift)
    if DIAG_ONLY:
        break

print(f"BEST cand {best[0]} lift={best[1]:.3f}" if best else "NO_GRIP among FK candidates")
print("FK_GRIP_DONE")
env.close(); app.close()
