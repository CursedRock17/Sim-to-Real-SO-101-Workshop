"""Full hybrid-integration loop: plan a top-down pick-and-place with so101_fk (no
MoveIt), execute it in Isaac, and record LeRobot episodes -- the planned replacement
for human teleop.

Per episode: reset (randomizes the block), read the block pose, plan
home->hover->descend->close->lift->transport->release->retreat entirely with the
FK-IK, execute at 30 Hz recording action+state (in LeRobot normalized units) and the
top + wrist cameras, score success (block in box), and save the episode.

Run in the teleop image (Isaac python):
  isaaclab.sh -p topdown_pipeline.py --episodes 5 --repo_id CursedRock17/so101_block_grab_planned
"""
import functools, os, argparse, math
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
pa = argparse.ArgumentParser()
pa.add_argument("--episodes", type=int, default=5)
pa.add_argument("--repo_id", type=str, default="CursedRock17/so101_block_grab_planned")
pa.add_argument("--out", type=str, default=None)
pa.add_argument("--save_success_only", action="store_true")
pa.add_argument("--fixed_block", type=str, default=None, help="x,y to place the block each episode")
pa.add_argument("--video", type=str, default=None, help="mp4 path; saves the first successful episode")
AppLauncher.add_app_launcher_args(pa)
a = pa.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import torch, numpy as np
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa
import so101_fk as fk
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 30
TASK = "Lerobot-So101-Teleop-Table-Task"
ISAAC_ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_OPEN, JAW_CLOSE = 1.7, -0.5   # open wide for capture range on descent
DOWN = np.array([0.0, 0.0, -1.0])
GRASP_Z, HOVER_Z = 0.05, 0.095       # finger-centre heights for the top-down grasp (ENV)
HIGH_Z, DROP_Z = 0.18, 0.11          # frame heights for the high carry / drop into box (ENV)
BOX_ENV = (0.10, 0.20)
SEED = np.array([0.114, 0.304, 0.017, 1.329, 0.521])   # median demo grasp config

# ---- LeRobot normalized-unit encoding (inverse of lerobot_interface mapping) ----
DEG_MIN = np.array([-110, -100, -100, -95, -160.0]); DEG_MAX = np.array([110, 100, 90, 95, 160.0])
GDEG_MIN, GDEG_MAX = -10.0, 100.0


def rad_to_norm(arm5, jaw_rad):
    deg = np.asarray(arm5, float) * 180 / math.pi
    n = (deg - DEG_MIN) / (DEG_MAX - DEG_MIN)
    arm_raw = n * 200 - 100
    gdeg = jaw_rad * 180 / math.pi
    grip_raw = ((gdeg - GDEG_MIN) / (GDEG_MAX - GDEG_MIN)) * 100
    return np.concatenate([arm_raw, [grip_raw]]).astype(np.float32)


# ---- planning (all FK-IK) ---------------------------------------------------
def lerp(qa, qb, n):
    return [qa + (qb - qa) * (i + 1) / n for i in range(n)]


def cart(bx, by, z_from, z_to, seed, n):
    out, q = [], seed
    for z in np.linspace(z_from, z_to, n):
        q, ok = fk.grasp_ik(fk.env_to_base((bx, by, z)), DOWN, q)
        if not ok:
            return None
        out.append(q)
    return out


def cart_xy(p_from, p_to, z, seed, n):
    out, q = [], seed
    for t in np.linspace(0, 1, n):
        x = p_from[0] + (p_to[0] - p_from[0]) * t; y = p_from[1] + (p_to[1] - p_from[1]) * t
        q, ok = fk.grasp_ik(fk.env_to_base((x, y, z)), DOWN, q)
        if not ok:
            return None
        out.append(q)
    return out


def pos_path(pts, seed, n_each):
    """ik_pos through a list of ENV frame targets, seeded continuously (smooth)."""
    out, q = [], seed
    prev = pts[0]
    for nxt in pts[1:]:
        for t in np.linspace(0, 1, n_each)[1:]:
            tgt = prev + (np.array(nxt) - prev) * t
            q, ok = fk.ik_pos(fk.env_to_base(tgt), q)
            if not ok:
                return None
            out.append(q)
        prev = np.array(nxt)
    return out


def plan(bx, by, home_q):
    # precise straight-down grasp
    grasp_q, ok = fk.grasp_ik(fk.env_to_base((bx, by, GRASP_Z)), DOWN, SEED)
    if not ok:
        return None
    hover_q, ok = fk.grasp_ik(fk.env_to_base((bx, by, HOVER_Z)), DOWN, grasp_q)
    if not ok:
        return None
    descend = cart(bx, by, HOVER_Z, GRASP_Z, hover_q, 12)
    if descend is None:
        return None
    gfz = float(fk.base_to_env(fk.fk(grasp_q)[0])[2])   # grasp frame height
    # high carry arc (position-only, orientation free so the arm tilts and reaches high):
    #   up over the block -> across to over the box -> down into the box
    carry = pos_path([(bx, by, gfz), (bx, by, HIGH_Z),
                      (BOX_ENV[0], BOX_ENV[1], HIGH_Z), (BOX_ENV[0], BOX_ENV[1], DROP_Z)],
                     grasp_q, n_each=18)
    if carry is None:
        return None
    lift_part = carry[:17]                       # roughly the vertical lift portion
    drop_q = carry[-1]
    frames = []                                   # (arm5, jaw)
    for q in lerp(home_q, hover_q, 22): frames.append((q, JAW_OPEN))
    for q in descend: frames.append((q, JAW_OPEN))
    for i in range(8): frames.append((grasp_q, JAW_OPEN + (JAW_CLOSE - JAW_OPEN) * (i + 1) / 8))
    for _ in range(14): frames.append((grasp_q, JAW_CLOSE))          # settle the grip
    for q in carry: frames.append((q, JAW_CLOSE))                    # lift high + carry + lower in
    for _ in range(6): frames.append((drop_q, JAW_CLOSE))            # settle in box
    for i in range(8): frames.append((drop_q, JAW_CLOSE + (JAW_OPEN - JAW_CLOSE) * (i + 1) / 8))
    for q in lerp(drop_q, home_q, 22): frames.append((q, JAW_OPEN))
    return frames, len(carry)


# ---- env + dataset ----------------------------------------------------------
env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=1))
obs, _ = env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]; jn = list(robot.data.joint_names)


def cam(o, k):
    img = o["visual"][k][0].detach().cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0, 255).astype(np.uint8)


h, w = cam(obs, "rgb_external_D455").shape[:2]
root = a.out or f"/root/datasets/{a.repo_id.split('/')[-1]}"
feats = {
    "action": {"dtype": "float32", "shape": (6,), "names": ISAAC_ARM + ["gripper"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ISAAC_ARM + ["gripper"]},
    "observation.images.top": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist.top": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
}
ds = LeRobotDataset.create(a.repo_id, fps=FPS, features=feats, root=root, robot_type="so101_follower")
print(f"dataset at {root}  img={h}x{w}")


def set_arm(q5, jaw, teleport=False):
    q = torch.zeros((1, robot.num_joints), device=dev)
    for k, n in enumerate(ISAAC_ARM): q[0, jn.index(n)] = float(q5[k])
    q[0, jn.index("Jaw")] = jaw
    if teleport:
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q); robot.write_data_to_sim()


import imageio.v2 as imageio
fixed = tuple(float(v) for v in a.fixed_block.split(",")) if a.fixed_block else None
video_saved = False
succ = 0
for ep in range(a.episodes):
    obs, _ = env.reset()
    if fixed is not None:                              # force a favorable block pose
        block.write_root_pose_to_sim(torch.tensor([[fixed[0], fixed[1], 0.05, 1.,0.,0.,0.]], device=dev))
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        for _ in range(3): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    bp = block.data.root_pos_w[0].tolist(); bx, by = bp[0], bp[1]
    home_q = np.array([float(robot.data.joint_pos[0, jn.index(n)]) for n in ISAAC_ARM])
    pl = plan(bx, by, home_q)
    if pl is None:
        print(f"ep {ep}: PLAN_FAIL block=({bx:.3f},{by:.3f})"); continue
    fr, jump = pl
    set_arm(fr[0][0], fr[0][1], teleport=True)
    for _ in range(2): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    z0 = float(block.data.root_pos_w[0, 2]); maxlift = 0.0
    vid_frames = []
    for fi, (q5, jaw) in enumerate(fr):
        act = torch.zeros((1, 6), device=dev)
        for k in range(5): act[0, k] = float(q5[k])
        act[0, 5] = jaw
        for _ in range(2): obs, _, _, _, _ = env.step(act)
        maxlift = max(maxlift, float(block.data.root_pos_w[0, 2]) - z0)
        top = cam(obs, "rgb_external_D455")
        if a.video and not video_saved:
            vid_frames.append(top.copy())
        state = obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy()
        ds.add_frame({
            "action": rad_to_norm(q5, jaw),
            "observation.state": rad_to_norm(state[:5], state[5]),
            "observation.images.top": top,
            "observation.images.wrist.top": cam(obs, "rgb_ego"),
            "task": "pick the block and place it in the box",
        })
    bf = block.data.root_pos_w[0].tolist()
    in_box = abs(bf[0] - BOX_ENV[0]) < 0.06 and abs(bf[1] - BOX_ENV[1]) < 0.06
    success = in_box and maxlift > 0.03
    print(f"ep {ep}: block=({bx:.3f},{by:.3f}) maxlift={maxlift:+.3f} "
          f"final=({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f}) in_box={in_box} SUCCESS={success}")
    if a.video and success and not video_saved:
        imageio.mimwrite(a.video, vid_frames, fps=FPS, quality=8)
        video_saved = True
        print(f"VIDEO_SAVED {a.video} ({len(vid_frames)} frames)")
    if success or not a.save_success_only:
        ds.save_episode(); succ += int(success)
    else:
        ds.clear_episode_buffer()

print(f"PIPELINE_DONE episodes={a.episodes} success={succ}")
env.close(); app.close()
