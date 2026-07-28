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
pa.add_argument("--episodes", type=int, default=5, help="fixed episode count (ignored if --target > 0)")
pa.add_argument("--target", type=int, default=0, help="collect until this many SUCCESSFUL episodes")
pa.add_argument("--max_attempts", type=int, default=500, help="cap on attempts when using --target")
pa.add_argument("--repo_id", type=str, default="CursedRock17/so101_block_grab_planned")
pa.add_argument("--out", type=str, default=None)
pa.add_argument("--save_success_only", action="store_true")
pa.add_argument("--fixed_block", type=str, default=None, help="x,y to place the block each episode")
pa.add_argument("--video", type=str, default=None, help="mp4 path; saves the first successful episode")
pa.add_argument("--seed", type=int, default=0)
pa.add_argument("--num_envs", type=int, default=1,
                help=">1 runs parallel envs (TiledCamera renders them in one pass) for a big "
                     "throughput win; sim runs in parallel, dataset writing stays sequential.")
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

# ---- strict "placed in box" metric (mirrors mdp.terms.vial_placed_on_rack) -------
# The box is a STATIC axis-aligned AssetBaseCfg at env (0.10, 0.20, 0.091), so we can
# check in world/env coords directly (no per-frame box-pose query needed). A block
# counts as placed only if its centre is inside the box footprint AND below the rim
# (actually dropped IN, not perched on top/beside) AND settled (near-zero velocity,
# so a block mid-bounce or rolling over the rim does not count).
# NOTE: extents are for the current CardboardBox.usd; verify if the box changes.
BOX_HALF = (0.06, 0.06)              # interior half-extents (x, y), env
BOX_FLOOR, BOX_RIM = 0.03, 0.12      # settled block-centre z range inside the box
VEL_SETTLED = 0.05                   # m/s: below this the block is at rest


def block_in_box(pos, lin_vel):
    """pos=(x,y,z) env, lin_vel=(3,) world. True iff the block is settled inside the box."""
    return (abs(pos[0] - BOX_ENV[0]) < BOX_HALF[0]
            and abs(pos[1] - BOX_ENV[1]) < BOX_HALF[1]
            and BOX_FLOOR < pos[2] < BOX_RIM
            and float(np.linalg.norm(lin_vel)) < VEL_SETTLED)

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
env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=a.num_envs))
obs, _ = env.reset()
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]; jn = list(robot.data.joint_names)


def cam(o, k, i=0):
    img = o["visual"][k][i].detach().cpu().numpy()
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
rng = np.random.default_rng(a.seed)
fixed = tuple(float(v) for v in a.fixed_block.split(",")) if a.fixed_block else None
save_success_only = a.save_success_only or a.target > 0
block_blue = u.scene["block_blue"] if "block_blue" in u.scene.keys() else None
AWAY = [0.45, 0.45, 0.05]                              # off-mat parking spot for the distractor
video_saved = False


def collect_parallel():
    """Vectorized collection over a.num_envs parallel envs. The sim/render (the
    throughput bottleneck) runs N envs at once via TiledCamera; dataset writing stays
    sequential (one episode at a time). Each batch = one full reset -> N fresh
    randomized episodes; per-env trajectories are padded to a common length and
    stepped in lockstep, frames buffered per env, then successful episodes saved.
    UNTESTED (needs GPU) -- smoke-test with `--num_envs 2 --target 2` once free.
    """
    N = a.num_envs
    save_only = a.save_success_only or a.target > 0
    succ = 0; attempts = 0
    print(f"parallel collection: num_envs={N}, target={a.target}")
    while succ < a.target and attempts < a.max_attempts:
        obs, _ = env.reset()
        attempts += N
        # block-amount DR: park the blue distractor off-mat in ~40% of envs
        if block_blue is not None:
            park = rng.random(N) < 0.4
            pose = torch.cat([block_blue.data.root_pos_w.clone(),
                              block_blue.data.root_quat_w.clone()], dim=1)     # (N,7)
            away = torch.tensor(AWAY, device=dev, dtype=pose.dtype)
            for i in range(N):
                if park[i]: pose[i, :3] = away
            block_blue.write_root_pose_to_sim(pose)
            block_blue.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev))
        for _ in range(3): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
        bpos = block.data.root_pos_w.detach().cpu().numpy()                   # (N,3)
        arm_idx = [jn.index(n) for n in ISAAC_ARM]
        home = robot.data.joint_pos[:, arm_idx].detach().cpu().numpy()        # (N,5)
        trajs = [plan(float(bpos[i, 0]), float(bpos[i, 1]), home[i]) for i in range(N)]
        trajs = [t[0] if t is not None else None for t in trajs]
        planned = [i for i in range(N) if trajs[i] is not None]
        if not planned:
            print(f"batch: all {N} plans failed"); continue
        L = max(len(trajs[i]) for i in planned)
        # snap each planned env's arm to its first frame (others stay at reset pose)
        q0 = robot.data.joint_pos.clone()
        for i in planned:
            for k, n in enumerate(ISAAC_ARM): q0[i, jn.index(n)] = float(trajs[i][0][0][k])
            q0[i, jn.index("Jaw")] = trajs[i][0][1]
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        for _ in range(2): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
        z0 = block.data.root_pos_w[:, 2].clone()                              # (N,)
        maxlift = torch.zeros(N, device=dev)
        bufs = {i: [] for i in planned}
        act = torch.zeros((N, 6), device=dev)
        for t in range(L):
            for i in range(N):
                tr = trajs[i]
                q5, jaw = (home[i], JAW_OPEN) if tr is None else (tr[t] if t < len(tr) else tr[-1])
                for k in range(5): act[i, k] = float(q5[k])
                act[i, 5] = jaw
            for _ in range(2): obs, _, _, _, _ = env.step(act)
            maxlift = torch.maximum(maxlift, block.data.root_pos_w[:, 2] - z0)
            state = obs["policy"]["joint_pos_obs"].detach().cpu().numpy()     # (N,6)
            for i in planned:
                if t < len(trajs[i]):                                        # record real frames only
                    bufs[i].append((act[i].detach().cpu().numpy().copy(), state[i].copy(),
                                    cam(obs, "rgb_external_D455", i), cam(obs, "rgb_ego", i)))
        bf = block.data.root_pos_w.detach().cpu().numpy()
        bvel = block.data.root_lin_vel_w.detach().cpu().numpy()
        for i in planned:
            ok = block_in_box(bf[i].tolist(), bvel[i].tolist()) and float(maxlift[i]) > 0.03
            if ok or not save_only:
                for (a6, st, top, wrist) in bufs[i]:
                    ds.add_frame({"action": rad_to_norm(a6[:5], a6[5]),
                                  "observation.state": rad_to_norm(st[:5], st[5]),
                                  "observation.images.top": top,
                                  "observation.images.wrist.top": wrist,
                                  "task": "pick the block and place it in the box"})
                ds.save_episode(); succ += int(ok)
        print(f"batch done: attempts~{attempts} planned={len(planned)}/{N} succ={succ}/{a.target}")
    ds.finalize()
    print(f"PIPELINE_DONE(parallel) attempts~{attempts} success={succ}")


if a.num_envs > 1:
    collect_parallel()
    env.close(); app.close()
    import sys; sys.exit(0)

succ = 0; ep = -1
while (succ < a.target) if a.target > 0 else (ep + 1 < a.episodes):
    ep += 1
    if a.target > 0 and ep >= a.max_attempts:
        print(f"REACHED max_attempts={a.max_attempts} with {succ} successes"); break
    obs, _ = env.reset()
    # block-amount DR: ~40% of episodes keep only the red block (park the blue one)
    one_block = block_blue is not None and rng.random() < 0.4
    if block_blue is not None:
        if one_block:
            block_blue.write_root_pose_to_sim(torch.tensor([[*AWAY, 1.,0.,0.,0.]], device=dev))
            block_blue.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
    if fixed is not None:                              # optional: force a favorable block pose
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
    vid_frames = []; aborted = False
    LIFT_CHECK = 78                              # by here the grasp+lift is done
    for fi, (q5, jaw) in enumerate(fr):
        act = torch.zeros((1, 6), device=dev)
        for k in range(5): act[0, k] = float(q5[k])
        act[0, 5] = jaw
        for _ in range(2): obs, _, _, _, _ = env.step(act)
        maxlift = max(maxlift, float(block.data.root_pos_w[0, 2]) - z0)
        if fi >= LIFT_CHECK and maxlift < 0.02:   # grasp failed -> abort before transport
            aborted = True; break
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
    bvel = block.data.root_lin_vel_w[0].tolist()
    success = (not aborted) and block_in_box(bf, bvel) and maxlift > 0.03
    print(f"attempt {ep} (succ {succ}/{a.target if a.target else a.episodes}): "
          f"block=({bx:.3f},{by:.3f}) one_block={one_block} maxlift={maxlift:+.3f} "
          f"aborted={aborted} final=({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f}) SUCCESS={success}")
    if a.video and success and not video_saved:
        imageio.mimwrite(a.video, vid_frames, fps=FPS, quality=8)
        video_saved = True
        print(f"VIDEO_SAVED {a.video} ({len(vid_frames)} frames)")
    if success or (not save_success_only and not aborted):
        ds.save_episode(); succ += int(success)
    else:
        ds.clear_episode_buffer()

print(f"PIPELINE_DONE attempts={ep + 1} success={succ}")
ds.finalize()          # CRITICAL: writes the parquet footers so the dataset is valid
print("FINALIZED")
env.close(); app.close()
