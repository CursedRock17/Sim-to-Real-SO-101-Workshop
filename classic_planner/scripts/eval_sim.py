"""Closed-loop SIM evaluation of a fine-tuned GR00T policy in the SO-101 table env.

Runs the policy in Lerobot-So101-Teleop-Table-Task and scores task success with the
same block_in_box() metric the data generator used -> a direct "does the policy
actually do the task" number, complementary to open-loop action MSE (eval_finetune.sh).

TWO-CONTAINER SETUP (both need the GPU, so run AFTER the finetune finishes):
  1) serve the checkpoint in the gr00t-ft container (has flash_attn/torchcodec):
       docker exec -d gr00t-ft bash -lc 'cd /Isaac-GR00T && \
         python gr00t/eval/run_gr00t_server.py \
           --model-path /workspace/ft_out/so101_block_pickplace/checkpoint-30000 \
           --embodiment-tag NEW_EMBODIMENT --port 5555'
  2) run this client in the Isaac container (has Isaac Lab + the table task):
       docker exec teleop-moveit-isaac bash -lc 'pip install -q pyzmq msgpack; \
         export CALIB_DIR=/workspace/Sim-to-Real-SO-101-Workshop/classic_planner/scripts; \
         cd $CALIB_DIR && /workspace/isaaclab/isaaclab.sh -p eval_sim.py --episodes 20'
  Both containers use --network=host, so the client reaches the server on localhost.

UNTESTED (needs the GPU free) -- smoke-test with --episodes 2 once the finetune ends.
"""
import functools, os, argparse, math
print = functools.partial(print, flush=True)
from isaaclab.app import AppLauncher
pa = argparse.ArgumentParser()
pa.add_argument("--episodes", type=int, default=20)
pa.add_argument("--host", type=str, default="localhost")
pa.add_argument("--port", type=int, default=5555)
pa.add_argument("--action_horizon", type=int, default=8, help="actions executed per policy query")
pa.add_argument("--max_control_steps", type=int, default=220, help="policy steps per episode cap")
pa.add_argument("--lang", type=str, default="pick the block and place it in the box")
AppLauncher.add_app_launcher_args(pa)
a = pa.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np, torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa
import io, zmq, msgpack   # minimal GR00T policy-server client -- importing the gr00t
                          # package would pull in Gr00tPolicy -> flash_attn (absent in
                          # the Isaac python), so we speak the zmq/msgpack wire protocol
                          # directly (matches gr00t/policy/server_client.py).


def _enc(obj):
    if isinstance(obj, np.ndarray):
        b = io.BytesIO(); np.save(b, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    return obj


def _dec(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class PolicyClient:
    """Drop-in for gr00t's PolicyClient over the same REQ/msgpack protocol."""

    def __init__(self, host="localhost", port=5555, timeout_ms=30000):
        self.ctx = zmq.Context(); self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.socket.send(msgpack.packb(req, default=_enc, use_bin_type=True))
        resp = msgpack.unpackb(self.socket.recv(), object_hook=_dec, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def get_action(self, obs):
        return tuple(self._call("get_action", {"observation": obs, "options": None}))

    def reset(self):
        return self._call("reset", {"options": None})

    def ping(self):
        return self._call("ping", requires_input=False)

TASK = "Lerobot-So101-Teleop-Table-Task"
ISAAC_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
STATE_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]  # policy order == Isaac order
BOX_ENV = (0.10, 0.20); BOX_HALF = (0.06, 0.06); BOX_FLOOR, BOX_RIM = 0.03, 0.12; VEL_SETTLED = 0.05
# LeRobot normalized <-> radians (same mapping used to record the dataset)
DEG_MIN = np.array([-110, -100, -100, -95, -160.0]); DEG_MAX = np.array([110, 100, 90, 95, 160.0])
GDEG_MIN, GDEG_MAX = -10.0, 100.0


def rad_to_norm(state6):
    arm, jaw = np.asarray(state6[:5], float), float(state6[5])
    arm_raw = ((arm * 180 / math.pi - DEG_MIN) / (DEG_MAX - DEG_MIN)) * 200 - 100
    grip_raw = ((jaw * 180 / math.pi - GDEG_MIN) / (GDEG_MAX - GDEG_MIN)) * 100
    return np.concatenate([arm_raw, [grip_raw]]).astype(np.float32)


def norm_to_rad(raw6):
    arm_raw, grip_raw = np.asarray(raw6[:5], float), float(raw6[5])
    arm = (DEG_MIN + (arm_raw + 100) / 200 * (DEG_MAX - DEG_MIN)) * math.pi / 180
    jaw = (GDEG_MIN + (grip_raw / 100) * (GDEG_MAX - GDEG_MIN)) * math.pi / 180
    return np.concatenate([arm, [jaw]]).astype(np.float32)


def block_in_box(pos, lin_vel):
    return (abs(pos[0] - BOX_ENV[0]) < BOX_HALF[0] and abs(pos[1] - BOX_ENV[1]) < BOX_HALF[1]
            and BOX_FLOOR < pos[2] < BOX_RIM and float(np.linalg.norm(lin_vel)) < VEL_SETTLED)


def _add_dim(o):                                     # mirror recursive_add_extra_dim
    out = {}
    for k, v in o.items():
        out[k] = _add_dim(v) if isinstance(v, dict) else (v[np.newaxis, ...] if isinstance(v, np.ndarray) else [v])
    return out


def build_input(front, wrist, state_norm, lang):
    obs = {"video": {"front": front, "wrist": wrist},
           "state": {"single_arm": state_norm[:5], "gripper": state_norm[5:6]},
           "language": {"annotation.human.task_description": lang}}
    return _add_dim(_add_dim(obs))                   # -> (B=1, T=1, ...)


env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=1))
u = env.unwrapped; dev = u.device
robot = u.scene["robot"]; block = u.scene["block_red"]
policy = PolicyClient(host=a.host, port=a.port)
try:
    policy.ping(); print(f"connected to policy server {a.host}:{a.port}")
except Exception as e:
    print(f"CANNOT REACH policy server at {a.host}:{a.port} -- start run_gr00t_server first. ({e})")
    env.close(); app.close(); raise SystemExit(1)


def cam(o, k):
    img = o["visual"][k][0].detach().cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0, 255).astype(np.uint8)


succ = 0
for ep in range(a.episodes):
    obs, _ = env.reset()
    try: policy.reset()                              # clear the policy's action-chunk state
    except Exception: pass
    for _ in range(3): u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    z0 = float(block.data.root_pos_w[0, 2]); maxlift = 0.0
    buf = []
    for step in range(a.max_control_steps):
        if not buf:
            state_rad = obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy()
            mi = build_input(cam(obs, "rgb_external_D455"), cam(obs, "rgb_ego"),
                             rad_to_norm(state_rad), a.lang)
            res = policy.get_action(mi)
            chunk = res[0] if isinstance(res, tuple) else res
            T = chunk["single_arm"].shape[1]
            for t in range(min(a.action_horizon, T)):
                raw6 = np.concatenate([chunk["single_arm"][0][t], chunk["gripper"][0][t]])
                buf.append(norm_to_rad(raw6))
        rad6 = buf.pop(0)
        act = torch.tensor(rad6[None, :], device=dev, dtype=torch.float32)
        for _ in range(2): obs, _, _, _, _ = env.step(act)   # 2 substeps = one 30 Hz control step
        maxlift = max(maxlift, float(block.data.root_pos_w[0, 2]) - z0)
    bf = block.data.root_pos_w[0].tolist(); bvel = block.data.root_lin_vel_w[0].tolist()
    ok = block_in_box(bf, bvel) and maxlift > 0.03
    succ += int(ok)
    print(f"ep {ep}: SUCCESS={ok} maxlift={maxlift:+.3f} final=({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f})")

print(f"SIM_EVAL_DONE episodes={a.episodes} success={succ} rate={succ/max(1,a.episodes):.2%}")
env.close(); app.close()
