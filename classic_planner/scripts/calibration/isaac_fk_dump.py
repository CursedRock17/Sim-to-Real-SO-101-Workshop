"""Isaac side of the frame calibration: drive the SO-101 to known joint configs
and record the ACTUAL joint values + gripper/base body world poses (ENV frame)."""
import functools, json
print = functools.partial(print, flush=True)
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa

TASK = "Lerobot-So101-Teleop-Table-Task"
env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=1))
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
jn = list(robot.data.joint_names)
bn = list(robot.data.body_names)
print("joint_names:", jn)
print("body_names:", bn)

ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
gidx = bn.index("gripper")
bidx = bn.index("base")

# Configs specified by ARM joint name (jaw stays 0). Varied to span the workspace.
configs = [
    {}, {"Rotation": 0.6}, {"Pitch": -0.5}, {"Elbow": 0.6}, {"Wrist_Pitch": 0.5},
    {"Wrist_Roll": 0.8},
    {"Rotation": 0.3, "Pitch": -0.4, "Elbow": 0.6, "Wrist_Pitch": 0.3, "Wrist_Roll": -0.5},
    {"Rotation": -0.5, "Pitch": -0.6, "Elbow": 0.8, "Wrist_Pitch": 0.6, "Wrist_Roll": 0.4},
]

dev = u.device
records = []
for cfg in configs:
    q = robot.data.default_joint_pos.clone()
    for name, val in cfg.items():
        q[0, jn.index(name)] = val
    for name in jn:  # zero everything not specified (incl. Jaw), keep specified
        if name not in cfg and name in ARM:
            q[0, jn.index(name)] = 0.0
        if name == "Jaw":
            q[0, jn.index(name)] = 0.0
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    for _ in range(15):
        u.sim.step(render=False)
        u.scene.update(u.sim.get_physics_dt())
    actual = {name: float(robot.data.joint_pos[0, jn.index(name)]) for name in ARM}
    gp = robot.data.body_pos_w[0, gidx].tolist(); gq = robot.data.body_quat_w[0, gidx].tolist()
    bp = robot.data.body_pos_w[0, bidx].tolist(); bq = robot.data.body_quat_w[0, bidx].tolist()
    records.append({"q": actual, "gripper_pos": gp, "gripper_quat_wxyz": gq,
                    "base_pos": bp, "base_quat_wxyz": bq})
    print("cfg", {k: round(v, 2) for k, v in actual.items()},
          "-> gripper", [round(x, 4) for x in gp])

out = {"joint_names": jn, "body_names": bn, "arm_joints": ARM, "records": records}
with open("/root/smoke/isaac_fk.json", "w") as f:
    json.dump(out, f, indent=2)
print("ISAAC_FK_DONE", len(records), "configs")
env.close(); app.close()
