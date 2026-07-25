"""MoveIt side of the calibration: URDF forward kinematics for the same joint
configs Isaac recorded (in base_link frame)."""
import functools, json, os
print = functools.partial(print, flush=True)
import numpy as np
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder

# Isaac ARM joint name -> URDF joint name, and the URDF arm-group order.
MAP = {"Rotation": "shoulder_pan", "Pitch": "shoulder_lift", "Elbow": "elbow_flex",
       "Wrist_Pitch": "wrist_flex", "Wrist_Roll": "wrist_roll"}
URDF_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

cfg = (MoveItConfigsBuilder("so101", package_name="so101_moveit_config")
       .robot_description(file_path="config/so101.urdf")
       .robot_description_semantic(file_path="config/so101.srdf")
       .robot_description_kinematics(file_path="config/kinematics.yaml")
       .joint_limits(file_path="config/joint_limits.yaml")
       .trajectory_execution(file_path="config/moveit_controllers.yaml")
       .moveit_cpp(file_path="config/moveit_cpp.yaml")
       .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
       .to_moveit_configs())
moveit = MoveItPy(node_name="moveit_py", config_dict=cfg.to_dict())
model = moveit.get_robot_model()

isaac = json.load(open("/root/smoke/isaac_fk.json"))

def link_pose(rs, link):
    T = np.array(rs.get_global_link_transform(link))  # 4x4 in base_link frame
    return T[:3, 3].tolist(), T[:3, :3].tolist()

out = []
for rec in isaac["records"]:
    q = rec["q"]
    vals = [q[k] for k in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]]
    rs = RobotState(model)
    rs.set_joint_group_positions("arm", vals)
    rs.update()
    gl_pos, gl_rot = link_pose(rs, "gripper_link")
    gf_pos, gf_rot = link_pose(rs, "gripper_frame_link")
    bl_pos, _ = link_pose(rs, "base_link")
    out.append({"q": q, "gripper_link_pos": gl_pos, "gripper_link_rot": gl_rot,
                "gripper_frame_pos": gf_pos, "gripper_frame_rot": gf_rot, "base_link_pos": bl_pos})
    print("q", {k: round(v, 2) for k, v in q.items()}, "gripper_link", [round(x, 4) for x in gl_pos])

json.dump(out, open("/root/smoke/urdf_fk.json", "w"), indent=2)
print("URDF_FK_DONE", len(out))
os._exit(0)
