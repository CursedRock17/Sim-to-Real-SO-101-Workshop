"""Shared SO-101 MoveIt planning helpers: config, top-down grasp IK, frame transforms.

Frame note: the Isaac scene expresses object poses in the ENV frame, while MoveIt
plans in the arm's base_link frame. The arm root sits in the ENV frame at
ARM_BASE_POS with a Z rotation of ARM_BASE_YAW (from assets/so101.py). env<->base
below assumes the URDF base_link axes align with the USD Robot prim axes -- this
alignment must be CALIBRATED against Isaac (drive a known joint config in sim,
compare the gripper world pose to URDF FK) before trusting the transform.
"""
import math
import numpy as np

from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder

TIP = "gripper_frame_link"
ARM = "arm"
GRIPPER = "gripper"

# Arm root in the Isaac ENV frame (assets/so101.py: pos=(-0.05,0,0), yaw=90deg).
ARM_BASE_POS = np.array([-0.05, 0.0, 0.0])
ARM_BASE_YAW = math.radians(90.0)


def build_config():
    return (
        MoveItConfigsBuilder("so101", package_name="so101_moveit_config")
        .robot_description(file_path="config/so101.urdf")
        .robot_description_semantic(file_path="config/so101.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(file_path="config/moveit_cpp.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )


# ---- frame transforms (ENV <-> base_link), planar Z rotation ----------------
def _rot_z(yaw, p):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = p
    return np.array([c * x - s * y, s * x + c * y, z])


def env_to_base(p_env):
    return _rot_z(-ARM_BASE_YAW, np.asarray(p_env, float) - ARM_BASE_POS)


def base_to_env(p_base):
    return _rot_z(ARM_BASE_YAW, np.asarray(p_base, float)) + ARM_BASE_POS


# ---- quaternion helpers -----------------------------------------------------
def q_axis_angle(axis, angle):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    s = math.sin(angle / 2.0)
    return np.array([a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2.0)])


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def make_pose(xyz, quat):
    p = Pose()
    p.position.x, p.position.y, p.position.z = [float(v) for v in xyz]
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = [float(v) for v in quat]
    return p


def find_grasp_ik(model, xyz):
    """Top-down grasp IK (approach = world -Z) with a yaw sweep for jaw alignment.
    Returns (RobotState, joints) for the first solution, else None."""
    for flip_axis in ([1, 0, 0], [0, 1, 0]):
        base_down = q_axis_angle(flip_axis, math.pi)
        for yaw in np.linspace(-math.pi, math.pi, 9):
            quat = q_mul(q_axis_angle([0, 0, 1], yaw), base_down)
            rs = RobotState(model)
            rs.set_to_default_values(ARM, "home")
            rs.update()
            if rs.set_from_ik(ARM, make_pose(xyz, quat), TIP, 0.1):
                rs.update()
                return rs, list(rs.get_joint_group_positions(ARM))
    return None


def waypoints(plan_result):
    return len(plan_result.trajectory.get_robot_trajectory_msg().joint_trajectory.points)
