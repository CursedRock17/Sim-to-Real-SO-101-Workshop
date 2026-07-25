"""Shared SO-101 MoveIt planning helpers: config, top-down grasp IK, frame transforms.

Frame transform (CALIBRATED against Isaac): objects are expressed in the ENV
frame; MoveIt plans in the URDF base_link frame. An 8-config forward-kinematics
calibration (URDF FK vs Isaac gripper world pose, rigid-fit residual 0.00 mm)
showed the two frames are AXIS-ALIGNED -- the rotation is identity, NOT the 90 deg
yaw one might infer from assets/so101.py. Only a translation separates them:
base_link's origin sits at BASE_ORIGIN_ENV in the ENV frame. The same calibration
confirmed the USD and upstream-URDF joint conventions match exactly.
"""
import math
import numpy as np

from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder

TIP = "gripper_frame_link"
ARM = "arm"
GRIPPER = "gripper"

# base_link origin expressed in the Isaac ENV frame (from FK calibration).
BASE_ORIGIN_ENV = np.array([-0.0658, 0.0208, 0.0325])


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


# ---- frame transforms (ENV <-> base_link): calibrated as translation-only -----
def env_to_base(p_env):
    return np.asarray(p_env, float) - BASE_ORIGIN_ENV


def base_to_env(p_base):
    return np.asarray(p_base, float) + BASE_ORIGIN_ENV


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


def ik_pose(model, xyz, quat):
    """IK for a specific pose; returns (RobotState, joints) or None."""
    rs = RobotState(model)
    rs.set_to_default_values(ARM, "home")
    rs.update()
    if rs.set_from_ik(ARM, make_pose(xyz, quat), TIP, 0.1):
        rs.update()
        return rs, list(rs.get_joint_group_positions(ARM))
    return None


def find_grasp_ik(model, xyz):
    """Top-down grasp IK (approach = world -Z) with a yaw sweep for jaw alignment.
    Returns (RobotState, joints, quat) for the first solution, else None."""
    for flip_axis in ([1, 0, 0], [0, 1, 0]):
        base_down = q_axis_angle(flip_axis, math.pi)
        for yaw in np.linspace(-math.pi, math.pi, 9):
            quat = q_mul(q_axis_angle([0, 0, 1], yaw), base_down)
            r = ik_pose(model, xyz, quat)
            if r is not None:
                return r[0], r[1], quat
    return None


def waypoints(plan_result):
    return len(plan_result.trajectory.get_robot_trajectory_msg().joint_trajectory.points)
