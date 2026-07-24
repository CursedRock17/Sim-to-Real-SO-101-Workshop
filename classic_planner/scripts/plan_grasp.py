"""Plan a top-down grasp trajectory for the SO-101 to a block position.

Strategy for the 5-DOF arm: use IK to find a joint configuration for a top-down
(vertical) grasp -- the one orientation family this arm can fully achieve -- then
let MoveIt/OMPL plan a collision-free JOINT trajectory to it (robust; avoids
fragile pose-goal planning on a rank-deficient 6-DOF Jacobian).

Run:
    source-ros
    colcon build --packages-select so101_moveit_config && source install/setup.bash
    python3 classic_planner/scripts/plan_grasp.py
"""
import math
import os
import numpy as np

from geometry_msgs.msg import Pose
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder

TIP = "gripper_frame_link"
GROUP = "arm"


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
    """Try top-down orientations (approach = world -Z) with a yaw sweep; return
    (RobotState, joints, yaw, flip_axis) for the first IK solution, else None."""
    for flip_axis in ([1, 0, 0], [0, 1, 0]):          # two ways to point tool down
        base_down = q_axis_angle(flip_axis, math.pi)
        for yaw in np.linspace(-math.pi, math.pi, 9):  # jaw alignment sweep
            quat = q_mul(q_axis_angle([0, 0, 1], yaw), base_down)
            rs = RobotState(model)
            rs.set_to_default_values(GROUP, "home")
            rs.update()
            ok = rs.set_from_ik(GROUP, make_pose(xyz, quat), TIP, 0.1)
            if ok:
                rs.update()
                joints = rs.get_joint_group_positions(GROUP)
                return rs, joints, yaw, flip_axis
    return None


def plan_to_state(arm, model, goal_state, from_state_name="rest"):
    start = RobotState(model)
    start.set_to_default_values(GROUP, from_state_name)
    start.update()
    arm.set_start_state(robot_state=start)
    arm.set_goal_state(robot_state=goal_state)
    return arm.plan()


def main():
    cfg = build_config()
    moveit = MoveItPy(node_name="moveit_py", config_dict=cfg.to_dict())
    model = moveit.get_robot_model()
    arm = moveit.get_planning_component(GROUP)

    # Candidate block positions in base_link frame (m). Scan to learn the
    # reachable top-down workspace, then plan to the first that solves.
    candidates = [
        (0.20, 0.00, 0.04), (0.17, 0.00, 0.04), (0.15, 0.00, 0.03),
        (0.15, 0.10, 0.04), (0.15, -0.10, 0.04),
        (0.10, 0.15, 0.04), (0.10, -0.15, 0.04),
        (0.00, 0.20, 0.04), (0.00, -0.20, 0.04),
    ]

    STANDOFF = 0.05  # pre-grasp height above the block (m)

    print("=== IK reachability scan (top-down grasp + pre-grasp standoff) ===")
    chosen = None
    for xyz in candidates:
        grasp = find_grasp_ik(model, xyz)
        pre = find_grasp_ik(model, (xyz[0], xyz[1], xyz[2] + STANDOFF))
        tag = "REACHABLE" if (grasp and pre) else ("grasp-only" if grasp else "unreachable")
        extra = f" joints={[round(j,3) for j in grasp[1]]}" if grasp else ""
        print(f"{tag} {xyz}{extra}")
        if grasp and pre and chosen is None:
            chosen = (xyz, grasp, pre)

    if chosen is None:
        print("GRASP_FAILED: no candidate with both grasp and pre-grasp reachable")
        os._exit(2)

    xyz, grasp, pre = chosen
    grasp_state, joints = grasp[0], grasp[1]
    pre_state = pre[0]
    print(f"\n=== Planning grasp at {xyz} (standoff {STANDOFF} m) ===")

    plan_pre = plan_to_state(arm, model, pre_state, "rest")
    if not (plan_pre and plan_pre.trajectory is not None):
        print("GRASP_FAILED: could not plan rest -> pre-grasp")
        os._exit(4)
    n1 = len(plan_pre.trajectory.get_robot_trajectory_msg().joint_trajectory.points)
    print(f"STAGE reach->pre-grasp OK waypoints={n1}")

    # Descend: plan pre-grasp -> grasp (start from pre_state joint config).
    start = RobotState(model)
    start.set_joint_group_positions(GROUP, pre[1])
    start.update()
    arm.set_start_state(robot_state=start)
    arm.set_goal_state(robot_state=grasp_state)
    plan_desc = arm.plan()
    if not (plan_desc and plan_desc.trajectory is not None):
        print("GRASP_FAILED: could not plan pre-grasp -> grasp")
        os._exit(5)
    n2 = len(plan_desc.trajectory.get_robot_trajectory_msg().joint_trajectory.points)
    print(f"STAGE pre-grasp->grasp OK waypoints={n2}")

    print(f"\nGRASP_PLAN_OK target={xyz} grasp_joints={[round(j,3) for j in joints]}")
    print("GRASP_PASSED")
    os._exit(0)


if __name__ == "__main__":
    main()
