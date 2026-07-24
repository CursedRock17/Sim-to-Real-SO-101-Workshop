"""Pipeline smoke test: plan the SO-101 arm between two named joint states.

Proves the MoveIt config loads (URDF/SRDF/kinematics/OMPL) and OMPL produces a
trajectory -- no IK yet. Run under system Python 3.12 with ROS sourced:
    source-ros
    colcon build --packages-select so101_moveit_config && source install/setup.bash
    python3 classic_planner/scripts/plan_smoke.py
"""
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder


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


def main():
    cfg = build_config()
    moveit = MoveItPy(node_name="moveit_py", config_dict=cfg.to_dict())
    model = moveit.get_robot_model()
    print("GROUPS:", model.joint_model_group_names)

    arm = moveit.get_planning_component("arm")

    start = RobotState(model)
    start.set_to_default_values("arm", "rest")
    start.update()
    arm.set_start_state(robot_state=start)

    arm.set_goal_state(configuration_name="home")

    result = arm.plan()
    if result and result.trajectory is not None:
        msg = result.trajectory.get_robot_trajectory_msg()
        pts = msg.joint_trajectory.points
        dur = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
        print(f"PLAN_OK waypoints={len(pts)} duration_s={dur:.3f}")
        print("SMOKE_PASSED")
    else:
        ec = getattr(result, "error_code", None)
        print("PLAN_FAILED error_code=", ec)
        print("SMOKE_FAILED")


if __name__ == "__main__":
    main()
