"""Plan a full SO-101 pick-and-place sequence (block -> box), top-down grasps.

Stages: rest -> pre-grasp -> grasp -> [close] -> lift -> over-box -> place ->
[open] -> retreat. Arm stages are collision-free OMPL joint plans; gripper stages
plan the 1-DOF gripper group. Also transforms the actual table_env_cfg scene
positions ENV->base and reports their reachability.

Run:
    source-ros
    colcon build --packages-select so101_moveit_config && source install/setup.bash
    python3 classic_planner/scripts/plan_pick_place.py
"""
import os
import numpy as np

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState

import so101_planning as sp

# Actual scene object positions in the ENV frame (mirrors table_env_cfg.py; in the
# record loop these will come live from Isaac per-episode). ENV->base uses the
# CALIBRATED transform in so101_planning (translation-only).
SCENE_BLOCK_ENV = (0.16, 0.06, 0.05)   # block_red init
SCENE_BOX_ENV = (0.10, 0.20, 0.06)     # cardboard box (place near the box floor)

# Known-reachable base-frame fallbacks (from the plan_grasp reachability scan).
BLOCK_FALLBACK = (0.15, -0.05, 0.03)
PLACE_FALLBACK = (0.13, 0.10, 0.03)


def reachable_above(model, xyz, heights=(0.06, 0.05, 0.04, 0.03, 0.02)):
    """First IK-reachable top-down waypoint above xyz (arm has a low vertical
    ceiling for top-down poses, so descend the offsets until one solves)."""
    for h in heights:
        r = sp.find_grasp_ik(model, (xyz[0], xyz[1], xyz[2] + h))
        if r:
            return r, xyz[2] + h
    return None, None


def choose(model, primary_xyz, fallback_xyz, label):
    res = sp.find_grasp_ik(model, primary_xyz)
    if res:
        print(f"{label}: using scene position {tuple(round(v,3) for v in primary_xyz)} (reachable)")
        return primary_xyz, res
    res = sp.find_grasp_ik(model, fallback_xyz)
    print(f"{label}: scene position {tuple(round(v,3) for v in primary_xyz)} NOT top-down reachable "
          f"-> using fallback {fallback_xyz}")
    return fallback_xyz, res


def plan_arm(arm, model, start_joints, goal_state, label):
    start = RobotState(model)
    start.set_joint_group_positions(sp.ARM, start_joints)
    start.update()
    arm.set_start_state(robot_state=start)
    arm.set_goal_state(robot_state=goal_state)
    r = arm.plan()
    if not (r and r.trajectory is not None):
        print(f"SEQUENCE_FAILED at arm stage: {label}")
        os._exit(4)
    print(f"STAGE {label:22s} waypoints={sp.waypoints(r)}")
    return list(goal_state.get_joint_group_positions(sp.ARM))


def plan_gripper(grip, model, from_name, to_name, label):
    start = RobotState(model)
    start.set_to_default_values(sp.GRIPPER, from_name)
    start.update()
    grip.set_start_state(robot_state=start)
    grip.set_goal_state(configuration_name=to_name)
    r = grip.plan()
    if not (r and r.trajectory is not None):
        print(f"SEQUENCE_FAILED at gripper stage: {label}")
        os._exit(5)
    print(f"STAGE {label:22s} waypoints={sp.waypoints(r)}")


def main():
    cfg = sp.build_config()
    moveit = MoveItPy(node_name="moveit_py", config_dict=cfg.to_dict())
    model = moveit.get_robot_model()
    arm = moveit.get_planning_component(sp.ARM)
    grip = moveit.get_planning_component(sp.GRIPPER)

    # Report the ENV->base transform of the real scene objects.
    print("=== scene ENV->base transform + reachability ===")
    for name, env in [("block", SCENE_BLOCK_ENV), ("box", SCENE_BOX_ENV)]:
        b = sp.env_to_base(env)
        r = sp.find_grasp_ik(model, tuple(b))
        print(f"  {name:5s} env={env} -> base={tuple(round(v,3) for v in b)} "
              f"radius={np.hypot(b[0],b[1]):.3f} top_down_reachable={r is not None}")

    # Pick reachable block + place targets (base frame).
    block_xyz, (grasp_state, grasp_joints) = choose(model, sp.env_to_base(SCENE_BLOCK_ENV), BLOCK_FALLBACK, "\nblock")
    place_xyz, (place_state, _) = choose(model, sp.env_to_base(SCENE_BOX_ENV), PLACE_FALLBACK, "place")

    pre, pre_z = reachable_above(model, block_xyz)
    lift, lift_z = reachable_above(model, block_xyz)
    over, over_z = reachable_above(model, place_xyz)
    if not (pre and lift and over):
        print("SEQUENCE_FAILED: pre-grasp/lift/over-box waypoint not reachable")
        os._exit(3)
    print(f"standoff heights: pre/lift z={pre_z:.3f} over z={over_z:.3f}")

    print("\n=== planning pick-place sequence ===")
    rest = RobotState(model)
    rest.set_to_default_values(sp.ARM, "rest")
    rest.update()
    j = list(rest.get_joint_group_positions(sp.ARM))

    j = plan_arm(arm, model, j, pre[0], "rest->pre-grasp")
    j = plan_arm(arm, model, j, grasp_state, "pre-grasp->grasp")
    plan_gripper(grip, model, "open", "closed", "close gripper")
    j = plan_arm(arm, model, j, lift[0], "grasp->lift")
    j = plan_arm(arm, model, j, over[0], "lift->over-box")
    j = plan_arm(arm, model, j, place_state, "over-box->place")
    plan_gripper(grip, model, "closed", "open", "open gripper")
    j = plan_arm(arm, model, j, rest, "place->retreat")

    print("\nSEQUENCE_OK")
    os._exit(0)


if __name__ == "__main__":
    main()
