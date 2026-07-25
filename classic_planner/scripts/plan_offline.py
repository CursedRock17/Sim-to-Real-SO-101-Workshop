"""Plan the block->box pick-place and emit a single 30 Hz, 6-DOF joint trajectory
(Isaac joint order) for the Isaac record agent to execute. Writes trajectory.json.

Fixed block/box pose for the first test run; the live loop will take poses per
episode. Run in teleop-moveit (source-ros + colcon build + source install).
"""
import bisect
import json
import os
import numpy as np

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
import so101_planning as sp

FPS = 30
DT = 1.0 / FPS
OUT = os.environ.get("CALIB_DIR", "/root/smoke") + "/trajectory.json"

# Isaac joint order and the URDF-arm -> Isaac-index map (calibration: 1:1).
ISAAC_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
URDF2ISAAC = {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 4}

# Gripper (Jaw) command values (tuned from the sim grip search). JAW_OPEN is wide
# so the fingers clear the block on descent; JAW_CLOSE clamps it.
JAW_OPEN = 1.5
JAW_CLOSE = -0.5

SCENE_BLOCK_ENV = (0.16, 0.06, 0.05)
SCENE_BOX_ENV = (0.10, 0.20, 0.06)


def resample(rtraj, jaw):
    """RobotTrajectory -> list of 6-DOF Isaac-order targets at 30 Hz (jaw held)."""
    jt = rtraj.get_robot_trajectory_msg().joint_trajectory
    names = list(jt.joint_names)
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points]
    poss = [list(p.positions) for p in jt.points]
    frames = []
    t = 0.0
    while t <= times[-1] + 1e-9:
        i = min(max(bisect.bisect_right(times, t) - 1, 0), len(times) - 2)
        a = 0.0 if times[i + 1] == times[i] else (t - times[i]) / (times[i + 1] - times[i])
        interp = [poss[i][k] * (1 - a) + poss[i + 1][k] * a for k in range(len(names))]
        six = [0.0] * 6
        for k, nm in enumerate(names):
            six[URDF2ISAAC[nm]] = interp[k]
        six[5] = jaw
        frames.append(six)
        t += DT
    return frames


def ramp_jaw(arm5, j0, j1, dur=0.5):
    n = max(1, int(dur / DT))
    return [list(arm5) + [j0 + (j1 - j0) * (i + 1) / n] for i in range(n)]


def main():
    moveit = MoveItPy(node_name="moveit_py", config_dict=sp.build_config().to_dict())
    model = moveit.get_robot_model()
    arm = moveit.get_planning_component(sp.ARM)

    # gripper_frame_link is a tool frame past the fingertips, so drive the grasp
    # target GRASP_DZ below the block so the actual fingers reach it.
    GRASP_DZ = -0.03
    bx, by, bz = sp.env_to_base(SCENE_BLOCK_ENV)
    block = (bx, by, bz + GRASP_DZ)
    place = tuple(sp.env_to_base(SCENE_BOX_ENV))
    grasp = sp.find_grasp_ik(model, block)
    over = sp.find_grasp_ik(model, (place[0], place[1], place[2] + 0.05))
    place_s = sp.find_grasp_ik(model, place)
    for nm, r in [("grasp", grasp), ("over", over), ("place", place_s)]:
        if r is None:
            print("PLAN_FAILED: unreachable", nm); os._exit(2)
    gq = grasp[2]  # grasp orientation, reused for the pre-grasp + vertical approach
    # pre-grasp uses the SAME orientation as the grasp so the wrist doesn't snap at
    # the descent start (which knocks the block).
    pre = sp.ik_pose(model, (bx, by, bz + 0.05), gq)
    if pre is None:
        print("PLAN_FAILED: unreachable pre"); os._exit(2)

    def plan(start_joints, goal_state):
        s = RobotState(model); s.set_joint_group_positions(sp.ARM, start_joints); s.update()
        arm.set_start_state(robot_state=s)
        arm.set_goal_state(robot_state=goal_state)
        r = arm.plan()
        if not (r and r.trajectory is not None):
            print("PLAN_FAILED"); os._exit(3)
        return r.trajectory, list(goal_state.get_joint_group_positions(sp.ARM))

    def straight(x, y, z0, z1, quat, jaw, n=20):
        """Vertical Cartesian move at fixed orientation -> 6-DOF frames (avoids the
        OMPL lateral swing that knocks the block on approach)."""
        frames, last = [], None
        for h in np.linspace(z0, z1, n):
            r = sp.ik_pose(model, (x, y, h), quat)
            if r is None:
                print("PLAN_FAILED: straight ik z=%.3f" % h); os._exit(4)
            frames.append(list(r[1]) + [jaw]); last = r[1]
        return frames, last

    rest = RobotState(model); rest.set_to_default_values(sp.ARM, "rest"); rest.update()
    j = list(rest.get_joint_group_positions(sp.ARM))
    traj = []

    t1, j = plan(j, pre[0]); traj += resample(t1, JAW_OPEN)                    # rest -> pre (free)
    d, j = straight(bx, by, bz + 0.05, bz + GRASP_DZ, gq, JAW_OPEN); traj += d  # vertical descent
    traj += ramp_jaw(j, JAW_OPEN, JAW_CLOSE)                                    # close on block
    u, j = straight(bx, by, bz + GRASP_DZ, bz + 0.05, gq, JAW_CLOSE); traj += u # vertical lift
    t4, j = plan(j, over[0]); traj += resample(t4, JAW_CLOSE)                   # transport (free)
    t5, j = plan(j, place_s[0]); traj += resample(t5, JAW_CLOSE)               # over -> place
    traj += ramp_jaw(place_s[1], JAW_CLOSE, JAW_OPEN)                           # release
    t6, j = plan(j, rest); traj += resample(t6, JAW_OPEN)                       # retreat (free)

    out = {"fps": FPS, "joint_order": ISAAC_ORDER, "block_env": SCENE_BLOCK_ENV,
           "box_env": SCENE_BOX_ENV, "jaw_open": JAW_OPEN, "jaw_close": JAW_CLOSE,
           "grasp_config": list(grasp[1]) + [JAW_OPEN],   # 6-DOF at the grasp pose
           "lift_config": list(lift[1]) + [JAW_CLOSE],     # 6-DOF after lifting
           "grasp_target_base": list(block),
           "trajectory": traj}
    json.dump(out, open(OUT, "w"))
    print(f"PLAN_OK frames={len(traj)} duration_s={len(traj)*DT:.2f} -> {OUT}")
    os._exit(0)


if __name__ == "__main__":
    main()
