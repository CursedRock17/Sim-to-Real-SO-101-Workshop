"""Generate reachable horizontal side-grasp candidates (grasp + straight-up lift
config) at the block grip point, for the Isaac grip tester. Run in teleop-moveit."""
import json, os
import numpy as np
from moveit.planning import MoveItPy
import so101_planning as sp

GRIP_ENV = (0.11, 0.02, 0.06)   # mid-height of the standing block
OFFSET = 0.03                    # tool driven past the block along the approach
LIFT_DZ = 0.08
JAW_OPEN = 1.5

moveit = MoveItPy(node_name="moveit_py", config_dict=sp.build_config().to_dict())
model = moveit.get_robot_model()
grip_base = np.array(sp.env_to_base(GRIP_ENV))
cands = sp.find_side_grasps(model, tuple(grip_base), offset=OFFSET)
print(f"reachable side grasps: {len(cands)}")

final = []
for j, q, a in cands:
    lift_target = grip_base + np.array([0.0, 0.0, LIFT_DZ]) + OFFSET * np.array(a)
    lr = sp.ik_pose(model, tuple(lift_target), q)
    if lr is not None:
        final.append({"joints": j, "lift_joints": lr[1], "quat": q, "approach": a})
print(f"with reachable lift: {len(final)}")

out = {"grip_env": GRIP_ENV, "grip_base": grip_base.tolist(), "offset": OFFSET,
       "jaw_open": JAW_OPEN, "candidates": final}
json.dump(out, open(os.environ.get("CALIB_DIR", "/root/smoke") + "/side_candidates.json", "w"))
print("GEN_DONE")
os._exit(0)
