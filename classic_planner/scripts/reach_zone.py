"""Map the SO-101's top-down-reachable workspace and project it into ENV coords.

Sweeps a grid of positions at block-grasp height, tests top-down IK, and reports
the reachable region in both base_link and ENV frames (via the calibrated
transform) plus a text map -- used to set block-spawn bounds in table_env_cfg.
"""
import math
import os
import numpy as np

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
import so101_planning as sp

GRASP_Z_BASE = 0.02  # block grasp height in base_link frame (~0.052 in ENV)
XS = np.round(np.arange(0.06, 0.305, 0.02), 3)
YS = np.round(np.arange(-0.20, 0.205, 0.02), 3)
YAWS = [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0]

moveit = MoveItPy(node_name="moveit_py", config_dict=sp.build_config().to_dict())
model = moveit.get_robot_model()


def reachable(x, y):
    down = sp.q_axis_angle([1, 0, 0], math.pi)  # tool pointing down
    for yaw in YAWS:
        quat = sp.q_mul(sp.q_axis_angle([0, 0, 1], yaw), down)
        rs = RobotState(model)
        rs.set_to_default_values(sp.ARM, "home")
        rs.update()
        if rs.set_from_ik(sp.ARM, sp.make_pose((x, y, GRASP_Z_BASE), quat), sp.TIP, 0.05):
            return True
    return False


grid = {}
pts = []
for x in XS:
    for y in YS:
        ok = reachable(float(x), float(y))
        grid[(round(float(x), 3), round(float(y), 3))] = ok
        if ok:
            pts.append((float(x), float(y)))

pts = np.array(pts)
env = pts + sp.BASE_ORIGIN_ENV[:2]
z_env = GRASP_Z_BASE + sp.BASE_ORIGIN_ENV[2]
print(f"reachable cells: {len(pts)} / {len(XS)*len(YS)}")
print(f"BASE bbox: x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}]")
print(f"ENV  bbox: x[{env[:,0].min():.3f},{env[:,0].max():.3f}] y[{env[:,1].min():.3f},{env[:,1].max():.3f}]  grasp z_env={z_env:.3f}")

# Largest axis-aligned rectangle (on the grid) centered at a reachable point.
step = 0.02
def cell_ok(x, y):
    return grid.get((round(x, 3), round(y, 3)), False)
best = None
for cx in XS:
    for cy in YS:
        if not cell_ok(cx, cy):
            continue
        hx = 0.0
        while cell_ok(cx - hx - step, cy) and cell_ok(cx + hx + step, cy):
            hx += step
        hy = 0.0
        while all(cell_ok(x, cy - hy - step) and cell_ok(x, cy + hy + step)
                  for x in np.round(np.arange(cx - hx, cx + hx + step/2, step), 3)):
            hy += step
        area = (2*hx + step) * (2*hy + step)
        if best is None or area > best[0]:
            best = (area, cx, cy, hx, hy)

_, cx, cy, hx, hy = best
ecx, ecy = cx + sp.BASE_ORIGIN_ENV[0], cy + sp.BASE_ORIGIN_ENV[1]
print("\n=== recommended block-spawn rectangle (safely inside reach) ===")
print(f"  BASE center=({cx:.3f},{cy:.3f}) half=({hx:.3f},{hy:.3f})")
print(f"  ENV  center=({ecx:.3f},{ecy:.3f}) half=({hx:.3f},{hy:.3f})  z_env={z_env:.3f}")
print(f"  ENV  x[{ecx-hx:.3f},{ecx+hx:.3f}] y[{ecy-hy:.3f},{ecy+hy:.3f}]")

print("\n=== reach map (ENV frame; '#' reachable, columns=x, rows=y top=+y) ===")
print("      x: " + " ".join(f"{x+sp.BASE_ORIGIN_ENV[0]:+.2f}" for x in XS))
for y in reversed(YS):
    row = "".join("  # " if grid[(round(float(x),3), round(float(y),3))] else "  . " for x in XS)
    print(f"y={y+sp.BASE_ORIGIN_ENV[1]:+.2f} " + row)
print("REACH_DONE")
os._exit(0)
