"""Solve the rigid transform mapping URDF-base FK -> Isaac ENV gripper poses.

Validates the env<->base transform and checks joint-convention consistency.
Reads isaac_fk.json + urdf_fk.json from a directory (argv[1], default ".").
"""
import json
import sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
iz = json.load(open(f"{D}/isaac_fk.json"))
ur = json.load(open(f"{D}/urdf_fk.json"))

P_env = np.array([r["gripper_pos"] for r in iz["records"]])   # ENV frame
P_gl = np.array([r["gripper_link_pos"] for r in ur])          # URDF base frame
P_gf = np.array([r["gripper_frame_pos"] for r in ur])
base_pos = np.array(iz["records"][0]["base_pos"])
w, x, y, z = iz["records"][0]["base_quat_wxyz"]


def quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def rigid_fit(A, B):  # B ~= R@A + t
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - R @ cA
    dist = np.sqrt(((B - ((R @ A.T).T + t))**2).sum(1))
    return R, t, dist.mean(), dist.max()


def euler_zyx_deg(R):
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw


print("=== rigid fit: URDF base-frame FK  ->  Isaac ENV gripper ===")
best = None
for name, P in [("gripper_link", P_gl), ("gripper_frame_link", P_gf)]:
    R, t, rms, mx = rigid_fit(P, P_env)
    r, p, yw = euler_zyx_deg(R)
    print(f"\n[{name}]  residual rms={rms*1000:.2f} mm  max={mx*1000:.2f} mm")
    print(f"  base->env translation t = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
    print(f"  base->env rotation  rpy = ({r:.2f}, {p:.2f}, {yw:.2f}) deg")
    if best is None or rms < best[0]:
        best = (rms, name, R, t)

print("\n=== reference (Isaac base body pose) ===")
print(f"  Isaac base_pos (ENV) = ({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f})")
print(f"  Isaac base rpy       = {tuple(round(v,2) for v in euler_zyx_deg(quat_to_R(w,x,y,z)))} deg")

rms, name, R, t = best
r, p, yw = euler_zyx_deg(R)
print(f"\n=== VERDICT (best link: {name}) ===")
print(f"  residual small (conventions match): {rms < 2e-3}  (rms={rms*1000:.2f} mm)")
print(f"  BASE_ORIGIN_ENV = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})  rotation rpy=({r:.2f},{p:.2f},{yw:.2f})")
