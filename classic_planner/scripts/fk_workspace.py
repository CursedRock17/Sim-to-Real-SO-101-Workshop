"""Map the SO-101's reachable HORIZONTAL-grasp workspace by forward kinematics.

The SO-101 is a 5-DOF arm: base yaw (shoulder_pan) + a planar 3R pitch chain
(shoulder_lift, elbow_flex, wrist_flex, parallel axes) + wrist_roll. KDL IK fails
on a fully-constrained horizontal (side) grasp because those 6-DOF orientations
lie off the arm's achievable manifold. So instead of asking IK for an orientation,
we FORWARD-sample the joints, run FK, and keep the configs that happen to land
near-horizontal at the block's grip height. FK never fails, so this maps the true
reachable set and hands back exact joint configs -- no IK involved.

A hand-rolled, vectorized numpy FK does the millions-of-samples sweep; it is first
VALIDATED against MoveIt's own RobotState FK at several configs (residual printed)
so we trust it. Run in teleop-moveit: source-ros + colcon build + source install.
"""
import functools
import json
import os
import numpy as np

print = functools.partial(print, flush=True)

import so101_planning as sp

_KEEPALIVE = []   # hold MoveItPy so it is not GC'd mid-run (teardown segfaults)

# ---- URDF chain base_link -> gripper_frame_link (xyz, rpy, revolute?) ----------
# Fixed-axis XYZ rpy. Revolute joints rotate about local +Z by the joint value.
CHAIN = [
    # name,           xyz,                                  rpy,                          revolute
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624),    (3.14159, 4.18253e-17, -3.14159), True),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542),    (-1.5708, -1.5708, 0.0),          True),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),      (0.0, 0.0, 1.5708),               True),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),       (0.0, 0.0, -1.5708),              True),
    ("wrist_roll",    (5.55112e-17, -0.0611, 0.0181),       (1.5708, 0.0486795, 3.14159),     True),
    ("gripper_frame", (-0.0079, -0.000218121, -0.0981274),  (0.0, 3.14159, 0.0),              False),
]
LIMITS = {  # rad, from the URDF <limit> tags
    "shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

GRIP_OFFSET = 0.03          # grip point sits this far back from gripper_frame_link
TARGET_ENV = (0.11, 0.02, 0.05)   # block center (grip point) we would like to reach
HORIZ_TOL_DEG = 20.0        # |approach elevation| below this counts as "horizontal"
Z_BAND = 0.015              # grip height must be within this of the target z


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r); cp, sp_ = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp_], [0, 1, 0], [-sp_, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotz_batch(q):
    """(N,) angles -> (N,3,3) rotations about local +Z."""
    c, s = np.cos(q), np.sin(q)
    R = np.zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1.0
    return R


def fk_batch(Q):
    """Q: (N,5) arm joint values -> (pos (N,3), R (N,3,3)) of gripper_frame_link."""
    N = Q.shape[0]
    pos = np.zeros((N, 3))
    R = np.tile(np.eye(3), (N, 1, 1))
    ji = 0
    for name, xyz, rpy, revolute in CHAIN:
        Tt = np.array(xyz)
        Rf = rpy_to_R(rpy)
        # accumulate fixed origin: p += R @ Tt ; R = R @ Rf
        pos = pos + np.einsum("nij,j->ni", R, Tt)
        R = np.einsum("nij,jk->nik", R, Rf)
        if revolute:
            Rq = rotz_batch(Q[:, ji]); ji += 1
            R = np.einsum("nij,njk->nik", R, Rq)
    return pos, R


def validate_against_moveit():
    """Confirm the numpy FK matches MoveIt RobotState FK before trusting the sweep."""
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    moveit = MoveItPy(node_name="moveit_py", config_dict=sp.build_config().to_dict())
    _KEEPALIVE.append(moveit)   # never let this be collected (avoids teardown segfault)
    model = moveit.get_robot_model()
    rs = RobotState(model)
    test_cfgs = {
        "home": [0, 0, 0, 0, 0],
        "rest": [0, -1.5, 1.5, 1.0, 0],
        "rand1": [0.5, -0.8, 0.9, -0.4, 1.2],
        "rand2": [-1.1, 0.6, -1.0, 0.7, -0.9],
    }
    print("=== FK validation (numpy vs MoveIt RobotState) ===")
    worst = 0.0
    for name, q in test_cfgs.items():
        rs.set_joint_group_positions(sp.ARM, np.array(q, float)); rs.update()
        T = np.array(rs.get_global_link_transform(sp.TIP))
        p_moveit = T[:3, 3]; R_moveit = T[:3, :3]
        p_np, R_np = fk_batch(np.array([q], float))
        dp = np.linalg.norm(p_np[0] - p_moveit) * 1000.0
        dR = np.linalg.norm(R_np[0] - R_moveit)
        worst = max(worst, dp)
        app_np = R_np[0][:, 2]; app_mv = R_moveit[:, 2]
        print(f"  {name:6s} dpos={dp:6.3f} mm  dR={dR:.4f}  approach_np={np.round(app_np,3)}")
    print(f"  worst position residual: {worst:.3f} mm")
    return worst < 1.0


def main():
    ok = validate_against_moveit()
    if not ok:
        print("FK_VALIDATION_FAILED -- numpy FK disagrees with MoveIt; aborting sweep")
        os._exit(2)
    print("FK validated.\n")

    # ---- big vectorized sweep ----
    rng = np.random.default_rng(0)
    N = 3_000_000
    Q = np.empty((N, 5))
    for k, jn in enumerate(ARM_JOINTS):
        lo, hi = LIMITS[jn]
        Q[:, k] = rng.uniform(lo, hi, N)
    pos, R = fk_batch(Q)
    approach = R[:, :, 2]                       # gripper_frame_link local +Z in base frame
    grip = pos - GRIP_OFFSET * approach         # grip point (between the fingers)
    grip_env = grip + sp.BASE_ORIGIN_ENV        # back to the ENV frame

    elev = np.degrees(np.arcsin(np.clip(approach[:, 2], -1, 1)))   # approach elevation
    horiz = np.abs(elev) < HORIZ_TOL_DEG
    at_h = np.abs(grip_env[:, 2] - TARGET_ENV[2]) < Z_BAND
    keep = horiz & at_h
    print(f"=== sweep: {N:,} samples, {keep.sum():,} near-horizontal at grip height ===")
    if keep.sum() == 0:
        print("NO near-horizontal configs at grip height -- widen tolerances"); os._exit(3)

    gk = grip_env[keep]
    print(f"reachable horizontal-grasp workspace (ENV frame, z~{TARGET_ENV[2]}):")
    print(f"  x: [{gk[:,0].min():.3f}, {gk[:,0].max():.3f}]  "
          f"y: [{gk[:,1].min():.3f}, {gk[:,1].max():.3f}]")
    r = np.hypot(gk[:, 0] - sp.BASE_ORIGIN_ENV[0], gk[:, 1] - sp.BASE_ORIGIN_ENV[1])
    print(f"  planar radius from base: [{r.min():.3f}, {r.max():.3f}] m")

    # jaw-open axis per sample: gripper joint about local +Z of gripper_link.
    # gripper_frame is gripper_link rotated by rpy(0,pi,0); undo it to get gripper_link R.
    Rf_inv = rpy_to_R((0.0, 3.14159, 0.0)).T
    Rgj = rpy_to_R((1.5708, -5.24284e-08, -1.41553e-15))

    def jaw_axis_of(mask):
        Rgl = np.einsum("nij,jk->nik", R[mask], Rf_inv)
        return np.einsum("nij,jk,k->ni", Rgl, Rgj, np.array([0.0, 0.0, 1.0]))

    jaw_axis = jaw_axis_of(keep)
    appk = approach[keep]

    # Forward approach: the tool must point AWAY from the base in the horizontal
    # plane (natural side grasp -- hand behind the block, not reaching back over the
    # base). outward = grip - base (xy); forward means approach . outward > 0.
    base_xy = sp.BASE_ORIGIN_ENV[:2]
    outward = gk[:, :2] - base_xy
    outward /= (np.linalg.norm(outward, axis=1, keepdims=True) + 1e-9)
    app_xy = appk[:, :2]
    fwd_dot = np.sum((app_xy / (np.linalg.norm(app_xy, axis=1, keepdims=True) + 1e-9)) * outward, axis=1)
    forward = fwd_dot > 0.3

    # A "clean" horizontal grasp: level approach AND jaws closing level (jaw_z ~ 0 ->
    # fingers left/right of the block, both clear of the mat), pointing forward.
    clean = (np.abs(elev[keep]) < 6.0) & (np.abs(jaw_axis[:, 2]) < 0.3) & forward
    print(f"\nclean FORWARD horizontal grasps (|elev|<6, jaws level): {clean.sum():,}")
    if clean.sum() > 0:
        gc = gk[clean]
        rc = np.hypot(gc[:, 0] - base_xy[0], gc[:, 1] - base_xy[1])
        print(f"  clean-grasp locations  x: [{gc[:,0].min():.3f}, {gc[:,0].max():.3f}]  "
              f"y: [{gc[:,1].min():.3f}, {gc[:,1].max():.3f}]  radius [{rc.min():.3f}, {rc.max():.3f}]")

    # Best achievable elevation as a function of forward radius (horizontality map):
    print("  best achievable |elev| vs forward radius (jaws-level, forward grasps):")
    rr = np.hypot(gk[:, 0] - base_xy[0], gk[:, 1] - base_xy[1])
    lvl = forward & (np.abs(jaw_axis[:, 2]) < 0.3)
    for r0 in (0.14, 0.18, 0.22, 0.26, 0.30, 0.34):
        band = lvl & (np.abs(rr - r0) < 0.02)
        if band.sum():
            print(f"    r={r0:.2f} m: min|elev|={np.abs(elev[keep][band]).min():4.1f} deg  ({band.sum()} configs)")
        else:
            print(f"    r={r0:.2f} m: none")

    # Recommended block placement: nearest clean forward-horizontal grip to current.
    cur = np.array(TARGET_ENV)
    pool = clean if clean.sum() > 0 else (forward & (np.abs(jaw_axis[:, 2]) < 0.3))
    gpool = gk[pool]
    rec_grip = gpool[int(np.argmin(np.linalg.norm(gpool - cur, axis=1)))]
    print(f"\nRECOMMENDED block placement (ENV): "
          f"[{rec_grip[0]:.3f}, {rec_grip[1]:.3f}, {rec_grip[2]:.3f}]  "
          f"(moved {np.linalg.norm(rec_grip-cur)*100:.1f} cm from current)")

    # Lift set: configs that put the grip point ~8 cm above, still roughly horizontal.
    LIFT_DZ = 0.08
    lift_mask = (np.abs(elev) < 30.0) & (np.abs(grip_env[:, 2] - (TARGET_ENV[2] + LIFT_DZ)) < 0.02)
    Ql, gl = Q[lift_mask], grip_env[lift_mask]

    def nearest_lift(qgrasp, xy):
        sel = np.where(np.hypot(gl[:, 0] - xy[0], gl[:, 1] - xy[1]) < 0.025)[0]
        if len(sel) == 0:
            return None
        j = sel[int(np.argmin(np.linalg.norm(Ql[sel] - qgrasp, axis=1)))]
        return [float(v) for v in Ql[j]]

    # Pre-grasp set = same horizontal set at grip height (keep). A pre-grasp for a
    # given grasp is a config whose grip point sits PRE_BACK behind the block along
    # -approach (jaws clear), closest in joint space so the forward drive is short.
    PRE_BACK = 0.06
    gkeep_xy = gk[:, :2]

    def nearest_pre(qgrasp, grip_xy, adir_xy):
        want = grip_xy - PRE_BACK * adir_xy
        sel = np.where(np.hypot(gkeep_xy[:, 0] - want[0], gkeep_xy[:, 1] - want[1]) < 0.02)[0]
        if len(sel) == 0:
            return None
        j = sel[int(np.argmin(np.linalg.norm(Qk[sel] - qgrasp, axis=1)))]
        return [float(v) for v in Qk[j]]

    # Emit candidates near the recommended grip: clean+forward, deduped, w/ pre+lift.
    Qk = Q[keep]
    idx_pool = np.where(pool)[0]
    order = idx_pool[np.argsort(np.linalg.norm(gk[idx_pool] - rec_grip, axis=1))]
    cands, seen = [], []
    for i in order:
        qi = Qk[i]
        if any(np.linalg.norm(qi - s) < 0.15 for s in seen):
            continue
        lift = nearest_lift(qi, gk[i][:2])
        adir_xy = appk[i][:2] / (np.linalg.norm(appk[i][:2]) + 1e-9)
        pre = nearest_pre(qi, gk[i][:2], adir_xy)
        if lift is None or pre is None:
            continue
        seen.append(qi)
        cands.append({
            "joints": [float(v) for v in qi], "pre_joints": pre, "lift_joints": lift,
            "grip_env": [float(v) for v in gk[i]], "approach": [float(v) for v in appk[i]],
            "jaw_axis": [float(v) for v in jaw_axis[i]], "elev_deg": float(elev[keep][i]),
        })
        if len(cands) >= 30:
            break

    print(f"\ntop clean candidates near recommended placement:")
    for c in cands[:8]:
        print(f"  elev={c['elev_deg']:+5.1f} grip={np.round(c['grip_env'],3)} "
              f"approach={np.round(c['approach'],2)} jaw_z={c['jaw_axis'][2]:+.2f} "
              f"q={np.round(c['joints'],2)}")

    out = {"grip_offset": GRIP_OFFSET, "target_env": list(TARGET_ENV),
           "recommended_grip_env": [float(v) for v in rec_grip], "lift_dz": LIFT_DZ,
           "horiz_tol_deg": HORIZ_TOL_DEG, "joint_order": ARM_JOINTS, "candidates": cands}
    path = os.environ.get("CALIB_DIR", "/root/smoke") + "/fk_candidates.json"
    json.dump(out, open(path, "w"))
    print(f"\nFK_WORKSPACE_OK candidates={len(cands)} -> {path}")
    os._exit(0)


if __name__ == "__main__":
    main()
