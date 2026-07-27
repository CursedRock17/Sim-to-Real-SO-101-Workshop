"""Standalone SO-101 forward kinematics + numerical IK (pure numpy, no ROS/MoveIt).

KDL fails on horizontal (side) grasps because a fully-constrained 6-DOF orientation
is off a 5-DOF arm's achievable manifold. But a 5-DOF arm has exactly enough DOF for
3 position + 2 approach-DIRECTION constraints (the jaw roll about the approach is left
to fall out). This module solves that reduced problem by damped-least-squares on the
analytic FK, so we get horizontal grasp configs AND smooth straight-line Cartesian
approaches -- and it imports nothing from ROS, so it runs in Isaac's python too.

The FK here is byte-for-byte the chain validated against MoveIt RobotState FK to
0.000 mm in fk_workspace.py.
"""
import numpy as np

# base_link -> gripper_frame_link (xyz, rpy fixed-axis XYZ, revolute about local +Z)
CHAIN = [
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624),    (3.14159, 4.18253e-17, -3.14159), True),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542),    (-1.5708, -1.5708, 0.0),          True),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),      (0.0, 0.0, 1.5708),               True),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),       (0.0, 0.0, -1.5708),              True),
    ("wrist_roll",    (5.55112e-17, -0.0611, 0.0181),       (1.5708, 0.0486795, 3.14159),     True),
    ("gripper_frame", (-0.0079, -0.000218121, -0.0981274),  (0.0, 3.14159, 0.0),              False),
]
LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385])
UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121])
BASE_ORIGIN_ENV = np.array([-0.0658, 0.0208, 0.0325])
GRIP_OFFSET = 0.03            # grip point sits this far back from gripper_frame_link
# Finger-gap centre relative to gripper_frame_link, in the TOOL frame (measured in
# Isaac via fk_center_probe.py: the TCP is offset laterally from where the fingers
# actually cage an object). grasp_centre = frame_pos + R @ GRASP_CENTER_TOOL.
GRASP_CENTER_TOOL = np.array([-0.0175, 0.0041, -0.0011])


def env_to_base(p):
    return np.asarray(p, float) - BASE_ORIGIN_ENV


def base_to_env(p):
    return np.asarray(p, float) + BASE_ORIGIN_ENV


def _rpy(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


_FIXED = [( np.array(xyz), _rpy(rpy), rev) for _, xyz, rpy, rev in CHAIN]


def fk(q):
    """q: (5,) arm joints -> (pos (3,), R (3,3)) of gripper_frame_link in base frame."""
    pos = np.zeros(3); R = np.eye(3); ji = 0
    for Tt, Rf, rev in _FIXED:
        pos = pos + R @ Tt
        R = R @ Rf
        if rev:
            c, s = np.cos(q[ji]), np.sin(q[ji]); ji += 1
            R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return pos, R


def approach_of(q):
    return fk(q)[1][:, 2]        # gripper_frame_link local +Z in base frame


def _residual(q, tgt_pos, tgt_app, wp, wa):
    pos, R = fk(q)
    return np.concatenate([wp * (tgt_pos - pos), wa * (tgt_app - R[:, 2])])


def _jac(q, tgt_pos, tgt_app, wp, wa, eps=1e-6):
    r0 = _residual(q, tgt_pos, tgt_app, wp, wa)
    J = np.zeros((6, 5))
    for i in range(5):
        dq = q.copy(); dq[i] += eps
        J[:, i] = (_residual(dq, tgt_pos, tgt_app, wp, wa) - r0) / eps
    return J, r0


def _ik_once(q, tgt_pos, tgt_app, iters, lam, wp, wa, pos_tol, app_tol):
    q = np.clip(q, LOWER, UPPER)
    for _ in range(iters):
        pos, R = fk(q)
        if np.linalg.norm(tgt_pos - pos) < pos_tol and np.linalg.norm(tgt_app - R[:, 2]) < app_tol:
            return q, True
        J, r = _jac(q, tgt_pos, tgt_app, wp, wa)
        # J = d(residual)/dq, so the LM step that reduces r is -(J^T J + lam^2 I)^-1 J^T r
        dq = -np.linalg.solve(J.T @ J + (lam ** 2) * np.eye(5), J.T @ r)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, LOWER, UPPER)
    pos, R = fk(q)
    ok = (np.linalg.norm(tgt_pos - pos) < pos_tol and np.linalg.norm(tgt_app - R[:, 2]) < app_tol)
    return q, ok


def ik(tgt_pos, tgt_app, seed, iters=200, lam=0.05, wp=1.0, wa=0.05,
       pos_tol=5e-4, app_tol=1e-2, restarts=12, rng=None):
    """Solve for arm joints placing gripper_frame_link at tgt_pos with approach
    (local +Z) along unit tgt_app. Roll about the approach is free (5-DOF exact).
    Tries `seed` first, then random restarts. Returns (q, ok), joint-limit clipped."""
    tgt_pos = np.asarray(tgt_pos, float)
    tgt_app = np.asarray(tgt_app, float); tgt_app = tgt_app / np.linalg.norm(tgt_app)
    rng = rng or np.random.default_rng(0)
    q, ok = _ik_once(np.asarray(seed, float).copy(), tgt_pos, tgt_app, iters, lam, wp, wa, pos_tol, app_tol)
    if ok:
        return q, True
    for _ in range(restarts):
        s = rng.uniform(LOWER, UPPER)
        q, ok = _ik_once(s, tgt_pos, tgt_app, iters, lam, wp, wa, pos_tol, app_tol)
        if ok:
            return q, True
    return q, False


def _jac_pos(q, tgt, eps=1e-6):
    p0 = tgt - fk(q)[0]
    J = np.zeros((3, 5))
    for i in range(5):
        dq = q.copy(); dq[i] += eps
        J[:, i] = ((tgt - fk(dq)[0]) - p0) / eps
    return J, p0


def ik_pos(tgt_pos, seed, iters=200, lam=0.05, tol=5e-4, restarts=10, rng=None):
    """Position-only IK: place gripper_frame_link at tgt_pos, orientation free (the
    arm tilts as needed to reach high). Damped-least-squares, seeded then restarts.
    Redundancy is resolved toward the seed, so stepping with the previous config as
    seed keeps the carry smooth. Returns (q, ok)."""
    tgt_pos = np.asarray(tgt_pos, float)
    rng = rng or np.random.default_rng(0)

    def once(q):
        q = np.clip(q, LOWER, UPPER)
        for _ in range(iters):
            if np.linalg.norm(tgt_pos - fk(q)[0]) < tol:
                return q, True
            J, r = _jac_pos(q, tgt_pos)
            dq = -np.linalg.solve(J.T @ J + (lam ** 2) * np.eye(5), J.T @ r)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, LOWER, UPPER)
        return q, np.linalg.norm(tgt_pos - fk(q)[0]) < tol

    q, ok = once(np.asarray(seed, float).copy())
    if ok:
        return q, True
    for _ in range(restarts):
        q, ok = once(rng.uniform(LOWER, UPPER))
        if ok:
            return q, True
    return q, False


def grasp_center(q):
    """World/base position where the fingers actually cage an object at config q."""
    pos, R = fk(q)
    return pos + R @ GRASP_CENTER_TOOL


def grasp_ik(center_base, approach, seed, iters=6, **kw):
    """Solve for a grasp config whose FINGER-GAP CENTRE sits at center_base with the
    given approach. Iterates: solve frame IK, measure the finger centre, shift the
    frame target to correct the residual. Returns (q, ok)."""
    center_base = np.asarray(center_base, float)
    approach = np.asarray(approach, float); approach = approach / np.linalg.norm(approach)
    target = center_base.copy()               # first guess: frame at the centre
    q = np.asarray(seed, float).copy()
    for _ in range(iters):
        q, ok = ik(target, approach, q, **kw)
        if not ok:
            return q, False
        err = center_base - grasp_center(q)
        target = target + err
        if np.linalg.norm(err) < 1e-3:
            return q, True
    return q, np.linalg.norm(center_base - grasp_center(q)) < 2e-3


def straight_approach(grip_base, approach, grasp_q, back=0.06, n=16):
    """Configs for a straight Cartesian approach along -approach, from `back` metres
    behind the block in to the block. Marches OUTWARD from the known grasp config
    (guaranteed solvable) so each IK is seeded from its solved neighbour, then
    reverses. grip_base: grip point in base frame; grasp_q: the solved grasp config.
    Returns (configs, ok) with configs[0]=pre-grasp ... configs[-1]=grasp."""
    approach = np.asarray(approach, float); approach = approach / np.linalg.norm(approach)
    grip_base = np.asarray(grip_base, float)
    q = np.asarray(grasp_q, float).copy()
    out = []
    for i, t in enumerate(np.linspace(0.0, back, n)):
        tgt_frame = (grip_base - t * approach) + GRIP_OFFSET * approach
        # t=0 is the grasp: hold the tool exactly horizontal. Backing off, keep the
        # path straight in POSITION but let the tool tilt (position-priority), since
        # a perfectly-horizontal wrist saturates wrist_flex near the reach edge.
        if i == 0:
            q, ok = ik(tgt_frame, approach, q, restarts=0)
        else:
            q, ok = ik(tgt_frame, approach, q, wa=0.01, app_tol=0.4, restarts=0)
        if not ok:
            return [], False
        out.append(q.copy())
    return out[::-1], True
