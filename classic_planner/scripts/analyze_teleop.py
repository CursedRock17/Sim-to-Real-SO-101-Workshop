"""Decode a real teleop episode into Isaac radians and run it through so101_fk to
reveal what the working grasp actually is: approach elevation, grip point (ENV),
and the joint config -- ground truth to replicate instead of reinventing.
"""
import json, math
import numpy as np
import so101_fk as fk

# LeRobot [-100,100] -> per-joint degree range -> radians (from lerobot_interface.py)
MINS = np.array([-110, -100, -100, -95, -160.0])   # pan,lift,elbow,wflex,wroll
MAXS = np.array([110, 100, 90, 95, 160.0])


def to_rad(raw5):
    n = (np.asarray(raw5, float) + 100.0) / 200.0
    return (MINS + n * (MAXS - MINS)) * math.pi / 180.0


ep = json.load(open("ep0.json"))["ep0"]
grip = np.array([r[5] for r in ep])
arm = np.array([to_rad(r[:5]) for r in ep])
N = len(ep)

# grasp = first frame after the open phase where grip settles to the held value (~3.5)
opened = np.where(grip > 20)[0]
o0 = opened[0]
held = np.where((grip < 12) & (np.arange(N) > o0))[0]
grasp_f = held[0]
print(f"frames={N}  open@{o0}  grasp(close)@{grasp_f}  grip_held~{np.median(grip[grasp_f:grasp_f+40]):.1f}")

print("\napproach -> grasp geometry (ENV frame):")
for f in [max(0, grasp_f-30), max(0, grasp_f-20), max(0, grasp_f-10), grasp_f, grasp_f+15, grasp_f+30]:
    q = arm[f]
    pos, R = fk.fk(q)
    app = R[:, 2]
    elev = math.degrees(math.asin(np.clip(-app[2], -1, 1)))   # + = pointing down
    grip_env = fk.base_to_env(pos - fk.GRIP_OFFSET * app)
    frame_env = fk.base_to_env(pos)
    tag = " <-- GRASP" if f == grasp_f else ""
    print(f"  f={f:3d} grip{['%+.0f'%grip[f]]} elev={elev:+5.1f}  grip_env={np.round(grip_env,3)} "
          f"frame_z={frame_env[2]:.3f} q={np.round(q,2)}{tag}")

# summarize the held (carry) height to see how high it lifts
g = arm[grasp_f]
pos, R = fk.fk(g)
print(f"\nGRASP config (rad, Isaac order pan..wroll): {np.round(g,4).tolist()}")
print(f"GRASP approach={np.round(R[:,2],3)}  grip_env={np.round(fk.base_to_env(pos-fk.GRIP_OFFSET*R[:,2]),3)}")
