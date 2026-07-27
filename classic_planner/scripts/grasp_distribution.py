"""Extract the grasp-point distribution across all teleop demo episodes: for each
episode find the grasp (open->held transition), FK the arm config, and collect the
grip point (ENV) + approach. Reports the zone to place/randomize the block in.
"""
import json, math
import numpy as np
import so101_fk as fk

MINS = np.array([-110, -100, -100, -95, -160.0]); MAXS = np.array([110, 100, 90, 95, 160.0])


def to_rad(raw5):
    n = (np.asarray(raw5, float) + 100.0) / 200.0
    return (MINS + n * (MAXS - MINS)) * math.pi / 180.0


eps = json.load(open("allep.json"))
grips_env, elevs, gzs, grasp_qs = [], [], [], []
for k, frames in eps.items():
    g = np.array([r[5] for r in frames]); N = len(frames)
    op = np.where(g > 20)[0]
    if len(op) == 0:
        continue
    held = np.where((g < 12) & (np.arange(N) > op[0]))[0]
    if len(held) == 0:
        continue
    f = held[0]
    q = to_rad(frames[f][:5])
    pos, R = fk.fk(q); app = R[:, 2]
    grip_env = fk.base_to_env(pos - fk.GRIP_OFFSET * app)
    grips_env.append(grip_env); elevs.append(math.degrees(math.asin(np.clip(-app[2], -1, 1))))
    gzs.append(grip_env[2]); grasp_qs.append(q)

G = np.array(grips_env); E = np.array(elevs)
print(f"episodes with a clean grasp: {len(G)}/{len(eps)}")
print(f"grasp approach elevation: mean {E.mean():.1f} deg  range [{E.min():.1f}, {E.max():.1f}]")
print(f"grip point ENV x: mean {G[:,0].mean():.3f}  range [{G[:,0].min():.3f}, {G[:,0].max():.3f}]")
print(f"grip point ENV y: mean {G[:,1].mean():.3f}  range [{G[:,1].min():.3f}, {G[:,1].max():.3f}]")
print(f"grip point ENV z: mean {G[:,2].mean():.3f}  range [{G[:,2].min():.3f}, {G[:,2].max():.3f}]")
r = np.hypot(G[:, 0] - fk.BASE_ORIGIN_ENV[0], G[:, 1] - fk.BASE_ORIGIN_ENV[1])
print(f"grip radius from base: mean {r.mean():.3f}  range [{r.min():.3f}, {r.max():.3f}]")
# median grasp config (representative)
med = np.median(np.array(grasp_qs), axis=0)
print(f"median grasp config (rad): {np.round(med,3).tolist()}")
print(f"median grip point ENV: {np.round(np.median(G,axis=0),3).tolist()}")
json.dump({"grips_env": G.tolist(), "elev_mean": float(E.mean()),
           "median_grasp_q": med.tolist(), "median_grip_env": np.median(G, axis=0).tolist(),
           "x_range": [float(G[:,0].min()), float(G[:,0].max())],
           "y_range": [float(G[:,1].min()), float(G[:,1].max())]},
          open("grasp_zone.json", "w"))
print("ZONE_SAVED grasp_zone.json")
