# Gate 0 — wiring it into a collection loop

Gate 0 is generator-agnostic: any loop that executes an oracle episode in Isaac and
records frames calls `evaluate_episode(...)` once per finished episode and only saves
`KEEP` episodes. It needs four signals, three of which the loop already has.

## Signals the loop must provide

| signal | how to get it |
|---|---|
| `plan_ok` | the oracle already knows: `True` if the plan solved and executed without an early abort (e.g. the FK loop's `maxlift < 0.02` abort at `LIFT_CHECK`, or a MoveIt planning failure). |
| `max_lift_m` | already tracked as `maxlift` (`block.z − z0`). |
| `block_final_pos`, `block_final_vel` | already read as `bf[i]`, `bvel[i]` (`block.data.root_pos_w` / `root_lin_vel_w`). |
| `pre_grasp_disp_m` | **new, ~3 lines:** max block-XY drift from its start, measured up to the frame the gripper closes. Clean top-down descent ≈ 0; a bumped block spikes it. |

Tracking `pre_grasp_disp_m` in the exec loop (block starts at `bx,by`, gripper closes
around `CLOSE_FRAME`):

```python
# before the per-frame loop
import math
start_xy = (bx, by)
pre_grasp_disp = 0.0
# inside the per-frame loop, while frame_idx < CLOSE_FRAME:
if frame_idx < CLOSE_FRAME:
    p = block.data.root_pos_w[i]          # env i
    pre_grasp_disp = max(pre_grasp_disp,
                         math.hypot(float(p[0]) - start_xy[0], float(p[1]) - start_xy[1]))
```

## The hook (replaces the inline `ok = block_in_box(...) and maxlift > 0.03`)

```python
from sim_to_real_so101.curation import BoxTaskSpec, ScorecardWriter, evaluate_episode

SPEC = BoxTaskSpec()                       # defaults match topdown_pipeline / eval_sim
scorecard = ScorecardWriter(f"{out_dir}/gate0_scorecard.csv")   # once, before the loop

# ... per finished episode i ...
res = evaluate_episode(
    plan_ok=plan_ok_i,
    block_final_pos=bf[i].tolist(),
    block_final_vel=bvel[i].tolist(),
    max_lift_m=float(maxlift[i]),
    pre_grasp_disp_m=float(pre_grasp_disp_i),
    spec=SPEC,
)
scorecard.log(episode_id, res)             # logs KEEP and DROP alike (audit + yield)

if res.keep or not save_only:              # KEEP-only unless collecting negatives
    for frame in episode_frames:
        ds.add_frame(frame)
    ds.save_episode()
    succ += int(res.success)

# ... after the loop ...
print(scorecard.close())                   # {"n_total":..., "n_keep":..., "yield":...}
ds.finalize()                              # MUST call, or parquet footers are lost
```

## What changed vs. the old gate

The old gate was `block_in_box(...) and maxlift > 0.03` — success + a crude grasp
proxy. Gate 0 keeps both and adds:
- **`approach_clean`** — drops episodes where the arm knocked the block before grasping
  (these look "successful" if the block happens to end in the box, but teach the policy
  to bump objects). This is the main new filter.
- **funnel `drop_reason`** — every dropped episode is labelled by its earliest failing
  stage (`plan_failed` / `knocked_block` / `grasp_failed` / `place_failed`), so the
  scorecard shows *where* the oracle is losing yield, not just how much.
- **per-episode scorecard + yield summary** — auditable, and lets you re-tune thresholds
  (`BoxTaskSpec`) from logged signals without re-running the sim.

Thresholds live in `BoxTaskSpec` (grasp_lift 3 cm, approach drift 1.5 cm, box geometry).
Keep `box_xy` in sync with `table_env_cfg.BOX_POS` if the scene moves.

# Gate 1 — trajectory quality (runs on Gate-0 KEEP episodes)

Gate 1 *scores* the survivors so the corpus looks like clean, deliberate demos. Call it
once per KEEP episode over the recorded per-frame joint trajectory (the same `(T, 6)`
you buffered). It's a scorer, not a hard filter — log the score/flags and let Gate 2
(batch) decide any drops.

```python
import numpy as np
from sim_to_real_so101.curation import (
    combined_fields, evaluate_episode, evaluate_trajectory, gate1_row, Gate1Spec,
)
import csv

writer = csv.DictWriter(open(f"{out_dir}/scorecard.csv", "w", newline=""),
                        fieldnames=["episode_id"] + combined_fields())
writer.writeheader()
G1 = Gate1Spec()

# ... per finished episode, after Gate 0 ...
g0 = evaluate_episode(plan_ok=..., block_final_pos=..., block_final_vel=...,
                      max_lift_m=..., pre_grasp_disp_m=...)

# joint trajectory the loop already buffered: list of 6-vectors -> (T, 6)
traj = np.asarray(episode_joint_frames, dtype=float)
# optional: block-centre distance to the box goal per frame (enables progress metric)
b2g = np.hypot(block_xy_traj[:, 0] - 0.10, block_xy_traj[:, 1] - 0.20)
g1 = evaluate_trajectory(traj, spec=G1, ref_len=batch_median_len, block_to_goal=b2g)

row = {"episode_id": episode_id}
row.update({k: v for k, v in _gate0_row(g0).items()})   # scorecard_fields()
row.update(gate1_row(g1))                                # gate1_fields()
writer.writerow(row)

keep = g0.keep                       # hard filter
# optional quality trimming: keep and g1.verdict == "PASS"
# optional idle trim before save: traj[g1.trim_start:g1.trim_end]
```

Notes:
- `ref_len` (duration ratio) is a *batch* quantity — pass `None` on the first pass, or
  the running median once you have one; the raw `n_frames` is logged regardless.
- Gate 1's `jerk_cap` is the one absolute threshold that needs calibration. Log the
  first batch, look at `peak_jerk`/`ldlj`/`quality_score` on the KEEP set, then set
  `jerk_cap` (and `pass_score`) from that distribution — don't guess it once and forget.
- `trim_start:trim_end` removes leading/trailing static frames before you record, so
  episodes start/end in motion (Gate 1's `idle` metric).
