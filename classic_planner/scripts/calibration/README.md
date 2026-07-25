# Isaac ↔ URDF frame calibration

Validates the `env ↔ base_link` transform used by the planner (`so101_planning.py`)
and confirms the USD/URDF joint conventions match. Re-run this if the arm asset,
URDF, or its placement in the scene ever changes.

## Procedure

Three steps; JSONs are exchanged via a shared directory (the container scratch
mount, here `/root/smoke`).

1. **Isaac side** — drive the arm to known configs, dump actual joint values +
   gripper/base world poses (ENV frame). Run in `teleop-docker`:
   ```
   isaaclab.sh -p calibration/isaac_fk_dump.py     # writes isaac_fk.json
   ```
2. **MoveIt side** — URDF forward kinematics for the same configs (base_link
   frame). Run in `teleop-moveit` (after `source-ros` + `colcon build` + source
   install):
   ```
   python3 calibration/urdf_fk_dump.py             # reads isaac_fk.json, writes urdf_fk.json
   ```
3. **Compare** — solve the rigid transform mapping URDF-base FK → Isaac ENV:
   ```
   python3 calibration/compare_frames.py <dir-with-both-json>
   ```

## Result (2026-07-24)

- `gripper_link` fit residual **0.00 mm** across 8 configs → **joint conventions
  match exactly** (no value remapping needed); Isaac `gripper` body == URDF
  `gripper_link`.
- Transform is **translation-only, identity rotation** (NOT the 90° yaw implied
  by `so101.py`): `base_link` origin sits at ENV `(-0.0658, 0.0208, 0.0325)`.

These values are baked into `scripts/so101_planning.py` as `BASE_ORIGIN_ENV`.
