# SO-101 Hybrid-Integration Pipeline — End to End

Scripted generation of **flawless pick-and-place demonstrations** in Isaac Sim to
replace human teleoperation when producing GR00T training data. A planner acts as a
privileged oracle: it plans a perfect trajectory, executes it in sim, and records it
as a LeRobot dataset — no human, no bias, and repeatable under domain randomization.

---

## 1. Trajectory generation: MoveIt 2 → custom FK-IK

**Start (MoveIt 2).** A `teleop-moveit` Docker image adds ROS 2 Jazzy + MoveIt 2 on
top of the Isaac teleop image (`docker/sim/Dockerfile.moveit`). Package
`so101_moveit_config` defines the arm as a chain `base_link → gripper_frame_link`,
driven via `moveit_py`. An 8-config FK calibration proved the Isaac ENV frame and
`base_link` are **translation-only, axis-aligned** — `BASE_ORIGIN_ENV =
(-0.0658, 0.0208, 0.0325)`, rigid-fit residual **0.00 mm**.

**The KDL wall.** The SO-101 is **5-DOF**. A fully-constrained 6-DOF orientation
goal is rank-deficient, and MoveIt's **KDL** solver cannot converge on it — a
horizontal side grasp returned **0 of 128** sampled orientations reachable. It
looked like a workspace limit; it was a solver limit.

**The fix (`classic_planner/scripts/so101_fk.py`).** Reframe IK as **3 position +
2 approach-direction** constraints (roll about the tool axis left free) — exactly 5
constraints for 5 DOF, well-posed. Implemented as:
- Analytic FK from the URDF, **validated 0.000 mm** vs MoveIt `RobotState` FK.
- Damped-least-squares `ik` (position + approach), position-only `ik_pos`,
  `grasp_ik` (places the *finger centre* on a target via a measured tool offset),
  and `straight_approach` for smooth Cartesian descents.
- Pure numpy, no ROS — so it runs inside Isaac's Python and **removes MoveIt from
  the runtime loop entirely**.

**The key discovery.** The real grasp was decoded from the user's own teleop demos
(`~/.cache/huggingface/lerobot/CursedRock17/so101_block_grab*`): approach elevation
**83–86° — straight down**, onto the *top* of the block. "The gripper was
horizontal" meant the **jaw line**, not the approach. A horizontal side grasp
topples the tall 22×22×50 mm block; **top-down never does**. This dissolved the
side-grasp problem. A separate Isaac probe measured the ~2 cm lateral offset between
the TCP frame and the finger gap (`GRASP_CENTER_TOOL`).

---

## 2. The pipeline (`classic_planner/scripts/topdown_pipeline.py`)

One Isaac script, no MoveIt:

1. **Reset** → triggers built-in domain randomization.
2. **Read** the randomized block pose.
3. **Plan** — precise straight-down `grasp_ik` grasp + **position-only high-carry
   arc** (up ~0.18 m, across, down into the box). Straight-down IK caps reach at
   ~0.10 m and the block hits the box walls; `ik_pos` reaches ~0.25 m by letting the
   tool tilt, matching the real demo's high lift.
4. **Execute** at 30 Hz; **record** `action` + `observation.state` (LeRobot
   normalized units) and the `top` + `wrist.top` cameras.
5. **Score** success (block in box); keep only flawless episodes.

**Domain randomization** (all on reset, plus in-script block-amount):
block placement (±3 cm + yaw), lighting exposure + sky light, robot color, mat
rotation, camera pose/FOV, and 1-vs-2-block variation.

Run:
```bash
isaaclab.sh -p topdown_pipeline.py --target 125 --max_attempts 260 \
  --out /root/datasets/so101_block_pickplace_planned \
  --repo_id CursedRock17/so101_block_pickplace_planned --video demo.mp4
```

---

## 3. Data pipeline (collection → conversion → hub)

- **Collection:** 125 flawless episodes / 239 attempts (~57% grasp success under
  full DR). Early-abort skips the transport for failed grasps.
- **Finalize (critical):** the dataset writer must call `ds.finalize()` or the
  parquet **footers are never written** and the data is unrecoverable. (Lost the
  first full run this way; the pipeline now finalizes.)
- **v3.0 → v2.1** (`helper_scripts/convert_v3_to_v2.py`) — GR00T fine-tuning uses
  v2.1. On this aarch64 box the system `ffmpeg` is the wrong arch; point it at the
  `imageio_ffmpeg` build: `ln -sf .../imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-* /usr/local/bin/ffmpeg`.
- **Push:** `HfApi().upload_folder(...)` →
  **https://huggingface.co/datasets/CursedRock17/so101_block_pickplace_planned**
  (public, v2.1, 125 episodes, `top` + `wrist.top` video).

---

## 4. GR00T-N1.6 fine-tuning

**Config** (ready in the dataset + repo):
- `meta/modality.json` maps `front→observation.images.top`,
  `wrist→observation.images.wrist.top`, state/action `single_arm[0:5]`+`gripper[5:6]`,
  annotation `human.task_description→task_index`.
- Modality config: `Isaac-GR00T/examples/SO100/so100_config.py`; embodiment tag
  `NEW_EMBODIMENT` (SO-100/101 are the same 6-DOF arm).

**Environment: the `real-robot:latest` Docker image** (the "real/sim" container),
NOT the Isaac python. It ships `/Isaac-GR00T` at `ead5283`/`n1.6-release` (the
model-card version), Py 3.10, torch 2.13+cu130 (works on the GB10), **flash_attn
2.8.3**, and **torchcodec 0.13 — the DEFAULT video backend, so `decord` is never
needed** (it's lazily imported only if explicitly requested). An earlier note here
wrongly concluded "blocked on aarch64 / x86-only"; the GB10 runs it fine.

Setup + command (validated; full 30k run trained at ~4 s/step ≈ 33 h):
```bash
docker run -d --name gr00t-ft --privileged --gpus all --network=host --shm-size=32g \
  -v $PWD/ft_data:/workspace/ft_data -v $PWD/ft_out:/workspace/ft_out \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash real-robot:latest -c "sleep infinity"
docker exec gr00t-ft bash -lc 'pip install timm omegaconf hydra-core; ln -sf /usr/bin/python3 /usr/local/bin/python'
docker exec -d gr00t-ft bash -lc 'cd /Isaac-GR00T && \
  NUM_GPUS=1 MAX_STEPS=30000 SAVE_STEPS=1000 GLOBAL_BATCH_SIZE=32 USE_WANDB=0 \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/ft_data/so101_block_pickplace_planned \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --output-dir /workspace/ft_out/so101_block_pickplace'
```

**Gotchas hit:** `--shm-size=32g` is REQUIRED (default 64 MB `/dev/shm` deadlocks the
DataLoader at batch 32 — hangs after step 2 with "No space left on device (28)");
`finetune.sh` calls `python` (symlink to `python3`); embodiment tag must be the enum
NAME `NEW_EMBODIMENT`; add `timm`/`omegaconf`/`hydra-core` (missing from the
inference image). Checkpoints save every 1000 steps to the host `ft_out/` (resumable).

---

## Status

| Stage | Status |
|---|---|
| FK-IK planner (bypasses KDL) | ✅ committed `ba684fb` (`naval_research`) |
| Top-down grasp verified in physics | ✅ |
| DR collection → 125 flawless episodes | ✅ |
| Dataset finalized + v3→v2.1 | ✅ |
| Push to hub (public) | ✅ |
| GR00T-N1.6 fine-tune | ✅ running on the GB10 via `real-robot:latest` (~4 s/step, ~33 h) |

## Gotchas / lessons
- Always `ds.finalize()` LeRobot datasets, or lose the run to footerless parquets.
- On aarch64, use the `imageio_ffmpeg` binary; the system `ffmpeg` is wrong-arch.
- GR00T fine-tunes fine on the GB10 via `real-robot:latest` (torchcodec is the
  default video backend, so `decord` is not needed; `flash_attn` is prebuilt).
  Requires `--shm-size=32g` or the DataLoader deadlocks at batch 32.
- Speed lever for future collections: run `num_envs > 1` (TiledCamera renders many
  envs in one pass) — the single-env render was the throughput bottleneck.
