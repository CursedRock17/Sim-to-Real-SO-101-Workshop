# Cloud inference on Brev: CPU laptop → L4 → SO-101

The laptop reads the arm and two cameras, sends observations through an SSH
tunnel, and executes returned actions; the L4 runs GR00T N1.6 inference.
The laptop does not need CUDA, model weights, or Isaac Sim.

This recipe targets **Linux on an Intel/AMD laptop with Docker**; Windows/WSL
requires separate USB/camera forwarding, and macOS Docker does not provide
the Linux device access used below.

| Setting | Value |
| --- | --- |
| Brev instance | `$BREV_INSTANCE` (set in step 1) |
| Policy port (server and laptop) | `5556` |
| HF repository | `CursedRock17/so101_teleop_vials_sim_and_real_finetune` |
| Selected model | **`checkpoint-30000`**, the previously tested checkpoint |
| HF revision | `ed795230464d2785715a5fbb853676522e712e66` |
| GR00T code revision | `ead52833afbbf4243f8cd5e7664f48a94de03b19` |
| Language instruction | `Pick up vial and place it in the target location` |
| Physical camera → model input | `front` → `external_D455`; `wrist` → `ego` |

The camera mapping above comes from checkpoint 30000's `processor_config.json`;
the old guide incorrectly claimed that the model expected `front` and `wrist`.
Preserve the physical views and robot calibration used for your successful trial.

## 1. Connect to the existing instance

On the laptop, install/login to Brev, then:

```bash
brev ls
export BREV_INSTANCE="isaac-gr00t-n1-6-post-training-e82e7a"
brev shell "$BREV_INSTANCE"
```

Set `BREV_INSTANCE` to the current instance name shown by `brev ls`.
Repeat the `export BREV_INSTANCE=...` assignment in each new laptop terminal
that runs Brev or SSH commands; separate terminals do not share exported variables.
This variable is used locally and does not need to be set inside the Brev shell.

The instance already exists; no new GPU is needed.
From a separate **local terminal at this repo's root**, copy the server:

```bash
brev copy helper_scripts/run_gr00t_server.py "${BREV_INSTANCE}:~/Isaac-GR00T/"
```

## 2. Fetch only checkpoint 30000's inference files

In the **Brev shell**:

```bash
cd ~/Isaac-GR00T
unset HF_HUB_OFFLINE
.venv/bin/python run_gr00t_server.py \
    --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \
    --checkpoint checkpoint-30000 \
    --revision ed795230464d2785715a5fbb853676522e712e66 \
    --download-only
```

The updated server uses Hugging Face's [filtered downloads](https://huggingface.co/docs/huggingface_hub/guides/download#filter-files-to-download)
to select `checkpoint-30000/*` and exclude training state (`*.pt`, `*.pth`,
`*.bin`, trainer state and W&B config).
The two safetensors shards total **9.81 GB (9.14 GiB)**, plus small config and
statistics files; this skips the **12.96 GB optimizer** and every other checkpoint.
Already cached blobs are reused.

`--download-only` prints the local checkpoint directory, which contains
`config.json`, `processor_config.json`, `statistics.json`, the shard index and
both weight shards.
Do not substitute the repository root or `--auto-checkpoint`: those select a
different model when a final model exists at the root.
Older copies of the server on Brev also downloaded the whole repository with
`--auto-checkpoint`, so copy the updated script first.

## 3. Serve that checkpoint on the L4

In the **Brev shell**, first check whether a server is already listening:

```bash
ss -ltnp | grep ':5556'
```

If it is already serving the intended checkpoint, reuse it; otherwise start:

```bash
cd ~/Isaac-GR00T
unset LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
nohup .venv/bin/python -u run_gr00t_server.py \
    --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \
    --checkpoint checkpoint-30000 \
    --revision ed795230464d2785715a5fbb853676522e712e66 \
    --offline --host 127.0.0.1 --port 5556 \
    > ~/gr00t_checkpoint_30000.log 2>&1 < /dev/null &
tail -f ~/gr00t_checkpoint_30000.log
```

Wait for `Server is ready and listening on tcp://127.0.0.1:5556`;
Ctrl-C exits `tail`, while the background server keeps running.
The existing launchable's `.venv` contains the inference dependencies, so these
commands use it directly without invoking dependency resolution through `uv run`.
Unsetting `LD_LIBRARY_PATH` avoids a stale CUDA compatibility override on the L4.

The server binds only to loopback and the laptop reaches it through SSH.
On a fresh launchable, if offline startup reports missing Eagle tokenizer or
processor assets, run the same command once without `HF_HUB_OFFLINE=1` and
`--offline`, let startup complete, then restart offline;
the explicit checkpoint filter still prevents downloading other training checkpoints.

## 4. Build the CPU client on the laptop

From the repo root on the **laptop**:

```bash
docker build -t so101-cloud-client:cpu -f docker/real/Dockerfile.cpu .
```

This image installs CPU PyTorch for LeRobot, the same pinned LeRobot driver
revision as the GB10 image, and GR00T's pinned client code without its GPU
dependencies; the image build checks that PyTorch has no CUDA support and that
the evaluation client imports successfully.
It needs neither `--gpus` nor NVIDIA Container Toolkit.
The pinned GR00T package also imports Transformers types when loading
`PolicyClient`, so the image includes that library without constructing a model.

Copy your existing calibration for this physical follower arm to the laptop's
`~/.cache/huggingface/lerobot/calibration/` and use the same `ROBOT_ID`;
the robot calibration file is normally under `robots/so101_follower/`.
Use your own arm's calibration, not a sample calibration from this repository.
Stop the previous controller before the laptop takes ownership of the arm.

## 5. Forward the port and check the policy without hardware

In a dedicated **laptop terminal**, set `BREV_INSTANCE` as in step 1, then:

```bash
brev port-forward "$BREV_INSTANCE" -p 5556:5556
```

Leave this running; equivalent SSH forwarding after `brev refresh` is:

```bash
ssh -N -L 127.0.0.1:5556:127.0.0.1:5556 "$BREV_INSTANCE"
```

In another **laptop terminal**:

```bash
docker run --rm --network=host so101-cloud-client:cpu \
    --policy_host=127.0.0.1 --policy_port=5556 \
    --model_front_key=external_D455 --model_wrist_key=ego \
    --timeout=15 --check_policy=true
```

This pings the server and checks its saved camera keys without opening the robot
or cameras; it is a connectivity/configuration check, not a model forward pass.
Host networking makes the laptop's forwarded port accessible inside the container.
The client also performs this check before each normal rollout.

## 6. Run a bounded rollout on the laptop

Connect the follower and both cameras, then identify the cameras by their stable
udev names on the **laptop**; each physical camera may expose multiple entries,
so use its `video-index0` capture device:

```bash
ls -l /dev/v4l/by-id/
```

Set the paths below to the two intended cameras, resolving the stable links to
their current `/dev/video*` nodes before Docker starts:

```bash
export ROBOT_PORT=/dev/ttyACM0
export ROBOT_ID=muninn
export CAMERA_EXTERNAL_HOST="$(readlink -f /dev/v4l/by-id/REPLACE_WITH_EXTERNAL-video-index0)"
export CAMERA_GRIPPER_HOST="$(readlink -f /dev/v4l/by-id/REPLACE_WITH_GRIPPER-video-index0)"
test -c "$CAMERA_EXTERNAL_HOST" && test -c "$CAMERA_GRIPPER_HOST"
mkdir -p outputs

docker run --rm -it --name so101-client \
    --network=host \
    --device="$ROBOT_PORT:/dev/ttyACM0" \
    --device="$CAMERA_EXTERNAL_HOST:/dev/video-front" \
    --device="$CAMERA_GRIPPER_HOST:/dev/video-wrist" \
    -v "$HOME/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration" \
    -v "$PWD/outputs:/workspace/outputs" \
    -e ROBOT_ID="$ROBOT_ID" \
    --entrypoint /bin/bash \
    so101-cloud-client:cpu
```

This opens a shell instead of the image's default evaluation entrypoint.
The fixed container names `/dev/video-front` and `/dev/video-wrist` prevent the
laptop's built-in webcam from changing which cameras the policy receives.
Inside the container, confirm the three mapped devices and then start evaluation:

```bash
ls -l /dev/ttyACM0 /dev/video-front /dev/video-wrist

python /workspace/so101_eval.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 --robot.id="$ROBOT_ID" \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: '/dev/video-front', width: 640, height: 480, fps: 30} }" \
    --policy_host=127.0.0.1 --policy_port=5556 \
    --model_front_key=external_D455 --model_wrist_key=ego \
    --lang_instruction="Pick up vial and place it in the target location" \
    --action_horizon=16 --timeout=60 --max_steps=300 \
    --record_video=true --video_path=/workspace/outputs/rollout.mp4
```

To enter the same named container from a second laptop terminal while it is
running, use `docker exec -it so101-client /bin/bash`.

This command **moves the arm**: the existing controller moves to its configured
initial pose at connection and its home pose on normal shutdown.
`--max_steps` bounds policy actions, not those setup/shutdown moves.
The recorded video appears at `outputs/rollout.mp4` on the laptop.
Booleans require values (`--record_video=true`), and each continuation `\` must
be the final character on its line.

The control loop executes up to 16 actions at 30 Hz, then waits for the next
inference response; network and inference latency add pauses between chunks,
so this is not a guaranteed continuous 30 Hz cloud control loop.
It sends raw camera arrays, so upload bandwidth also affects responsiveness.

For an initial test from the **existing GB10**, use `helper_scripts/real_basic.sh`
and its existing `python3 gr00t/eval/real_robot/SO100/so101_eval.py` command,
adding the same policy host/port and the two `--model_*_key` options above;
the host must run its own SSH tunnel and the container must use host networking.
The new CPU image is the distributable laptop client; the existing Blackwell
image remains specific to the GB10 setup.

## Verification performed on 2026-09-17

- Loaded checkpoint 30000 offline on the existing Brev L4 and received valid
  action arrays of shape `(1, 16, 5)` for the arm and `(1, 16, 1)` for the gripper.
- Built `so101-cloud-client:cpu` on the instance's x86 CPU; verified
  `torch==2.7.1+cpu` and `torch.version.cuda is None` without exposing a GPU
  to the client container.
- Ran the packaged client's preflight and adapter against the L4 with synthetic
  camera frames and joint states; received 16 finite decoded robot actions.
- Confirmed that unavailable servers time out and incorrect camera mappings fail
  before the client opens hardware.
- Ran four model-selection regression tests covering filtered downloads,
  missing checkpoints, offline selection and invalid arguments.
- Sent a synthetic observation from the development machine through an SSH
  tunnel to the L4; the measured round trip was approximately 1.23 seconds.

These checks did not connect to or move a physical robot, and the CPU container
was tested on the cloud host rather than the destination laptop.
Latency measurements use synthetic images and are not a guarantee of real-camera
performance or successful manipulation over the laptop's network connection.

## Troubleshooting and stopping

- **Timeout/refused connection:** verify the server log, loopback listener and
  laptop tunnel; `--timeout` now sets actual socket send/receive deadlines.
- **Camera key mismatch:** checkpoint 30000 expects `external_D455` and `ego`;
  preserve the explicit mapping while keeping physical cameras named `front`/`wrist`.
- **CUDA error 803:** unset `LD_LIBRARY_PATH` in the server shell and retry;
  inspect startup scripts for an obsolete `/usr/local/cuda-12.8/compat` override.
- **Missing cached files:** repeat the filtered download online, then serve offline.
- **Disk full from an earlier whole-repo download:** inspect the HF cache before
  deleting anything; this recipe does not delete or alter existing checkpoints.
- **USB errors:** verify laptop device paths, calibration ID and that no other
  controller has the same serial port open.

After a rollout, Ctrl-C the laptop port forward when finished.
To stop the background policy, identify its PID with
`pgrep -af run_gr00t_server.py` and terminate that specific process;
Ctrl-C in the log viewer does not stop a `nohup` server.
When finished with the GPU session, you can stop the instance with
`brev stop "$BREV_INSTANCE"`; retained storage may still incur charges.
