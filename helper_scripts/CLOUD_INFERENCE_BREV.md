# Cloud inference on Brev (GPU server ↔ CPU laptop)

Run the GR00T N1.6 vials policy on a rented **L4 GPU** in the cloud, and drive the
real SO-101 from a **CPU-only laptop**. The laptop only sends camera frames + joint
state and receives actions; all the VLA compute happens in the cloud.

```
┌── laptop (CPU) ──┐   SSH tunnel   ┌── Brev L4 (GPU) ──┐
│ so101_eval.py     │  localhost:5555 │  run_gr00t_server │
│ arm + 2 cameras   │ ◀────────────▶  │  Gr00tPolicy      │
└───────────────────┘   ZMQ actions   └───────────────────┘
```

- **Model:** `CursedRock17/so101_teleop_vials_sim_and_real_finetune`
- **Language instruction:** `Pick up vial and place it in the target location`
- **Cameras the client sends:** `front`, `wrist` (already matches the model)

---

> **Relation to the GR00T tutorial**
> ([10-groot](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html#hands-on-run-gr00t-post-training-yourself)):
> do **step 1** (launch the launchable + `brev shell`), then **skip steps 2–8**
> (dataset conversion + fine-tuning) — we already have a trained model. This doc
> is the inference/eval flow, which the tutorial doesn't cover. The launchable
> below is NVIDIA's tutorial environment (GR00T container + deps), independent of
> this repo; you bring your code + model into it (steps 2–3).

## 1. Start the GPU instance (choose ONE)

**A. The GR00T launchable (fastest — it ships the GR00T container):**
Deploy the launchable (this provisions the GPU environment; it is *not* this repo):

    https://brev.nvidia.com/launchable/deploy?launchableID=env-3DfuAZZiLMXlclVQt1ddp0fEsOH

Set the GPU to **L4** and Deploy. Then get a shell:

```bash
brev login
brev ls                       # find the instance name once it's RUNNING
brev shell <instance>
```

**B. Manual custom container** (Brev console → **+ New → Container Mode →
Custom Container**): give it a GR00T image (the one you fine-tuned with, x86_64),
JupyterLab = No, GPU = **L4**, then Deploy. Confirm the price before deploying.

> The laptop is the thin client, so any x86 GR00T container with the `gr00t`
> package works on the server. The aarch64 image from `docker/real/` is for the
> GB10, not a Brev x86 GPU — don't use it here.

## 2. Put the server script on the instance

Run this from a **separate local terminal at your repo root** (not inside the
`brev shell` from step 1 — `brev copy` is a local command). The launchable's
SSH user is `ubuntu` and cannot write `/workspace`; copy into the cloned repo,
which is writable:

```bash
brev copy ./helper_scripts/run_gr00t_server.py <instance>:~/Isaac-GR00T/
```

## 3. Download the final model + start the server (in the instance shell)

The launchable env is **uv-managed** (no `source .venv` — `uv run` resolves deps
on demand). Two hard-won rules: (a) the HF repo carries **every** training
checkpoint (~215 GB) — fetch only the final model once, then serve **offline** so
it never re-pulls them; (b) on an **L4** you must drop the `LD_LIBRARY_PATH` compat
override the startup script sets for an H100 (see troubleshooting: Error 803).

```bash
cd ~/Isaac-GR00T
export PATH=$HOME/.local/bin:$PATH

# --- one-time: download ONLY the final model (skip every checkpoint-*) → ~10 GB ---
#   (private repo? `huggingface-cli login` or `export HF_TOKEN=hf_xxx` first)
uv run huggingface-cli download CursedRock17/so101_teleop_vials_sim_and_real_finetune \
    --exclude "checkpoint-*"

# --- start the server ---
export HF_HUB_OFFLINE=1        # serve local model only; never re-download checkpoints
unset LD_LIBRARY_PATH          # L4 CUDA-803 fix (drops the H100/driver-550 compat path)
nohup uv run python run_gr00t_server.py \
    --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \
    --port 5555 > ~/gr00t_server.log 2>&1 &

sleep 30; tail -n 5 ~/gr00t_server.log     # wait for: serving on tcp://*:5555
```

Do **not** use `run_gr00t_server.py --auto-checkpoint` here — it triggers a full
`snapshot_download` that re-pulls all 215 GB and fills the disk. The `--exclude`
download + `HF_HUB_OFFLINE=1` is what keeps the cache at ~10 GB.

## 4. Tunnel the port to the laptop (new laptop terminal)

```bash
brev port-forward <instance> -p 5555:5555
```

Leave this running. It maps the cloud server to `localhost:5555` over SSH — nothing
is exposed on the public internet. Equivalent raw-SSH form (after `brev refresh`
writes the SSH config): `ssh <instance> -L 5555:localhost:5555`.

## 5. Run the robot client (on the machine wired to the arm + cameras)

On the GB10, `helper_scripts/real_basic.sh` already launches the `real-robot`
container with everything the client needs — `--network=host` (so `localhost:5555`
is the tunnel), `/dev` passthrough (arm + cameras), `docker/env` (robot vars), and
`docker/real/scripts` mounted over the in-image eval (so the `--record_video` edits
are live). Launch it, then run the eval **inside** that shell:

```bash
./helper_scripts/real_basic.sh          # drops you into the container shell

# inside the container:
source /root/env                        # ROBOT_PORT, ROBOT_ID(=muninn), CAMERA_GRIPPER/EXTERNAL
cd /Isaac-GR00T
python3 gr00t/eval/real_robot/SO100/so101_eval.py \
    --robot.type=so101_follower \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: $CAMERA_GRIPPER, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: $CAMERA_EXTERNAL, width: 640, height: 480, fps: 30} }" \
    --policy_host=localhost --policy_port=5555 \
    --lang_instruction="Pick up vial and place it in the target location" \
    --record_video=true --video_path=/workspace/models/rollout.mp4 --max_steps=300
```

Gotchas that cost time: this image has **no `uv`** — use `python3` directly; the
eval uses **draccus**, so booleans need a value (`--record_video=true`, not bare);
`--max_steps` bounds the motion and the mp4 length; the video lands on the host at
`models/rollout.mp4` (the `models` mount). Why `--network=host`: in a bridged
container `localhost` is the container's own loopback, not the host where the
tunnel lands, so the client would never reach the server.

> **Note:** the `real-robot` image is aarch64 (built for the GB10). A CPU **laptop**
> client would need an x86 build, or just the two client deps bare — `lerobot` +
> `gr00t`'s `PolicyClient` (pure zmq/msgpack/numpy, no CUDA).

## 6. Stop (avoid idle charges)

```bash
# Ctrl-C the server and the port-forward, then:
brev stop <instance>      # keep it to resume later
# brev delete <instance>  # tear it down completely
```

---

### Sanity check the link before touching the robot

From the laptop, with the tunnel up, a quick ZMQ ping confirms the round-trip:

```python
from gr00t.policy.server_client import PolicyClient
print(PolicyClient(host="127.0.0.1", port=5555).ping())  # -> ok
```

### Troubleshooting

- **`Connection refused` on the client:** the server isn't up yet, or the
  port-forward terminal died. Confirm step 3 printed `serving on tcp://*:5555`
  and step 4 is still running.
- **`CUDA error 803` / `torch.cuda.is_available()` is False (but `nvidia-smi` works):**
  the startup script pins `LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat` for an
  **H100 on driver 550**; an **L4 host runs the newer 580 driver**, so that compat
  path forces an older `libcuda.570` over the native `libcuda.580` → kernel/user
  driver mismatch (Error 803). `nvidia-smi` (NVML) is unaffected, which is why it
  still looks fine. Fix: `unset LD_LIBRARY_PATH` before launching, and remove the
  persisted line so it doesn't return every shell / after a stop-start:
  ```bash
  sed -i '/cuda-12.8\/compat/d' ~/.bashrc
  ```
- **`Unrecognized model` / missing config.json:** point `--model-path` at the folder
  that holds `config.json`. For this repo that's the snapshot **root** (the final
  model), which is what the `--exclude "checkpoint-*"` download in step 3 gives you.
- **Wrong / jerky actions:** verify `--lang_instruction` matches training exactly
  and the client is sending `front` + `wrist` in the order the model expects.
- **`No space left on device` / server core-dumps on launch:** the HF repo ships
  *every* training checkpoint, so a plain `snapshot_download` (or `--auto-checkpoint`)
  pulls ~200 GB+ and fills the instance disk — then `uv` can't install deps and the
  server aborts. Fix: point `--model-path` at the **final model** (the snapshot root
  has `config.json` + `model-*.safetensors`; that's all inference needs) and, if the
  disk is already full, delete the checkpoint-only blobs. Because `/tmp` is on the
  full root disk, write the delete lists to **`/dev/shm`** (RAM), not `/tmp`:

  (On the launchable the cache actually lives at `/ephemeral/cache/huggingface`,
  symlinked from `~/.cache/huggingface` — either path works below.)

  ```bash
  M=~/.cache/huggingface/hub/models--<org>--<repo>
  SNAP=$(ls -d $M/snapshots/*/ | head -1)
  find $SNAP -type l -not -path '*/checkpoint-*' -exec readlink -f {} \; | sort -u > /dev/shm/keep
  find $SNAP -type l     -path '*/checkpoint-*' -exec readlink -f {} \; | sort -u > /dev/shm/ckpt
  comm -23 /dev/shm/ckpt /dev/shm/keep | xargs -r rm -f    # deletes checkpoint-only blobs
  rm -rf $SNAP/checkpoint-*                                 # then the (now-dangling) symlink dirs
  ```

  This keeps the final model intact (its blobs are in `keep`) and is idempotent, so
  it's safe to re-run if interrupted (e.g. by a `brev stop`).
