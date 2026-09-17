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

## 3. Start the inference server (in the instance shell)

The launchable env is **uv-managed** (there's no `source .venv` — `uv run`
resolves deps into `.venv` on demand), so run the server through `uv` from the
repo root:

```bash
cd ~/Isaac-GR00T
export PATH=$HOME/.local/bin:$PATH        # ensure `uv` is on PATH

# If the model is private, authenticate first:
#   huggingface-cli login        (or:  export HF_TOKEN=hf_xxx)

uv run python run_gr00t_server.py \
    --model-path CursedRock17/so101_teleop_vials_sim_and_real_finetune \
    --port 5555 \
    --auto-checkpoint          # downloads the repo, picks checkpoint-* if nested
```

Wait for `serving on tcp://*:5555`. First run downloads the weights (~a few GB).

## 4. Tunnel the port to the laptop (new laptop terminal)

```bash
brev port-forward <instance> -p 5555:5555
```

Leave this running. It maps the cloud server to `localhost:5555` over SSH — nothing
is exposed on the public internet. Equivalent raw-SSH form (after `brev refresh`
writes the SSH config): `ssh <instance> -L 5555:localhost:5555`.

## 5. Run the robot client (laptop, with the arm + cameras connected)

Run the client inside the **real-robot container, without `--gpus all`** — the
client only needs USB/camera/LeRobot access, not the GPU:

```bash
docker run --rm -it \
    --network=host \                     # REQUIRED: so localhost:5555 is the tunnel
    --device=/dev/ttyACM0 \              # the SO-101 arm (adjust to your port)
    --device=/dev/video0 --device=/dev/video1 \   # the 2 cameras
    <real-robot-image> \
    python3 docker/real/scripts/so101_eval.py \
        --policy_host=localhost \
        --policy_port=5555 \
        --lang_instruction="Pick up vial and place it in the target location"
```

Why `--network=host`: inside a bridged container, `localhost` is the container's
own loopback, **not** the host where the tunnel lands — so the client would never
reach the server. Host networking makes `localhost:5555` the forwarded port.

The client only needs `lerobot` (arm + camera drivers) and `gr00t`'s `PolicyClient`
(pure zmq/msgpack/numpy — no CUDA, no flash-attn), so no GPU is required.

> **Arch caveat:** the `docker/real/` image is built for the aarch64 GB10. It runs
> as-is only on an **ARM** laptop; on an x86 laptop you need an x86 build (or run
> the two client deps — `lerobot` + `PolicyClient` — bare, no container).

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
- **`Unrecognized model` / missing config.json:** the repo nests weights under
  `checkpoint-XXXXX`. Keep `--auto-checkpoint`, or point `--model-path` straight
  at the checkpoint folder.
- **Wrong / jerky actions:** verify `--lang_instruction` matches training exactly
  and the client is sending `front` + `wrist` in the order the model expects.
- **`No space left on device` / server core-dumps on launch:** the HF repo ships
  *every* training checkpoint, so a plain `snapshot_download` (or `--auto-checkpoint`)
  pulls ~200 GB+ and fills the instance disk — then `uv` can't install deps and the
  server aborts. Fix: point `--model-path` at the **final model** (the snapshot root
  has `config.json` + `model-*.safetensors`; that's all inference needs) and, if the
  disk is already full, delete the checkpoint-only blobs. Because `/tmp` is on the
  full root disk, write the delete lists to **`/dev/shm`** (RAM), not `/tmp`:

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
