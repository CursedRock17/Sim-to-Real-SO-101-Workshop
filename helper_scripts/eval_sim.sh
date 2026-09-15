#!/bin/bash
# Launch a headless closed-loop sim eval of a GR00T policy in the SO-101 table task.
#
# Prereq: a GR00T policy server is already serving the checkpoint you want, e.g.
#   docker exec -d real-robot bash -lc 'cd /Isaac-GR00T && python3 \
#     gr00t/eval/run_gr00t_server.py \
#     --model-path /workspace/models/CursedRock17/gr00t-n1.6-so101-box-merged/checkpoint-45000 \
#     --embodiment-tag NEW_EMBODIMENT --port 5557'
#
# Usage: helper_scripts/eval_sim.sh [eval_sim.py args...]
#   e.g. helper_scripts/eval_sim.sh --checkpoint-name checkpoint-45000 --episodes 20 --port 5557
set -e

REPO=/workspace/Sim-to-Real-SO-101-Workshop
SCRIPT=$REPO/source/sim_to_real_so101/scripts/eval_sim.py

docker run --rm --name teleop-eval \
  --network=host --gpus all --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ./docker/env:/root/env \
  -v "$(pwd)/source:$REPO/source" \
  -v "$(pwd)/outputs:$REPO/outputs" \
  -w "$REPO" \
  teleop-docker:latest \
  /workspace/isaaclab/_isaac_sim/python.sh "$SCRIPT" "$@"
