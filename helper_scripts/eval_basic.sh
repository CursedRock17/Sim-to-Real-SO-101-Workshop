#!/usr/bin/env bash
# Bring up the GR00T policy server on the fine-tuned checkpoint and print the two
# eval commands (open-loop action MSE + closed-loop sim task-success).
#
# Reuses a running gr00t-ft/gr00t-server if present; otherwise launches a fresh
# server container from the real-robot image with ft_out/ft_data/HF-cache mounted
# (unlike real_basic.sh, which is set up for the physical robot).
#
#   bash helper_scripts/eval_basic.sh [checkpoint-name]     # default: checkpoint-30000
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CKPT_NAME="${1:-checkpoint-30000}"
CKPT="/workspace/ft_out/so101_block_pickplace/${CKPT_NAME}"
PORT=5555
mkdir -p "$REPO/ft_out" "$REPO/ft_data"

# 1) find a running server container, else launch one from the real-robot image
SERVER=""
for c in gr00t-ft gr00t-server; do
    docker ps --format '{{.Names}}' | grep -qx "$c" && { SERVER="$c"; break; }
done
if [ -z "$SERVER" ]; then
    SERVER=gr00t-server
    echo "launching $SERVER (real-robot image, ft_out/ft_data/HF mounted)..."
    docker run -d --name "$SERVER" --rm --privileged --gpus all --network=host --shm-size=16g \
        -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
        -v "$REPO/ft_out":/workspace/ft_out \
        -v "$REPO/ft_data":/workspace/ft_data \
        -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
        --entrypoint /bin/bash real-robot:latest -c "sleep infinity" >/dev/null
    docker exec "$SERVER" bash -lc 'ln -sf /usr/bin/python3 /usr/local/bin/python;
        pip install -q timm omegaconf hydra-core 2>/dev/null || true'
fi
echo "server container: $SERVER"

# 2) start the policy server if it is not already serving
if ! docker exec "$SERVER" bash -lc 'pgrep -f run_gr00t_server >/dev/null'; then
    echo "starting policy server on :$PORT for $CKPT ..."
    docker exec -d "$SERVER" bash -lc "cd /Isaac-GR00T && \
        python gr00t/eval/run_gr00t_server.py --model-path '$CKPT' \
            --embodiment-tag NEW_EMBODIMENT --port $PORT > /tmp/server.log 2>&1"
fi

# 3) wait for the 3B model to load
echo -n "waiting for server to be ready (model load ~1-2 min)"
for _ in $(seq 1 80); do
    if docker exec "$SERVER" bash -lc 'grep -qi "ready\|listening" /tmp/server.log 2>/dev/null'; then
        echo " -> READY"; break
    fi
    if docker exec "$SERVER" bash -lc 'grep -qiE "Error|Traceback" /tmp/server.log 2>/dev/null'; then
        echo " -> SERVER ERROR:"; docker exec "$SERVER" bash -lc 'tail -8 /tmp/server.log'; exit 1
    fi
    echo -n "."; sleep 3
done

# 4) which Isaac container is up (teleop-moveit image) for the sim client
ISAAC=$(docker ps --format '{{.Names}}\t{{.Image}}' | awk -F'\t' '$2 ~ /teleop-moveit/ {print $1; exit}')
ISAAC="${ISAAC:-<start with: bash helper_scripts/moveit_basic.sh>}"

cat <<EOF

============================================================================
GR00T policy server is up:  container=$SERVER  port=$PORT  checkpoint=$CKPT
============================================================================

# --- (A) Open-loop eval: predicted-vs-GT action MSE + plots (server container) ---
docker cp classic_planner/scripts/eval_finetune.sh $SERVER:/root/eval_finetune.sh
docker exec $SERVER bash /root/eval_finetune.sh $CKPT

# --- (B) Closed-loop SIM eval: task success in the table env (Isaac container) ---
docker exec $ISAAC bash -lc '
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install -q pyzmq msgpack
  cd /workspace/Sim-to-Real-SO-101-Workshop/classic_planner/scripts
  /workspace/isaaclab/isaaclab.sh -p eval_sim.py --episodes 20 --host localhost --port $PORT'

# stop the server when done:
docker exec $SERVER bash -lc 'pkill -f run_gr00t_server'    # or: docker rm -f $SERVER
EOF
