#!/usr/bin/env bash
# Open-loop evaluation of the fine-tuned GR00T-N1.6 checkpoint on the planned dataset.
# Loads the checkpoint, predicts action chunks on held-out trajectories, and plots
# predicted vs ground-truth actions (per-joint, with MSE) -> the standard sanity check.
#
# RUN INSIDE the gr00t-ft container, and only AFTER training frees the GPU (it loads
# the 3B model on CUDA and would contend with the running finetune):
#   docker cp classic_planner/scripts/eval_finetune.sh gr00t-ft:/root/eval_finetune.sh
#   docker exec gr00t-ft bash /root/eval_finetune.sh              # latest checkpoint
#   docker exec gr00t-ft bash /root/eval_finetune.sh /workspace/ft_out/so101_block_pickplace/checkpoint-30000
#
# Plots land in /workspace/ft_out/eval_plots (host: ./ft_out/eval_plots). torchcodec
# is the default video backend (no decord needed).
set -euo pipefail

OUT=/workspace/ft_out/so101_block_pickplace
DATASET=/workspace/ft_data/so101_block_pickplace_planned
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "No checkpoint found (looked in $OUT). Pass one explicitly." >&2
    exit 1
fi
echo "Evaluating checkpoint: $CKPT"

ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null || true
mkdir -p /workspace/ft_out/eval_plots
cd /Isaac-GR00T

python gr00t/eval/open_loop_eval.py \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$CKPT" \
    --traj-ids 0 1 2 3 4 5 6 7 \
    --action-horizon 16 \
    --modality-keys single_arm gripper \
    --save-plot-path /workspace/ft_out/eval_plots

echo "Done. Predicted-vs-GT action plots in ./ft_out/eval_plots (and MSE printed above)."

# --- Closed-loop (in-sim) eval, optional / future -------------------------------
# For a task-success eval, run the policy in the table env instead of open-loop:
#   1) serve the checkpoint:  python gr00t/eval/run_gr00t_server.py \
#          --embodiment-tag NEW_EMBODIMENT --model-path "$CKPT"
#   2) drive Lerobot-So101-Teleop-Table-Task with the gr00t client
#      (helper_scripts/eval_so101.py pattern) and score with block_in_box() from
#      topdown_pipeline.py. Needs the GPU for BOTH Isaac and the 3B model.
