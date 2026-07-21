# Simulation Scripts
## Evaluate in Sim
bash /workspace/isaaclab/isaaclab.sh -p /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/scripts/lerobot_eval \
    --task Lerobot-So101-Teleop-Vials-To-Rack-Eval \
    --rename_map '{"external_D455": "front", "ego": "wrist"}' \
    --action_horizon 16 \
    --lang_instruction "Pick up the vial and place it in the yellow rack" \
    --rerun

## Run Eval
bash /workspace/isaaclab/isaaclab.sh -p /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/scripts/lerobot_eval.py \
	--task Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval \
	--rename_map '{"external_D455": "front", "ego": "wrist"}'     \
	--action_horizon 16     \
	--lang_instruction "Pick up the vial and place it in the yellow rack" \
	--policy_port 5556     
	--rerun

# Real Life Scripts
## Run Server, Terminal 1
python3 Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
	--port 5556 \
	--model-path /workspace/models/$MODEL

## Real Time Eval, Terminal 2
python3 Isaac-GR00T/gr00t/eval/real_robot/SO100/so101_eval.py   --robot.type=so101_follower   --robot.port="$ROBOT_PORT"   --robot.id="$ROBOT_ID"   --robot.cameras="{
      wrist:  {type: opencv, index_or_path: $CAMERA_GRIPPER, width: 640, height: 480, fps: 30, fourcc: MJPG},
      front:  {type: opencv, index_or_path: $CAMERA_EXTERNAL, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }"   --policy_host=localhost   --policy_port=5556   --lang_instruction="Pick up the vial and place it in the yellow rack"   --rerun True



## Teleop Real
lerobot-teleoperate   --robot.type=so101_follower   --robot.port=$ROBOT_PORT   --robot.id=$ROBOT_ID   --teleop.type=so101_leader   --teleop.port=$TELEOP_PORT   --teleop.id=$TELEOP_ID   --display_data=true   --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": '"$CAMERA_GRIPPER"',
      "width": 640,
      "height": 480,
      "fps": 30, "fourcc": "MJPG"
    },
    "front": {
      "type": "opencv",
      "index_or_path": '"$CAMERA_EXTERNAL"',
      "width": 640,
      "height": 480,
      "fps": 30, "fourcc": "MJPG"
    }
  }'


# Operating Real with Attention
## Server, Terminal 1:
python3 Isaac-GR00T/gr00t/eval/run_gr00t_server_attn.py --model-path /workspace/models/$MODEL --port 5556 --camera-keys external_D455 ego --patch-grid-hw 9 9 --colormap hot --gamma 2.5 --embodiment-tag NEW_EMBODIMENT

## Client, Terminal 2:
python3 Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so101_attn.py   --robot.type=so101_follower   --robot.port="$ROBOT_PORT"   --robot.id="$ROBOT_ID"   --robot.cameras="{
      ego:  {type: opencv, index_or_path: $CAMERA_GRIPPER, width: 640, height: 480, fps: 30, fourcc: MJPG},
      external_D455:  {type: opencv, index_or_path: $CAMERA_EXTERNAL, width: 640, height: 480, fps: 30, fourcc: MJPG}
  }"   --policy_host=localhost   --policy_port=5556   --lang_instruction="Pick up the vial and place it in the yellow rack" --play_sounds false --rerun false --plot true --timeout=60
