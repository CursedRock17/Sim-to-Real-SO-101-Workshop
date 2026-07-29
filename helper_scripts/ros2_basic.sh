#!/bin/bash
# Launch the standalone ROS 2 Jazzy + MoveIt 2 + pick_ik container (so101-ros:jazzy).
# Runs ALONGSIDE the Isaac Sim container (sim_basic.sh) and talks to it over ROS 2 DDS
# on the shared host network. ROS is auto-sourced by the image entrypoint, so `ros2`,
# `moveit_py`, and `pick_ik` work immediately -- no `source-ros`, no Python collision.
#
# Build first (once):  docker build -t so101-ros:jazzy docker/ros/
xhost + >/dev/null 2>&1
mkdir -p "$(pwd)/ros2_ws/src"
docker run -it --rm --name so101-ros --network=host --privileged --gpus all \
    -e DISPLAY=$DISPLAY \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v "$(pwd)/ros2_ws":/workspace/ros2_ws \
    so101-ros:jazzy
