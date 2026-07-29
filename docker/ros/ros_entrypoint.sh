#!/bin/bash
# Source ROS 2 (and pick_ik source overlay if present), then run the command.
# Standard ROS-docker pattern: makes `docker run so101-ros:jazzy <cmd>` work without
# the caller having to source anything, for both interactive and scripted use.
set -e
source /opt/ros/jazzy/setup.bash
[ -f /opt/pick_ik_ws/install/setup.bash ] && source /opt/pick_ik_ws/install/setup.bash
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
exec "$@"
