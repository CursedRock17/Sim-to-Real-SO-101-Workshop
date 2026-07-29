# Standalone ROS 2 Jazzy + MoveIt 2 (+ pick_ik) container.
#
# Runs SEPARATELY from the Isaac Sim container (docker/sim) and talks to it over
# ROS 2 DDS on --network=host. Keeping ROS out of the Isaac image is the whole point:
# the Isaac image ships an embedded kit-Python 3.11, so `source /opt/ros/jazzy/setup.bash`
# there runs the wrong python and dies ("SRE module mismatch"). Here python3 is the
# stock Ubuntu 24.04 Python 3.12 that ROS 2 Jazzy expects, so sourcing just works.
#
# Mirrors NVIDIA's ubuntu_24_jazzy_python_312_minimal approach (separate Ubuntu 24.04
# base) but installs ROS from apt binaries instead of a source build -- far simpler,
# and message-compatible with the Isaac Sim Jazzy ROS 2 bridge.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV ROS_DISTRO=jazzy
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# Locale + repo prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg2 lsb-release software-properties-common ca-certificates && \
    locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# ROS 2 apt repository (keyring + source list)
RUN add-apt-repository -y universe && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list

# ROS 2 Jazzy base + MoveIt 2 + moveit_py + FastDDS RMW + build/eval tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-moveit \
        ros-jazzy-moveit-py \
        ros-jazzy-moveit-setup-assistant \
        ros-jazzy-rmw-fastrtps-cpp \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
        git && \
    rm -rf /var/lib/apt/lists/*

# pick_ik: best-effort binary install (non-fatal). If there's no arm64 deb we build
# it from source in a follow-up step rather than failing the whole image.
RUN apt-get update && \
    ( apt-get install -y --no-install-recommends ros-jazzy-pick-ik \
        && echo "PICK_IK: installed from apt" \
      || echo "PICK_IK: no binary deb -- build from source separately" ) ; \
    rm -rf /var/lib/apt/lists/* ; true

RUN rosdep init 2>/dev/null || true && rosdep update || true

# Auto-source ROS for every shell (stock Python 3.12 -- no Isaac collision here).
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /opt/pick_ik_ws/install/setup.bash ] && source /opt/pick_ik_ws/install/setup.bash' >> /root/.bashrc && \
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /root/.bashrc

# Entrypoint sources ROS for every `docker run` (interactive or scripted).
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
