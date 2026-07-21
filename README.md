# Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac

This is an offshoot of the original SO-101 Training in Isaac Sim Workshop fit with custom environments for naval research.

## Requirements

This runs on the following GPUs:

- DGX Spark, GB10 (Blackwell)

OS and Software tested:
- Ubuntu Linux >22.04
- Docker
- CUDA Toolkit
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


## Installation

1. Create directory and clone this repo
```bash
mkdir ~/Documents
cd ~/Documents
git clone https://github.com/CursedRock17/Sim-to-Real-SO-101-Workshop -b naval_research
```

### Building the Docker images

2. Navigate to the repo
```bash
cd ~/Documents/Sim-to-Real-SO-101-Workshop
```

#### Teleop & Simulation container

3. From the repo root directory, run:
```bash
docker build -t teleop-docker -f docker/sim/Dockerfile .
```

#### Real Robot & Inference Server - this may take a while to build


For **Blackwell** architecture GPUs:

4. From the repo root directory, run:
```bash
./docker/real/build.sh blackwell
```

### Starting the images

To start the Teleop & Simulation container, run [./helper_scripts/sim_basic.sh](./helper_scripts/sim_basic.sh)
or:

```bash
xhost + 
docker run --name teleop \
   -it --privileged --gpus all --rm --network=host \
   -e ACCEPT_EULA=Y \
   -e PRIVACY_CONSENT=Y \
   -e DISPLAY=$DISPLAY \
   -e NVIDIA_DRIVER_CAPABILITIES=all \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
   -v ./docker/env:/root/env \
   -v $(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source \
   -v $(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs \
   -v $(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets \
   teleop-docker:latest
```

To start the Real Robot & Inference Server run [./helper_scripts/real_basic.sh](./helper_scripts/real_basic.sh)
or:

```bash
xhost +
docker run -it --rm --name real-robot --network host --privileged --gpus all \
    -e DISPLAY \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/models:/workspace/models \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/docker/env:/root/env \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100 \
    real-robot \
    /bin/bash
```

## Models and Datasets

### Downloading model weights

First, [install the HuggingFace command-line-interface (CLI)](https://huggingface.co/docs/huggingface_hub/en/guides/cli#command-line-interface-cli)

## Tasks
TODO: Add in our tasks

### Tasks

#### Debug envs
- `Lerobot-So101-Teleop-Base` : Teleop debug
- `Lerobot-So101-Teleop-Task` : Lightbox, cameras, non-task related debug

#### Tasks
- `Lerobot-So101-Teleop-Vials-To-Rack` - Main task for the workshop - pick up the vial and place it in the yellow rack
- `Lerobot-So101-Teleop-Vials-To-Rack-DR` - Same as above but with domain randomization

#### Eval
- `Lerobot-So101-Teleop-Vials-To-Rack-Eval` - Evaluation without domain randomization (fixed orange robot, no lighting/mat DR)
- `Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval` - Evaluation with full domain randomization


## Commands

- `list_envs` - List environments in this repo
- `zero_agent` - Debug script with zero actions
- `random_agent` - Debug script with random actions
- `lerobot_agent` - LeRobot SO101 teleop script
- `lerobot_eval` - Model evaluation script
- `lerobot_push_dataset` - LeRobot Dataset push to hub script
