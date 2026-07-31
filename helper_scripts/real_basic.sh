xhost +
docker run -it --rm --name real-robot --network host --privileged --gpus all --shm-size=16g \
    -e DISPLAY \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/models:/workspace/models \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/docker/env:/root/env \
    -v ~/Documents/Sim-to-Real-SO-101-Workshop/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100 \
    real-robot \
    /bin/bash
