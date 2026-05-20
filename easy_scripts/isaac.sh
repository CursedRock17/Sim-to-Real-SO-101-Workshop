xhost +
docker run --rm -it --gpus all --rm --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
nvcr.io/nvidia/isaac-lab:2.3.2 bash
