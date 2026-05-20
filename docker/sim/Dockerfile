FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta1

ARG LEROBOT_REF=d656da8ccca5989ff0a2207e81fbfa2c2d5bafb1

ENV OMNI_KIT_ACCEPT_EULA=YES
ENV PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ENV PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"


WORKDIR /workspace
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /workspace/lerobot
RUN git checkout "${LEROBOT_REF}"

# Install lerobot without deps to avoid upgrading packages that break
# Isaac Lab / Isaac Sim packages (torch, torchvision, USD, Kit, etc.)
RUN $PYTHON -m pip install --no-deps -e .

# Keep the shared dependency set compatible with LeRobot 0.5.1 and Isaac Lab develop.
RUN printf '%s\n' \
    "numpy>=2.0.0,<2.3.0" \
    "packaging>=24.2,<26.0" \
    "pillow==12.1.1" \
    "setuptools>=71.0.0,<81.0.0" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "gymnasium==1.2.1" \
    > /tmp/constraints.txt

# Install the LeRobot extras used by the workshop without downgrading Isaac Lab 3.
RUN $PYTHON -m pip install -c /tmp/constraints.txt \
    "opencv-python-headless>=4.9.0,<4.14.0" \
    "Pillow==12.1.1" \
    "einops>=0.8.0,<0.9.0" \
    "draccus==0.10.0" \
    "huggingface-hub[hf-transfer,cli]>=1.0.0,<2.0.0" \
    "requests>=2.32.0,<3.0.0" \
    "safetensors>=0.4.3,<1.0.0" \
    "termcolor>=2.4.0,<4.0.0" \
    "transformers>=5.8.1" \
    "tqdm>=4.66.0,<5.0.0" \
    "datasets>=4.0.0,<5.0.0" \
    "pandas>=2.0.0,<3.0.0" \
    "pyarrow>=21.0.0,<30.0.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "wandb>=0.24.0,<0.25.0" \
    "pynput>=1.7.8,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "feetech-servo-sdk>=1.0.0,<2.0.0" \
    "rerun-sdk>=0.24.0,<0.27.0" \
    "pyzmq>=26.2.1,<28.0.0" \
    #"torchcodec>=0.10.0,<0.11.0" \
    "cmake>=3.29.0.1,<4.2.0"

RUN curl --proto "=https" --tlsv1.2 -sSf -L -o /tmp/ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /usr/local --strip-components=1 && \
    ldconfig && \
    rm /tmp/ffmpeg.tar.xz

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Sim-to-Real-SO-101-Workshop

COPY ./docker/sim/entrypoint.sh /workspace/Sim-to-Real-SO-101-Workshop/docker/sim/entrypoint.sh
COPY ./docker/utils.sh /workspace/Sim-to-Real-SO-101-Workshop/docker/utils.sh
RUN chmod +x docker/sim/entrypoint.sh

RUN $PYTHON -m pip install --upgrade pip

RUN mkdir -p /tmp/pycache && chmod 1777 /tmp/pycache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

RUN cat docker/utils.sh >> /root/.bashrc
ENTRYPOINT ["/workspace/Sim-to-Real-SO-101-Workshop/docker/sim/entrypoint.sh"]
CMD ["bash"]
