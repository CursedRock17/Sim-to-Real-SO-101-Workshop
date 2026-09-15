#!/bin/bash
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim

export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source ${ISAAC_SIM}/setup_python_env.sh
source /root/env 2>/dev/null

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

python -m pip install -e /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/

# The `lerobot_agent` console entry point pip just installed has a shebang pointing
# at the bare kit python3, which bypasses python.sh and never primes the kit/carb
# plugin loader -- so `import omni.usd` fails to resolve libomni.usd.so. Overwrite
# it in place (kit/python/bin is ahead of /usr/local/bin on PATH, so a wrapper
# there wouldn't shadow it) with one that runs through python.sh, like isaaclab.sh -p.
LEROBOT_AGENT_BIN=/workspace/isaaclab/_isaac_sim/kit/python/bin/lerobot_agent
cat > "$LEROBOT_AGENT_BIN" << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh \
  /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/scripts/lerobot_agent.py "$@"
WRAPPER
chmod +x "$LEROBOT_AGENT_BIN"

exec "$@"
