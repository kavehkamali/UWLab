#!/usr/bin/env bash
# Single training job on 2 GPUs via torchrun + train.py --distributed.
# ``--num-envs`` is the *global* parallel env count; train.py splits it by WORLD_SIZE (one Isaac process per GPU).
set -euo pipefail

export VIRTUAL_ENV="${VIRTUAL_ENV:-/home/kaveh/projects/API/UWLab/env_uwlab}"
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-/home/kaveh/uwlab_hf_assets}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export UWLAB_PATH="${UWLAB_PATH:-/home/kaveh/projects/API/UWLab}"
cd "${UWLAB_PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# --- NCCL / torch.distributed (single-node Isaac + torchrun) ---
# Isaac can hold the Python GIL or block in CUDA/PhysX for long stretches while ranks still need
# NCCL collectives (e.g. parameter broadcast). PyTorch's NCCL "heartbeat" monitor then aborts after ~480s
# even when training is healthy. Prefer disabling the monitor; optionally raise the timeout instead.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
# This machine previously logged CUDA P2P peer-access warnings; disabling P2P avoids flaky NVLink/P2P paths.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

# Default: stable on 2× RTX 6000 Ada for this task (PhysX 64k material limit + GPU headroom for PhysX buffers).
# 49k+ global envs filled VRAM and caused PhysX CUDA OOM / scene corruption; 32k total stayed healthy in testing.
# Override with NUM_ENV_TOTAL=... for sweeps (must stay divisible by NPROC).
NUM_ENV_TOTAL="${NUM_ENV_TOTAL:-32768}"
TASK="${TASK:-OmniReset-FFWSG2-PegApproachOnly-A-v0}"
SEED="${SEED:-42}"
LOG_REWARD_EVERY="${LOG_REWARD_EVERY:-12}"
NPROC="${NPROC:-2}"

TORCHRUN="${VIRTUAL_ENV}/bin/torchrun"
if [[ ! -x "${TORCHRUN}" ]]; then
  echo "[ERROR] torchrun not found at ${TORCHRUN} (install torch in the UWLab venv)." >&2
  exit 1
fi
# torchrun runs the entrypoint with the *same* interpreter that launched torchrun (the venv Python).
# Do not pass the python binary as the first argument — torchrun would try to parse the ELF as source.

if (( NUM_ENV_TOTAL % NPROC != 0 )); then
  echo "[ERROR] NUM_ENV_TOTAL (${NUM_ENV_TOTAL}) must be divisible by NPROC (${NPROC}) so each rank gets the same env count." >&2
  exit 1
fi

echo "[start_peg_approachonly_ddp_2gpu] UWLAB_PATH=${UWLAB_PATH}"
echo "[start_peg_approachonly_ddp_2gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[start_peg_approachonly_ddp_2gpu] global num_envs=${NUM_ENV_TOTAL}  task=${TASK}  nproc=${NPROC}"

# Stop the older dual-single-GPU tmux layout if it is still around.
pkill -f "train.py.*OmniReset-FFWSG2-PegApproachOnly-A-v0" 2>/dev/null || true
pkill -f "train.py.*OmniReset-FFWSG2-PegApproachOnly-B-v0" 2>/dev/null || true
tmux kill-session -t peg-approach-a 2>/dev/null || true
tmux kill-session -t peg-approach-b 2>/dev/null || true

exec "${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  scripts/reinforcement_learning/skrl/train.py \
  --headless \
  --distributed \
  --device cuda \
  --task "${TASK}" \
  --num_envs "${NUM_ENV_TOTAL}" \
  --seed "${SEED}" \
  --log-reward-terms-every "${LOG_REWARD_EVERY}" \
  "$@"
