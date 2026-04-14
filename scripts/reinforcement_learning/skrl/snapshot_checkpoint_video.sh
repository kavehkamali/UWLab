#!/usr/bin/env bash
# Record short debug videos from the latest SKRL checkpoint WITHOUT slowing the main training process.
#
# Run this in a *separate* tmux session while training runs headless (no --video on train.py).
#
# Required env:
#   TASK              Gym task id, e.g. OmniReset-FFWSG2-PegPartialAssemblySmoke-v0
#   RUN_DIR           Absolute path to a specific run folder containing checkpoints/, e.g.
#                     /home/kaveh/projects/API/UWLab/logs/skrl/omnireset_ffw_sg2_peg/2026-04-12_18-46-30_ppo_torch
#
# Optional env:
#   SLEEP_SEC         Seconds between attempts (default: 900)
#   VIDEO_LENGTH      Steps to record per clip (default: 240)
#   NUM_ENVS          Parallel envs for the play job (default: 1; keep small for speed)
#   CUDA_VISIBLE_DEVICES  GPU for the snapshot job (default: 1)
#
set -euo pipefail

: "${TASK:?Set TASK to the Gym id (same as training)}"
: "${RUN_DIR:?Set RUN_DIR to the absolute skrl run directory that contains checkpoints/}"

SLEEP_SEC="${SLEEP_SEC:-900}"
VIDEO_LENGTH="${VIDEO_LENGTH:-240}"
NUM_ENVS="${NUM_ENVS:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

UWLAB="/home/kaveh/projects/API/UWLab"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$UWLAB/env_uwlab}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-$HOME/uwlab_hf_assets}"

cd "$UWLAB"

last_ckpt=""

while true; do
  latest="$(ls -t "$RUN_DIR"/checkpoints/agent_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "[snapshot-video] no checkpoints yet under $RUN_DIR/checkpoints; sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
    continue
  fi

  if [[ "$latest" == "$last_ckpt" ]]; then
    echo "[snapshot-video] no new checkpoint (still $(basename "$latest")); sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
    continue
  fi

  echo "[snapshot-video] recording from $(basename "$latest")"
  last_ckpt="$latest"

  ./uwlab.sh -p scripts/reinforcement_learning/skrl/play.py     --headless     --task "$TASK"     --checkpoint "$latest"     --num_envs "$NUM_ENVS"     --device cuda:0     --video     --video_length "$VIDEO_LENGTH"     || echo "[snapshot-video] play.py failed (continuing)"

  sleep "$SLEEP_SEC"
done
