#!/usr/bin/env bash
# Dual-GPU training: PegApproachOnly A (cuda:0) vs B (cuda:1), ~16k envs each.
set -euo pipefail
export VIRTUAL_ENV="${VIRTUAL_ENV:-/home/kaveh/projects/API/UWLab/env_uwlab}"
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-/home/kaveh/uwlab_hf_assets}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
cd /home/kaveh/projects/API/UWLab

UID_DIR="/tmp/tmux-$(id -u)"
rm -rf "$UID_DIR"
mkdir -m 700 -p "$UID_DIR"

pkill -f "train.py.*OmniReset-FFWSG2-PegApproachOnly-A-v0" 2>/dev/null || true
pkill -f "train.py.*OmniReset-FFWSG2-PegApproachOnly-B-v0" 2>/dev/null || true
tmux kill-session -t peg-approach-a 2>/dev/null || true
tmux kill-session -t peg-approach-b 2>/dev/null || true
tmux kill-server 2>/dev/null || true
sleep 2

ENV_EXPORTS="export VIRTUAL_ENV=${VIRTUAL_ENV} UWLAB_CLOUD_ASSETS_DIR=${UWLAB_CLOUD_ASSETS_DIR} OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA}"
TASK_A="OmniReset-FFWSG2-PegApproachOnly-A-v0"
TASK_B="OmniReset-FFWSG2-PegApproachOnly-B-v0"

echo "[start_peg_approachonly_dual_tmux] UWLAB_CLOUD_ASSETS_DIR=${UWLAB_CLOUD_ASSETS_DIR}"
echo "[start_peg_approachonly_dual_tmux] Launching ${TASK_A} on cuda:0 ..."
tmux new-session -d -s peg-approach-a \
  "${ENV_EXPORTS} && bash uwlab.sh -p scripts/reinforcement_learning/skrl/train.py --headless --task ${TASK_A} --device cuda:0 --num_envs 16384 --seed 42 --log-reward-terms-every 12 2>&1 | tee /tmp/peg_approachonly_a.log"

echo "[start_peg_approachonly_dual_tmux] Waiting 45s before ${TASK_B} on cuda:1 ..."
sleep 45

tmux new-session -d -s peg-approach-b \
  "${ENV_EXPORTS} && bash uwlab.sh -p scripts/reinforcement_learning/skrl/train.py --headless --task ${TASK_B} --device cuda:1 --num_envs 16384 --seed 43 --log-reward-terms-every 12 2>&1 | tee /tmp/peg_approachonly_b.log"

echo "tmux ls:"
tmux ls
echo "Logs: /tmp/peg_approachonly_a.log /tmp/peg_approachonly_b.log"
