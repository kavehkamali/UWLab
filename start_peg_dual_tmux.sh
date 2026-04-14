#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/home/kaveh/projects/API/UWLab/env_uwlab
export UWLAB_CLOUD_ASSETS_DIR=/home/kaveh/uwlab_hf_assets
export OMNI_KIT_ACCEPT_EULA=YES
cd /home/kaveh/projects/API/UWLab
UID_DIR=/tmp/tmux-$(id -u)
rm -rf "$UID_DIR"
mkdir -m 700 -p "$UID_DIR"
pkill -f "train.py.*OmniReset-FFWSG2-PegPartialAssemblySmoke-v0" 2>/dev/null || true
pkill -f "train.py.*OmniReset-FFWSG2-PegPartialAssemblySmoke-ApproachLift-v0" 2>/dev/null || true
tmux kill-session -t peg-full-gpu0 2>/dev/null || true
tmux kill-session -t peg-full-gpu1 2>/dev/null || true
tmux kill-server 2>/dev/null || true
sleep 2
ENV_EXPORTS="export VIRTUAL_ENV=${VIRTUAL_ENV} UWLAB_CLOUD_ASSETS_DIR=${UWLAB_CLOUD_ASSETS_DIR} OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA}"
TASK=OmniReset-FFWSG2-PegPartialAssemblySmoke-ApproachLift-v0
echo "[start_peg_dual_tmux] UWLAB_CLOUD_ASSETS_DIR=${UWLAB_CLOUD_ASSETS_DIR}"
echo "[start_peg_dual_tmux] Launching ${TASK} on cuda:0 (full run) ..."
tmux new-session -d -s peg-full-gpu0 "${ENV_EXPORTS} && bash uwlab.sh -p scripts/reinforcement_learning/skrl/train.py --headless --task ${TASK} --device cuda:0 --num_envs 16384 --seed 42 --log-reward-terms-every 12 2>&1 | tee /tmp/peg_full_gpu0.log"
echo "[start_peg_dual_tmux] Waiting 45s before second ${TASK} on cuda:1 (stagger Kit/GPU init) ..."
sleep 45
tmux new-session -d -s peg-full-gpu1 "${ENV_EXPORTS} && bash uwlab.sh -p scripts/reinforcement_learning/skrl/train.py --headless --task ${TASK} --device cuda:1 --num_envs 16384 --seed 43 --log-reward-terms-every 12 2>&1 | tee /tmp/peg_full_gpu1.log"
echo "tmux ls:"
tmux ls
echo "Logs: /tmp/peg_full_gpu0.log /tmp/peg_full_gpu1.log"
