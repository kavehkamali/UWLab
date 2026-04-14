#!/usr/bin/env bash
# Batch-record inference videos (orbiting camera) for PegApproachOnly A/B from many SKRL checkpoints.
#
# Videos are written under each run's logs/.../videos/play/ by play.py; this script copies each
# clip to OUTPUT_DIR with a unique name so later checkpoints do not overwrite earlier ones.
#
# Usage (on the lab machine, from UWLab repo root or any cwd — script cds to UWLAB):
#   bash scripts/reinforcement_learning/skrl/record_approachonly_checkpoint_videos.sh
#
# Optional environment:
#   UWLAB              Repo root (default: /home/kaveh/projects/API/UWLab)
#   OUTPUT_DIR         Where to copy mp4s (default: $UWLAB/logs/skrl/approach_only_inference_videos/<timestamp>)
#   RUN_DIR_A          Absolute path to run folder .../omnireset_ffw_sg2_peg_approach_only_a/<date>_ppo_torch
#   RUN_DIR_B          Same for B (if unset, picks latest run under each experiment log root)
#   CUDA_DEVICE        cuda:N for play jobs (default: 0)
#   VIDEO_LENGTH       Steps per clip (default: 480 — ~2× default for a slower half-speed orbit lap)
#   NUM_ENVS           Inference envs (default: 1)
#   SKIP_EXISTING      If 1, skip when destination mp4 already exists (default: 1)
#   INCLUDE_BEST       If 1, also record best_agent.pt (default: 0)
#   CKPT_MAX_PER_SIDE  If set and >0, record at most this many checkpoints per side, spread
#                        evenly across saved agent_*.pt (default: 0 = all).
#   Task B uses OmniReset-FFWSG2-PegApproachOnly-B-InferCkpt256-v0 (registered in omnireset_sg2 config __init__.py)
#   so Hydra loads the 256×256 agent yaml matching older B checkpoints.
#
set -euo pipefail

UWLAB="${UWLAB:-/home/kaveh/projects/API/UWLab}"
cd "$UWLAB"

export VIRTUAL_ENV="${VIRTUAL_ENV:-$UWLAB/env_uwlab}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-$HOME/uwlab_hf_assets}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-480}"
NUM_ENVS="${NUM_ENVS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INCLUDE_BEST="${INCLUDE_BEST:-0}"
CKPT_MAX_PER_SIDE="${CKPT_MAX_PER_SIDE:-0}"
pick_at_most_lines() {
  local max="$1"
  local -a lines=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    lines+=("$line")
  done
  local n="${#lines[@]}"
  (( n == 0 )) && return 0
  if (( max <= 0 || n <= max )); then
    printf '%s\n' "${lines[@]}"
    return 0
  fi
  if (( max == 1 )); then
    printf '%s\n' "${lines[n - 1]}"
    return 0
  fi
  local i idx
  for ((i = 0; i < max; i++)); do
    idx=$((i * (n - 1) / (max - 1)))
    printf '%s\n' "${lines[idx]}"
  done
}

TS="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$UWLAB/logs/skrl/approach_only_inference_videos/$TS}"
mkdir -p "$OUTPUT_DIR"

ROOT_A="$UWLAB/logs/skrl/omnireset_ffw_sg2_peg_approach_only_a"
ROOT_B="$UWLAB/logs/skrl/omnireset_ffw_sg2_peg_approach_only_b"

pick_latest_run() {
  local root="$1"
  ls -td "$root"/*_ppo_torch 2>/dev/null | head -n 1 || true
}

if [[ -z "${RUN_DIR_A:-}" ]]; then
  RUN_DIR_A="$(pick_latest_run "$ROOT_A")"
fi
if [[ -z "${RUN_DIR_B:-}" ]]; then
  RUN_DIR_B="$(pick_latest_run "$ROOT_B")"
fi

if [[ -z "$RUN_DIR_A" || ! -d "$RUN_DIR_A/checkpoints" ]]; then
  echo "[record] ERROR: RUN_DIR_A not found. Set RUN_DIR_A to a run with checkpoints/ (got: ${RUN_DIR_A:-empty})" >&2
  exit 1
fi
if [[ -z "$RUN_DIR_B" || ! -d "$RUN_DIR_B/checkpoints" ]]; then
  echo "[record] ERROR: RUN_DIR_B not found. Set RUN_DIR_B to a run with checkpoints/ (got: ${RUN_DIR_B:-empty})" >&2
  exit 1
fi

echo "[record] OUTPUT_DIR=$OUTPUT_DIR"
echo "[record] RUN_DIR_A=$RUN_DIR_A"
echo "[record] RUN_DIR_B=$RUN_DIR_B"

record_one() {
  local label="$1"
  local task="$2"
  local run_dir="$3"
  local ckpt_path="$4"
  local ckpt_tag="$5"

  local dest="$OUTPUT_DIR/${label}_${ckpt_tag}_orbit.mp4"
  if [[ "$SKIP_EXISTING" == "1" && -f "$dest" ]]; then
    echo "[record] skip (exists): $dest"
    return 0
  fi

  echo "[record] --- $label $ckpt_tag ---"
  rm -rf "$run_dir/videos/play" 2>/dev/null || true
  mkdir -p "$run_dir/videos/play"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" ./uwlab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --headless \
    --task "$task" \
    --checkpoint "$ckpt_path" \
    --num_envs "$NUM_ENVS" \
    --device "cuda:0" \
    --video \
    --video_length "$VIDEO_LENGTH" \
    --orbit-camera

  # play.py writes under <run_dir>/videos/play/
  local src
  src="$(ls -t "$run_dir/videos/play"/*.mp4 2>/dev/null | head -n 1 || true)"
  if [[ -z "$src" ]]; then
    echo "[record] WARNING: no mp4 produced for $label $ckpt_tag under $run_dir/videos/play" >&2
    return 0
  fi
  cp -f "$src" "$dest"
  echo "[record] wrote $dest"
}

list_mid_checkpoints() {
  local d="$1"
  shopt -s nullglob
  # Numeric agent_<timesteps>.pt only, sorted by timestep (stable paths after cut).
  for f in "$d"/checkpoints/agent_*.pt; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f" .pt)"
    num="${base#agent_}"
    [[ "$num" =~ ^[0-9]+$ ]] || continue
    printf '%010d\t%s\n' "$num" "$f"
  done | sort -n | cut -f2-
}

TASK_A="OmniReset-FFWSG2-PegApproachOnly-A-v0"
TASK_B="OmniReset-FFWSG2-PegApproachOnly-B-InferCkpt256-v0"

while read -r ckpt; do
  [[ -z "$ckpt" ]] && continue
  tag="$(basename "$ckpt" .pt)"
  record_one "A" "$TASK_A" "$RUN_DIR_A" "$ckpt" "$tag"
done < <(list_mid_checkpoints "$RUN_DIR_A" | pick_at_most_lines "$CKPT_MAX_PER_SIDE")

while read -r ckpt; do
  [[ -z "$ckpt" ]] && continue
  tag="$(basename "$ckpt" .pt)"
  record_one "B" "$TASK_B" "$RUN_DIR_B" "$ckpt" "$tag"
done < <(list_mid_checkpoints "$RUN_DIR_B" | pick_at_most_lines "$CKPT_MAX_PER_SIDE")

if [[ "$INCLUDE_BEST" == "1" ]]; then
  if [[ -f "$RUN_DIR_A/checkpoints/best_agent.pt" ]]; then
    record_one "A" "$TASK_A" "$RUN_DIR_A" "$RUN_DIR_A/checkpoints/best_agent.pt" "best_agent"
  fi
  if [[ -f "$RUN_DIR_B/checkpoints/best_agent.pt" ]]; then
    record_one "B" "$TASK_B" "$RUN_DIR_B" "$RUN_DIR_B/checkpoints/best_agent.pt" "best_agent"
  fi
fi

echo "[record] done. Clips: $(ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ') files in $OUTPUT_DIR"
