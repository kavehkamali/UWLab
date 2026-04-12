# FFW-SG2 peg partial assembly (smoke) — training, inference, and video

Task id: `OmniReset-FFWSG2-PegPartialAssemblySmoke-v0`  
Env config: `uwlab_tasks/.../omnireset_sg2/config/ffw_sg2_peg_partial_smoke_env_cfg.py`

Assume repo root `UWLab`, venv `./env_uwlab`, and:

```bash
export UWLAB_CLOUD_ASSETS_DIR=/path/to/uwlab_hf_assets
export OMNI_KIT_ACCEPT_EULA=YES
cd /path/to/UWLab
```

---

## Training (headless, no dataset-style video)

Training **does not** record MP4s unless you pass `--video`. For maximum throughput, **omit** `--video` and **omit** `--enable_cameras`** (default off).

**PhysX material limit:** keep `--num_envs` at **8192** or lower per GPU process (64k PhysX materials cap).

```bash
./env_uwlab/bin/python scripts/reinforcement_learning/skrl/train.py \
  --task OmniReset-FFWSG2-PegPartialAssemblySmoke-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 8192 \
  --seed 42 \
  --log-reward-terms-every 24 \
  2>&1 | tee /tmp/ffw_sg2_peg_train.log
```

- **`--log-reward-terms-every 24`**: prints mean reward + per-term breakdown about once per PPO rollout; use **`0`** to disable.
- Logs and checkpoints: `logs/skrl/omnireset_ffw_sg2_peg/<timestamp>_ppo_torch/` (see `skrl_ppo_cfg.yaml` for `write_interval` / `checkpoint_interval`).

### Remove training-time video rendering

1. **Do not** pass `--video` (that wraps `RecordVideo` and forces camera work).
2. **Do not** pass `--enable_cameras` unless you need sensors or video.
3. If you previously enabled video in a tmux command, delete those flags from the launch line.

---

## Inference (policy play, no MP4)

Requires a `.pt` checkpoint compatible with the current observation size (same task / MDP).

```bash
./env_uwlab/bin/python scripts/reinforcement_learning/skrl/play.py \
  --task OmniReset-FFWSG2-PegPartialAssemblySmoke-v0 \
  --checkpoint /path/to/logs/skrl/omnireset_ffw_sg2_peg/<run>/checkpoints/best_agent.pt \
  --headless \
  --device cuda:0 \
  --num_envs 4 \
  --seed 0
```

Omit `--video`; run runs until you interrupt (or add your own early exit).

---

## Inference + fixed-camera video (viewer from env config)

Uses **`ViewerCfg`** in the env (static eye / lookat during the clip). Enables cameras and `RecordVideo`.

```bash
./env_uwlab/bin/python scripts/reinforcement_learning/skrl/play.py \
  --task OmniReset-FFWSG2-PegPartialAssemblySmoke-v0 \
  --checkpoint /path/to/best_agent.pt \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --video \
  --video_length 400 \
  --num_envs 4 \
  --seed 0
```

MP4 default folder: `<run_dir>/videos/play/` next to the checkpoint’s experiment (see play log for `video_folder`).

---

## Inference + rotating camera video (orbit viewport)

Same as fixed video, plus **orbit** flags (360° over `--video_length`):

```bash
./env_uwlab/bin/python scripts/reinforcement_learning/skrl/play.py \
  --task OmniReset-FFWSG2-PegPartialAssemblySmoke-v0 \
  --checkpoint /path/to/best_agent.pt \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --video \
  --video_length 480 \
  --orbit-camera \
  --orbit-radius 1.75 \
  --orbit-z-offset 0.52 \
  --num_envs 4 \
  --seed 0
```

Requires `env_cfg.viewer` / `lookat` (this task provides it).

---

## Scene-only video (no checkpoint, random small motions)

Useful to verify table / peg / hole layout and camera orbit **without** a trained policy:

```bash
./env_uwlab/bin/python scripts/reinforcement_learning/record_ffw_peg_table_view.py \
  --headless \
  --device cuda:0 \
  --num_envs 1 \
  --video_length 480 \
  --video_folder /tmp/ffw_peg_scene_video \
  --orbit-radius 1.75 \
  --orbit-z-offset 0.52 \
  --seed 3
```

- **Fixed camera** for this script: add `--no-orbit-camera`.
- Output: `rl-video-step-0.mp4` inside `--video_folder`.

---

## tmux (optional)

```bash
tmux new-session -d -s ffw_sg2_peg_train "bash -lc 'cd /path/to/UWLab && export UWLAB_CLOUD_ASSETS_DIR=... OMNI_KIT_ACCEPT_EULA=YES && ./env_uwlab/bin/python scripts/reinforcement_learning/skrl/train.py ... 2>&1 | tee /tmp/ffw_sg2_peg_train.log'"
tmux attach -t ffw_sg2_peg_train
```
