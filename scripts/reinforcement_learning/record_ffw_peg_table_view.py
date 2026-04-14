#!/usr/bin/env python3
"""Record a short RGB video of the peg scene (table / fixture layout) without a policy checkpoint."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record peg-scene video (no checkpoint).")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=240)
parser.add_argument(
    "--video_folder",
    type=str,
    default="/tmp/ffw_sg2_peg_table_view_video",
    help="Output folder for rl-video-step-0.mp4",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no-orbit-camera",
    action="store_true",
    help="Disable orbital camera motion (default: slow orbit — half the legacy angular rate; π rad over --video_length unless you lengthen the clip).",
)
parser.add_argument("--orbit_radius", type=float, default=1.65, help="Horizontal orbit radius (m).")
parser.add_argument("--orbit_z_offset", type=float, default=0.48, help="Eye height above look-at Z (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    Path(args_cli.video_folder).mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(Path(args_cli.video_folder) / "run")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.video_folder,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    obs, _ = env.reset(seed=args_cli.seed)
    device = env_cfg.sim.device
    if not (isinstance(device, str) and device.startswith("cuda")):
        device = "cuda:0"

    look = tuple(float(x) for x in env_cfg.viewer.lookat)
    lx, ly, lz = look

    # Isaac Lab: use action manager total dim (Box shape can be misleading for vectorized envs).
    act_dim = env.unwrapped.action_manager.total_action_dim
    n = env_cfg.scene.num_envs
    n_steps = max(1, args_cli.video_length - 1)
    for step in range(args_cli.video_length):
        if not args_cli.no_orbit_camera:
            # Half the angular speed of the legacy orbit (π rad over the clip; double --video_length for a slow 360°).
            theta = math.pi * (step / n_steps)
            r = args_cli.orbit_radius
            eye = (
                lx + r * math.cos(theta),
                ly + r * math.sin(theta),
                lz + args_cli.orbit_z_offset,
            )
            env.unwrapped.sim.set_camera_view(eye=eye, target=look)
        actions = 0.05 * torch.randn(n, act_dim, device=device)
        obs, _rew, _term, _trunc, _info = env.step(actions)

    env.close()
    print(f"[INFO] Video folder: {args_cli.video_folder}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
