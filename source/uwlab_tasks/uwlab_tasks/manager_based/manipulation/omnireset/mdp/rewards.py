# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from functools import lru_cache
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from ..assembly_keypoints import Offset
from . import utils
from .collision_analyzer_cfg import CollisionAnalyzerCfg
from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ee_asset_distance_tanh(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.root_asset_cfg = cfg.params.get("root_asset_cfg")
        self.target_asset_cfg = cfg.params.get("target_asset_cfg")
        self.std = cfg.params.get("std")

        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")

        self.root_asset = env.scene[self.root_asset_cfg.name]
        root_usd_path = self.root_asset.cfg.spawn.usd_path
        root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
        root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
        self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))

        self.target_asset = env.scene[self.target_asset_cfg.name]
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        root_asset_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        root_asset_offset_metadata_key: str,
        target_asset_offset_metadata_key: str | None = None,
        std: float = 0.1,
    ) -> torch.Tensor:
        root_asset_alignment_pos_w, root_asset_alignment_quat_w = self.root_asset_offset.combine(
            self.root_asset.data.body_link_pos_w[:, root_asset_cfg.body_ids].view(-1, 3),
            self.root_asset.data.body_link_quat_w[:, root_asset_cfg.body_ids].view(-1, 4),
        )
        if self.target_asset_offset is None:
            target_asset_alignment_pos_w = self.target_asset.data.root_pos_w.view(-1, 3)
            target_asset_alignment_quat_w = self.target_asset.data.root_quat_w.view(-1, 4)
        else:
            target_asset_alignment_pos_w, target_asset_alignment_quat_w = self.target_asset_offset.apply(
                self.target_asset
            )
        target_asset_in_root_asset_frame_pos, target_asset_in_root_asset_frame_angle_axis = (
            math_utils.compute_pose_error(
                root_asset_alignment_pos_w,
                root_asset_alignment_quat_w,
                target_asset_alignment_pos_w,
                target_asset_alignment_quat_w,
            )
        )

        pos_distance = torch.norm(target_asset_in_root_asset_frame_pos, dim=1)

        return 1 - torch.tanh(pos_distance / std)


@lru_cache(maxsize=32)
def _omnireset_assembled_offset_from_usd(usd_path: str) -> Offset:
    meta = utils.read_metadata_from_usd_directory(usd_path)
    assembled = meta.get("assembled_offset")
    return Offset(pos=tuple(assembled.get("pos")), quat=tuple(assembled.get("quat")))


def compute_omnireset_insertion_alignment(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg,
    receptive_asset_cfg: SceneEntityCfg,
    command_context: str = "task_command",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pose errors for insertive-in-receptive frame; same geometry as ``ProgressContext``."""
    task_command = env.command_manager.get_term(command_context)
    success_position_threshold = task_command.success_position_threshold
    success_orientation_threshold = task_command.success_orientation_threshold

    insertive_asset: Articulation | RigidObject = env.scene[insertive_asset_cfg.name]  # type: ignore
    receptive_asset: Articulation | RigidObject = env.scene[receptive_asset_cfg.name]  # type: ignore

    insertive_offset = _omnireset_assembled_offset_from_usd(insertive_asset.cfg.spawn.usd_path)
    receptive_offset = _omnireset_assembled_offset_from_usd(receptive_asset.cfg.spawn.usd_path)

    insertive_asset_alignment_pos_w, insertive_asset_alignment_quat_w = insertive_offset.apply(insertive_asset)
    receptive_asset_alignment_pos_w, receptive_asset_alignment_quat_w = receptive_offset.apply(receptive_asset)
    insertive_asset_in_receptive_asset_frame_pos, insertive_asset_in_receptive_asset_frame_quat = (
        math_utils.subtract_frame_transforms(
            receptive_asset_alignment_pos_w,
            receptive_asset_alignment_quat_w,
            insertive_asset_alignment_pos_w,
            insertive_asset_alignment_quat_w,
        )
    )
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(insertive_asset_in_receptive_asset_frame_quat)
    euler_xy_distance = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    xyz_distance = torch.norm(insertive_asset_in_receptive_asset_frame_pos, dim=1)
    position_aligned = xyz_distance < success_position_threshold
    orientation_aligned = euler_xy_distance < success_orientation_threshold
    success = orientation_aligned & position_aligned
    return euler_xy_distance, xyz_distance, position_aligned, orientation_aligned, success


def _sync_omnireset_progress_context_from_alignment(
    env: ManagerBasedRLEnv,
    euler_xy_distance: torch.Tensor,
    xyz_distance: torch.Tensor,
    position_aligned: torch.Tensor,
    orientation_aligned: torch.Tensor,
    success: torch.Tensor,
) -> None:
    """Expose live ``.success`` / monitor state for event terms when ``progress_context`` weight is 0."""
    if "progress_context" not in env.reward_manager.active_terms:
        return
    term = env.reward_manager.get_term_cfg("progress_context").func
    if not isinstance(term, ProgressContext):
        return
    term.publish_alignment(env, euler_xy_distance, xyz_distance, position_aligned, orientation_aligned, success)


class ProgressContext(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore
        self.receptive_asset: Articulation | RigidObject = env.scene[cfg.params.get("receptive_asset_cfg").name]  # type: ignore

        self.insertive_asset_offset = _omnireset_assembled_offset_from_usd(self.insertive_asset.cfg.spawn.usd_path)
        self.receptive_asset_offset = _omnireset_assembled_offset_from_usd(self.receptive_asset.cfg.spawn.usd_path)

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.xyz_distance = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self._omni_align_monitor_step: int = -1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.continuous_success_counter[:] = 0

    def publish_alignment(
        self,
        env: ManagerBasedRLEnv,
        euler_xy_distance: torch.Tensor,
        xyz_distance: torch.Tensor,
        position_aligned: torch.Tensor,
        orientation_aligned: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        self.euler_xy_distance[:] = euler_xy_distance
        self.xyz_distance[:] = xyz_distance
        self.position_aligned[:] = position_aligned
        self.orientation_aligned[:] = orientation_aligned
        self.success[:] = success

        step = int(env.common_step_counter)
        if self._omni_align_monitor_step == step:
            return
        self._omni_align_monitor_step = step

        self.continuous_success_counter[:] = torch.where(
            success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), success
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        receptive_asset_cfg: SceneEntityCfg,
        command_context: str = "task_command",
    ) -> torch.Tensor:
        euler_xy, xyz, pos_a, ori_a, succ = compute_omnireset_insertion_alignment(
            env, insertive_asset_cfg, receptive_asset_cfg, command_context
        )
        self.publish_alignment(env, euler_xy, xyz, pos_a, ori_a, succ)
        return torch.zeros(env.num_envs, device=env.device)


def dense_success_reward(
    env: ManagerBasedRLEnv,
    std: float,
    insertive_asset_cfg: SceneEntityCfg,
    receptive_asset_cfg: SceneEntityCfg,
    command_context: str = "task_command",
) -> torch.Tensor:
    euler_xy_distance, xyz_distance, pos_a, ori_a, succ = compute_omnireset_insertion_alignment(
        env, insertive_asset_cfg, receptive_asset_cfg, command_context
    )
    _sync_omnireset_progress_context_from_alignment(env, euler_xy_distance, xyz_distance, pos_a, ori_a, succ)

    angle_m = torch.exp(-euler_xy_distance / std)
    xyz_m = torch.exp(-xyz_distance / std)
    return torch.mean(torch.stack([angle_m, xyz_m], dim=0), dim=0)


def success_reward(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg,
    receptive_asset_cfg: SceneEntityCfg,
    command_context: str = "task_command",
) -> torch.Tensor:
    euler_xy_distance, xyz_distance, pos_a, ori_a, succ = compute_omnireset_insertion_alignment(
        env, insertive_asset_cfg, receptive_asset_cfg, command_context
    )
    _sync_omnireset_progress_context_from_alignment(env, euler_xy_distance, xyz_distance, pos_a, ori_a, succ)
    return torch.where(pos_a & ori_a, 1.0, 0.0)





def wrist_min_distance_to_asset_tanh(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
    std: float = 0.18,
) -> torch.Tensor:
    """Encourage either wrist link to move toward the target asset root (0 far, approaches 1 when close)."""

    robot: Articulation = env.scene[robot_cfg.name]
    target = env.scene[target_asset_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    left_pos = robot.data.body_link_pos_w[:, left_idx]
    right_pos = robot.data.body_link_pos_w[:, right_idx]
    target_pos = target.data.root_pos_w
    d_left = torch.norm(left_pos - target_pos, dim=-1)
    d_right = torch.norm(right_pos - target_pos, dim=-1)
    d = torch.minimum(d_left, d_right)
    return 1.0 - torch.tanh(d / std)


def insertive_xy_near_receptor_tanh(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg,
    receptive_asset_cfg: SceneEntityCfg,
    std: float = 0.10,
) -> torch.Tensor:
    """Encourage the peg to move over the hole in the table plane (XY only)."""

    insertive = env.scene[insertive_asset_cfg.name]
    receptive = env.scene[receptive_asset_cfg.name]
    dxy = torch.norm(insertive.data.root_pos_w[:, :2] - receptive.data.root_pos_w[:, :2], dim=-1)
    return 1.0 - torch.tanh(dxy / std)


def insertive_height_above_surface(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg,
    surface_z: float,
    scale: float = 0.10,
) -> torch.Tensor:
    """Normalize peg height above a fixed tabletop plane (proxy for lift progress)."""

    insertive = env.scene[insertive_asset_cfg.name]
    h = insertive.data.root_pos_w[:, 2] - float(surface_z)
    h = torch.clamp(h, min=0.0)
    return torch.clamp(h / float(scale), max=1.0)


def gripper_excitation_near_insertive(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    insertive_asset_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
    proximity_std: float = 0.22,
    joint_scale: float = 0.60,
) -> torch.Tensor:
    """Proxy grasp shaping: reward primary gripper motion when wrists are near the peg.

    This is **not** a contact-based grasp detector; it only couples gripper joint magnitude to proximity so
    learning can pick up a crude reach-then-actuate prior.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    insertive = env.scene[insertive_asset_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    peg = insertive.data.root_pos_w
    d_left = torch.norm(robot.data.body_link_pos_w[:, left_idx] - peg, dim=-1)
    d_right = torch.norm(robot.data.body_link_pos_w[:, right_idx] - peg, dim=-1)
    d = torch.minimum(d_left, d_right)
    gate = torch.exp(-d / float(proximity_std))

    q = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids]
    excitation = torch.mean(torch.abs(q), dim=1)
    shaped = torch.tanh(excitation / float(joint_scale))
    return gate * shaped


def wrist_min_distance_to_asset_exp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
    sigma: float = 0.32,
) -> torch.Tensor:
    """Distance-shaped approach reward using ``exp(-d / sigma)`` (more signal when wrists are far from peg).

    Terminal logging prints *weighted* values; this stays > ~0.02 for typical ~0.5--1.0 m distances, so you can
    see it move as the policy improves. Prefer this over ``tanh`` for wide workspaces.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    target = env.scene[target_asset_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    left_pos = robot.data.body_link_pos_w[:, left_idx]
    right_pos = robot.data.body_link_pos_w[:, right_idx]
    target_pos = target.data.root_pos_w
    d_left = torch.norm(left_pos - target_pos, dim=-1)
    d_right = torch.norm(right_pos - target_pos, dim=-1)
    d = torch.minimum(d_left, d_right)
    return torch.exp(-d / float(sigma))


def wrists_min_distance_to_asset_exp_sum(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
    sigma: float = 0.32,
) -> torch.Tensor:
    """Both wrists: ``exp(-dL/sigma) + exp(-dR/sigma)`` (pin = insertive root).

    Unlike :func:`wrist_min_distance_to_asset_exp`, this **does not** take ``min(left, right)``; both
    hands must be close to the peg for a high score (max 2.0 when both distances are ~0).
    """

    robot: Articulation = env.scene[robot_cfg.name]
    target = env.scene[target_asset_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    left_pos = robot.data.body_link_pos_w[:, left_idx]
    right_pos = robot.data.body_link_pos_w[:, right_idx]
    target_pos = target.data.root_pos_w
    d_left = torch.norm(left_pos - target_pos, dim=-1)
    d_right = torch.norm(right_pos - target_pos, dim=-1)
    s = float(sigma)
    return torch.exp(-d_left / s) + torch.exp(-d_right / s)


def wrists_clearance_above_surface_exp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    surface_z: float,
    min_clearance_m: float = 0.10,
    sigma_m: float = 0.055,
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
) -> torch.Tensor:
    """Encourage **both** wrist links to sit above the work surface before reaching down to the peg.

    Uses world-frame Z of each wrist link. ``surface_z`` is the tabletop height; ``min_clearance_m`` is how
    far above that plane each wrist should ideally be (meters). Deficits are penalized with a soft exponential.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    z_l = robot.data.body_link_pos_w[:, left_idx][:, 2]
    z_r = robot.data.body_link_pos_w[:, right_idx][:, 2]
    z_need = float(surface_z) + float(min_clearance_m)
    sig = float(sigma_m)
    deficit_l = torch.relu(z_need - z_l)
    deficit_r = torch.relu(z_need - z_r)
    return 0.5 * (torch.exp(-deficit_l / sig) + torch.exp(-deficit_r / sig))


class WristToInsertiveApproachProgress(ManagerTermBase):
    """Rewards per-step reductions in min wrist--peg distance (meters), clipped for stability."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.target_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        self.left = cfg.params.get("left_body_name", "arm_l_link7")
        self.right = cfg.params.get("right_body_name", "arm_r_link7")
        self.clip_m = float(cfg.params.get("clip_m", 0.012))
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        self.target = env.scene[self.target_cfg.name]
        self.prev_d = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)

    def _min_dist(self) -> torch.Tensor:
        left_idx = self.robot.find_bodies(self.left)[0][0]
        right_idx = self.robot.find_bodies(self.right)[0][0]
        left_pos = self.robot.data.body_link_pos_w[:, left_idx]
        right_pos = self.robot.data.body_link_pos_w[:, right_idx]
        peg = self.target.data.root_pos_w
        return torch.minimum(torch.norm(left_pos - peg, dim=-1), torch.norm(right_pos - peg, dim=-1))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        with torch.no_grad():
            d = self._min_dist()
            self.prev_d[env_ids] = d[env_ids].detach()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        left_body_name: str = "arm_l_link7",
        right_body_name: str = "arm_r_link7",
        clip_m: float = 0.012,
    ) -> torch.Tensor:
        d = self._min_dist()
        clip_m = float(clip_m)
        prog = torch.clamp((self.prev_d - d) / clip_m, 0.0, 1.0)
        self.prev_d = d.detach()
        return prog


class WristsToInsertiveApproachProgressSum(ManagerTermBase):
    """Per-step progress for **both** wrists toward the insertive peg (root frame).

    Rewards average of clipped reductions in left and right wrist distances (meters).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.target_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        self.left = cfg.params.get("left_body_name", "arm_l_link7")
        self.right = cfg.params.get("right_body_name", "arm_r_link7")
        self.clip_m = float(cfg.params.get("clip_m", 0.012))
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        self.target = env.scene[self.target_cfg.name]
        self.prev_left = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)
        self.prev_right = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)

    def _distances(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_idx = self.robot.find_bodies(self.left)[0][0]
        right_idx = self.robot.find_bodies(self.right)[0][0]
        left_pos = self.robot.data.body_link_pos_w[:, left_idx]
        right_pos = self.robot.data.body_link_pos_w[:, right_idx]
        peg = self.target.data.root_pos_w
        d_left = torch.norm(left_pos - peg, dim=-1)
        d_right = torch.norm(right_pos - peg, dim=-1)
        return d_left, d_right

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        with torch.no_grad():
            d_left, d_right = self._distances()
            self.prev_left[env_ids] = d_left[env_ids].detach()
            self.prev_right[env_ids] = d_right[env_ids].detach()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        target_asset_cfg: SceneEntityCfg,
        left_body_name: str = "arm_l_link7",
        right_body_name: str = "arm_r_link7",
        clip_m: float = 0.012,
    ) -> torch.Tensor:
        d_left, d_right = self._distances()
        clip_m = float(clip_m)
        prog_l = torch.clamp((self.prev_left - d_left) / clip_m, 0.0, 1.0)
        prog_r = torch.clamp((self.prev_right - d_right) / clip_m, 0.0, 1.0)
        self.prev_left = d_left.detach()
        self.prev_right = d_right.detach()
        return 0.5 * (prog_l + prog_r)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=1), 0, 1e4)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1), 0, 1e4
    )


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1), 0, 1e4)


class collision_free(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._env = env

        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)

    def __call__(self, env: ManagerBasedRLEnv, collision_analyzer_cfg: CollisionAnalyzerCfg) -> torch.Tensor:
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        return collision_free
