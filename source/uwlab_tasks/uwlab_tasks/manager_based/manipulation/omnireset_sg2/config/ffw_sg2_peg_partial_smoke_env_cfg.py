# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""FFW-SG2 bimanual peg-in-hole on a work surface.

**Start configuration (vetted layout)** — see also ``README_FFWSG2_PEG_SMOKE.md`` in this directory.

- Work surface + peg + peg-hole are shifted in **+X** by ``_SCENE_FORWARD_OFFSET_X`` (base clearance plus
  ``_ROBOT_HEAD_WIDTH_M``) so default arm poses clear the tabletop volume.
- **Peg** spawns on the table beside the hole; **peg-hole** is **dynamic** (``kinematic_enabled=False``),
  spawned slightly above the tabletop (``_RECEPTIVE_SPAWN_Z``) so it **settles under gravity**.
- **Table** cuboid remains **kinematic** (fixed). Whole-body joint targets use a **per-joint scale map**:
  stronger ``lift_joint`` and a higher default for any remaining DoFs (wheels, base, etc.) so the policy can
  reposition the base and torso Z while still using fine deltas on the arms.

The policy is rewarded for aligning the peg with the assembled pose (OmniReset ``ProgressContext`` /
success rewards). Joint-position actions cover **all** actuated joints with scaled deltas (arms, grippers,
  torso lift, wheels / base as named by the asset).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.collision_analyzer_cfg import CollisionAnalyzerCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands_cfg import TaskCommandCfg

from .ffw_sg2_articulation_cfg import FFW_SG2_CFG
from .ffw_sg2_tabletop_anchor_rewards import (
    WristsToWorldPointApproachProgressSum,
    wrists_min_distance_to_world_point_exp_sum,
)

# Workbench-style surface so peg / peg-hole read as resting on a table (not mid-air over the floor).
_TABLE_THICKNESS_M = 0.05
_TABLE_SURFACE_Z = 0.82
_TABLE_CENTER_Z = _TABLE_SURFACE_Z - _TABLE_THICKNESS_M / 2.0
# Shift the whole work surface + peg + fixture in +X so default arm poses sit in front of the
# tabletop cuboid instead of intersecting it at initialization.
# Extra clearance ≈ FFW-SG2 head width (order-of-magnitude for this platform).
_ROBOT_HEAD_WIDTH_M = 0.20
_SCENE_FORWARD_OFFSET_X = 0.30 + _ROBOT_HEAD_WIDTH_M
# Peg-hole fixture XY; peg spawn is offset on the table so it is not pre-inserted.
_PEG_HOLE_XY = (0.12 + _SCENE_FORWARD_OFFSET_X, 0.0)
# Geometric center of the tabletop **upper surface** (matches ``work_surface`` init XY + ``_TABLE_SURFACE_Z``).
_TABLETOP_TOP_CENTER_XYZ = (_PEG_HOLE_XY[0] + 0.03, _PEG_HOLE_XY[1], _TABLE_SURFACE_Z)
_PEG_TABLE_OFFSET_X = 0.10  # peg rests to the +X side of the hole on the tabletop
# Peg-hole starts above the tabletop and is dynamic so it settles under gravity (not kinematic).
_RECEPTIVE_SPAWN_Z = _TABLE_SURFACE_Z + 0.14
# Slight lift so the peg settles onto the table in the first frames instead of interpenetrating the cuboid.
_PEG_ON_TABLE_Z = _TABLE_SURFACE_Z + 0.025


@configclass
class FfwSg2PegPartialAssemblySceneCfg(InteractiveSceneCfg):
    """Scene: FFW-SG2 + work surface + peg (on table) + peg-hole + lighting."""

    # Explicit for large ``num_envs``: physics state is replicated across env clones (Isaac Lab default is True).
    replicate_physics: bool = True

    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    work_surface: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.75, _TABLE_THICKNESS_M),
            # Cloner: inherit geometry from env_0 across clones (fewer duplicated PhysX material bindings).
            copy_from_source=False,
            # Bind every env's tabletop collision to one USD physics material prim (PhysX material sharing).
            physics_material_path="/World/UWLAB_SharedMaterials/FFWSG2_TablePhys",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75,
                dynamic_friction=0.75,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0] + 0.03, _PEG_HOLE_XY[1], _TABLE_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            copy_from_source=False,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0] + _PEG_TABLE_OFFSET_X, _PEG_HOLE_XY[1], _PEG_ON_TABLE_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            copy_from_source=False,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _RECEPTIVE_SPAWN_Z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class FfwSg2PegInsertionCommandsCfg:
    """Static task command: supplies success thresholds from peg-hole USD metadata."""

    task_command = TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class FfwSg2PegInsertionRewardsCfg:
    """Insertion-style rewards: alignment / success from ``ProgressContext`` plus collision quality."""

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore[misc]
        weight=0.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.35,
        params={
            "std": 0.22,
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=1.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    collision_free = RewTerm(
        func=task_mdp.collision_free,
        params={
            "collision_analyzer_cfg": CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.001,
                asset_cfg=SceneEntityCfg("insertive_object"),
                obstacle_cfgs=[SceneEntityCfg("receptive_object")],
            )
        },
        weight=0.25,
    )

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionRewardsApproachLiftCfg:
    """Peg smoke rewards plus explicit approach / transport / lift / grasp-proxy shaping."""

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore[misc]
        weight=0.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    wrist_to_peg = RewTerm(
        func=task_mdp.wrist_min_distance_to_asset_exp,
        weight=0.45,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "sigma": 0.32,
        },
    )

    wrist_approach_progress = RewTerm(
        func=task_mdp.WristToInsertiveApproachProgress,  # type: ignore[misc]
        weight=0.55,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "clip_m": 0.012,
        },
    )

    peg_xy_to_hole = RewTerm(
        func=task_mdp.insertive_xy_near_receptor_tanh,
        weight=0.14,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
            "std": 0.16,
        },
    )

    peg_lift = RewTerm(
        func=task_mdp.insertive_height_above_surface,
        weight=0.20,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "surface_z": _TABLE_SURFACE_Z,
            "scale": 0.07,
        },
    )

    gripper_near_peg = RewTerm(
        func=task_mdp.gripper_excitation_near_insertive,
        weight=0.18,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper_l_joint1", "gripper_r_joint1"]),
            "proximity_std": 0.30,
            "joint_scale": 0.35,
        },
    )

    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.08,
        params={
            "std": 2.0,
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=1.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    collision_free = RewTerm(
        func=task_mdp.collision_free,
        params={
            "collision_analyzer_cfg": CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.001,
                asset_cfg=SceneEntityCfg("insertive_object"),
                obstacle_cfgs=[SceneEntityCfg("receptive_object")],
            )
        },
        weight=0.14,
    )

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionRewardsApproachOnlyCfgA:
    """Approach shaping **A** plus sparse/dense insertion rewards so the task objective stays aligned with assembly."""

    wrists_clearance_above_surface = RewTerm(
        func=task_mdp.wrists_clearance_above_surface_exp,
        weight=0.58,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "surface_z": _TABLE_SURFACE_Z,
            "min_clearance_m": 0.18,
            "sigma_m": 0.048,
        },
    )

    both_wrists_to_pin = RewTerm(
        func=wrists_min_distance_to_world_point_exp_sum,
        weight=0.38,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ,
            "sigma": 0.36,
        },
    )

    both_wrists_approach_progress = RewTerm(
        func=WristsToWorldPointApproachProgressSum,  # type: ignore[misc]
        weight=0.32,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ,
            "clip_m": 0.018,
        },
    )

    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.28,
        params={
            "std": 0.22,
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=1.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionRewardsApproachOnlyCfgB:
    """Approach shaping **B** plus sparse/dense insertion rewards (stronger tabletop pull than **A**)."""

    wrists_clearance_above_surface = RewTerm(
        func=task_mdp.wrists_clearance_above_surface_exp,
        weight=0.42,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "surface_z": _TABLE_SURFACE_Z,
            "min_clearance_m": 0.18,
            "sigma_m": 0.042,
        },
    )

    both_wrists_to_pin = RewTerm(
        func=wrists_min_distance_to_world_point_exp_sum,
        weight=0.52,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ,
            "sigma": 0.22,
        },
    )

    both_wrists_approach_progress = RewTerm(
        func=WristsToWorldPointApproachProgressSum,  # type: ignore[misc]
        weight=0.40,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ,
            "clip_m": 0.011,
        },
    )

    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.28,
        params={
            "std": 0.22,
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=1.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionTerminationsCfg:
    """Do not use ``obb_no_overlap`` here: it fires when the peg moves away from its reset pose, which is
    exactly what we want during insertion."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class FfwSg2PegInsertionEventCfg:
    """Intentionally minimal: peg pose comes from scene init (table), not ``assembly_sampling_event``."""

    pass


@configclass
class FfwSg2PegInsertionEventsCfgHighArmNoise:
    """Randomize arm joint offsets on reset (does not move base / head / lift by default)."""

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
            "position_range": (-0.18, 0.18),
            "velocity_range": (-0.08, 0.08),
        },
    )


@configclass
class FfwSg2PegInsertionEventsCfgApproachOnlyShared:
    """Shared resets for ApproachOnly A/B (hyperparameter tuning should not hinge on different spawn noise)."""

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.07, 0.07),
        },
    )

    # Bias the torso lift upward so wrists start with more clearance over the tabletop volume.
    reset_lift_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lift_joint"]),
            "position_range": (0.10, 0.24),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.004, 0.004),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.45, 0.45),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("insertive_object"),
        },
    )


@configclass
class FfwSg2PegInsertionEventsCfgSmokeArmAndPeg:
    """Arm jitter plus peg XY/yaw randomization for learnable dense reward."""

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
            "position_range": (-0.18, 0.18),
            "velocity_range": (-0.08, 0.08),
        },
    )

    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.06, 0.06),
                "z": (-0.004, 0.004),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.55, 0.55),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("insertive_object"),
        },
    )

# Split joint position actions so ``lift_joint`` never shares a scale dict with a ``.*`` catch-all (Hydra /
# OmegaConf can reorder dict keys and Isaac then raises ``Multiple matches for 'lift_joint'``).


@configclass
class FfwSg2PegPartialAssemblyActionsCfg:
    """Torso lift (coarse), then arms/grippers (fine), then all remaining DoFs (wheels, head, etc.)."""

    torso_lift = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["lift_joint"],
        scale=0.22,
        use_default_offset=True,
    )
    arms_grippers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint.*", "arm_r_joint.*", "gripper_l_joint.*", "gripper_r_joint.*"],
        scale=0.08,
        use_default_offset=True,
    )
    wheels_head_misc = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!lift_joint$)(?!arm_[lr]_joint)(?!gripper_[lr]_joint).+$"],
        scale=0.14,
        use_default_offset=True,
    )


@configclass
class FfwSg2PegPartialAssemblyObservationsCfg:
    """Robot state, previous action, and peg/hole poses for manipulation."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action)

        peg_in_hole_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )
        peg_in_robot = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )
        hole_in_robot = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FfwSg2PegPartialAssemblySmokeEnvCfg(ManagerBasedRLEnvCfg):
    """Bimanual-capable peg insertion on the work surface (peg starts on the table, not pre-assembled)."""

    scene: FfwSg2PegPartialAssemblySceneCfg = FfwSg2PegPartialAssemblySceneCfg(num_envs=1, env_spacing=2.0)
    commands: FfwSg2PegInsertionCommandsCfg = FfwSg2PegInsertionCommandsCfg()
    events: FfwSg2PegInsertionEventsCfgSmokeArmAndPeg = FfwSg2PegInsertionEventsCfgSmokeArmAndPeg()
    terminations: FfwSg2PegInsertionTerminationsCfg = FfwSg2PegInsertionTerminationsCfg()
    observations: FfwSg2PegPartialAssemblyObservationsCfg = FfwSg2PegPartialAssemblyObservationsCfg()
    actions: FfwSg2PegPartialAssemblyActionsCfg = FfwSg2PegPartialAssemblyActionsCfg()
    rewards: FfwSg2PegInsertionRewardsCfg = FfwSg2PegInsertionRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.55, 0.0, 1.10),
        lookat=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _TABLE_SURFACE_Z + 0.12),
        origin_type="world",
        env_index=0,
        asset_name="receptive_object",
        resolution=(960, 540),
    )

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120.0
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 1024 * 1024
        self.sim.render_interval = self.decimation


@configclass
class FfwSg2PegPartialAssemblySmokeApproachLiftEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Same peg scene with approach / XY / lift / grasp-proxy shaping and higher arm reset noise."""

    rewards: FfwSg2PegInsertionRewardsApproachLiftCfg = FfwSg2PegInsertionRewardsApproachLiftCfg()
    events: FfwSg2PegInsertionEventsCfgHighArmNoise = FfwSg2PegInsertionEventsCfgHighArmNoise()


@configclass
class FfwSg2PegPartialAssemblyApproachOnlyAEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Approach-to-table-center shaping **A**, insertion success/dense rewards, and full-body joint actions."""

    rewards: FfwSg2PegInsertionRewardsApproachOnlyCfgA = FfwSg2PegInsertionRewardsApproachOnlyCfgA()
    events: FfwSg2PegInsertionEventsCfgApproachOnlyShared = FfwSg2PegInsertionEventsCfgApproachOnlyShared()


@configclass
class FfwSg2PegPartialAssemblyApproachOnlyBEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Approach-to-table-center shaping **B**, insertion success/dense rewards, and full-body joint actions."""

    rewards: FfwSg2PegInsertionRewardsApproachOnlyCfgB = FfwSg2PegInsertionRewardsApproachOnlyCfgB()
    events: FfwSg2PegInsertionEventsCfgApproachOnlyShared = FfwSg2PegInsertionEventsCfgApproachOnlyShared()
