# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""FFW-SG2 bimanual peg-in-hole on a work surface.

**Start configuration (vetted layout)** — see also ``README_FFWSG2_PEG_SMOKE.md`` in this directory.

- Work surface + peg + peg-hole are shifted in **+X** by ``_SCENE_FORWARD_OFFSET_X`` (base clearance plus
  ``_ROBOT_HEAD_WIDTH_M``) so default arm poses clear the tabletop volume.
- **Peg** spawns on the table beside the hole; **peg-hole** is **dynamic** (``kinematic_enabled=False``),
  spawned slightly above the tabletop (``_RECEPTIVE_SPAWN_Z``) so it **settles under gravity**.
- **Table** cuboid remains **kinematic** (fixed). Peg uses small joint-scale actions via env ``__post_init__``.

The policy is rewarded for aligning the peg with the assembled pose (OmniReset ``ProgressContext`` /
success rewards). Both arms are controlled via a single joint-position action on all robot joints.
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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.collision_analyzer_cfg import CollisionAnalyzerCfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands_cfg import TaskCommandCfg

from .ffw_sg2_articulation_cfg import FFW_SG2_CFG

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
_PEG_TABLE_OFFSET_X = 0.10  # peg rests to the +X side of the hole on the tabletop
# Peg-hole starts above the tabletop and is dynamic so it settles under gravity (not kinematic).
_RECEPTIVE_SPAWN_Z = _TABLE_SURFACE_Z + 0.14
# Slight lift so the peg settles onto the table in the first frames instead of interpenetrating the cuboid.
_PEG_ON_TABLE_Z = _TABLE_SURFACE_Z + 0.025


@configclass
class FfwSg2PegPartialAssemblySceneCfg(InteractiveSceneCfg):
    """Scene: FFW-SG2 + work surface + peg (on table) + peg-hole + lighting."""

    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    work_surface: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.75, _TABLE_THICKNESS_M),
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

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.35, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)

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
class FfwSg2PegInsertionTerminationsCfg:
    """Do not use ``obb_no_overlap`` here: it fires when the peg moves away from its reset pose, which is
    exactly what we want during insertion."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class FfwSg2PegInsertionEventCfg:
    """Intentionally minimal: peg pose comes from scene init (table), not ``assembly_sampling_event``."""

    pass


@configclass
class FfwSg2PegPartialAssemblyActionsCfg:
    """Joint targets for all FFW-SG2 DoFs (both arms, grippers, lift, head)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.08,
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
    events: FfwSg2PegInsertionEventCfg = FfwSg2PegInsertionEventCfg()
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
