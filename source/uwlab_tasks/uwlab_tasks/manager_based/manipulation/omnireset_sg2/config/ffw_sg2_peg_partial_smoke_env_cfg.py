# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""FFW-SG2 in-scene with peg + peghole (OmniReset partial-assembly stack, smoke milestone)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.partial_assemblies_cfg import (
    PartialAssembliesActionsCfg,
    PartialAssembliesEventCfg,
    PartialAssembliesObservationsCfg,
    PartialAssembliesRewardsCfg,
    PartialAssembliesTerminationCfg,
)

from .ffw_sg2_articulation_cfg import FFW_SG2_CFG

OBJECT_SPAWN_HEIGHT = 0.5


@configclass
class FfwSg2PegPartialAssemblySceneCfg(InteractiveSceneCfg):
    """Scene: FFW-SG2 + peg + peg-hole + lighting (OmniReset-style props)."""

    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT * 2), rot=(1.0, 0.0, 0.0, 0.0)),
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
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
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
class FfwSg2PegPartialAssemblySmokeEnvCfg(ManagerBasedRLEnvCfg):
    """OmniReset partial-assembly events/rewards with SG2 present (no SG2 actions yet)."""

    scene: FfwSg2PegPartialAssemblySceneCfg = FfwSg2PegPartialAssemblySceneCfg(num_envs=1, env_spacing=2.0)
    events: PartialAssembliesEventCfg = PartialAssembliesEventCfg()
    terminations: PartialAssembliesTerminationCfg = PartialAssembliesTerminationCfg()
    observations: PartialAssembliesObservationsCfg = PartialAssembliesObservationsCfg()
    actions: PartialAssembliesActionsCfg = PartialAssembliesActionsCfg()
    rewards: PartialAssembliesRewardsCfg = PartialAssembliesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="receptive_object"
    )

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 4.0
        self.sim.dt = 1 / 120.0
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005
