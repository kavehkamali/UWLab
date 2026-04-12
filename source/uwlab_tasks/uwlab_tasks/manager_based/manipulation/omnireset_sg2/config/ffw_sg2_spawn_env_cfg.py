# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal headless spawn + step smoke for ROBOTIS FFW-SG2 in Isaac Lab (Milestone 1)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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

from .ffw_sg2_articulation_cfg import FFW_SG2_CFG


@configclass
class FfwSg2SpawnSceneCfg(InteractiveSceneCfg):
    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
class FfwSg2SpawnObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FfwSg2SpawnActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.05,
        use_default_offset=True,
    )


@configclass
class FfwSg2SpawnRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class FfwSg2SpawnTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FfwSg2SpawnEnvCfg(ManagerBasedRLEnvCfg):
    scene: FfwSg2SpawnSceneCfg = FfwSg2SpawnSceneCfg(num_envs=1, env_spacing=4.0)
    observations: FfwSg2SpawnObservationsCfg = FfwSg2SpawnObservationsCfg()
    actions: FfwSg2SpawnActionsCfg = FfwSg2SpawnActionsCfg()
    rewards: FfwSg2SpawnRewardsCfg = FfwSg2SpawnRewardsCfg()
    terminations: FfwSg2SpawnTerminationsCfg = FfwSg2SpawnTerminationsCfg()
    commands = None
    curriculum = None
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.5, 2.5, 1.8),
        lookat=(0.0, 0.0, 0.5),
        origin_type="world",
        env_index=0,
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
