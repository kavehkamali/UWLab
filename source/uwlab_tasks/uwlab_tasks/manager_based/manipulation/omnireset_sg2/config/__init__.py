# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="OmniReset-FFWSG2-SpawnSmoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.ffw_sg2_spawn_env_cfg:FfwSg2SpawnEnvCfg"},
    disable_env_checker=True,
)



gym.register(
    id="OmniReset-FFWSG2-PegPartialAssemblySmoke-ApproachLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblySmokeApproachLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_approach_lift.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-FFWSG2-PegApproachOnly-A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblyApproachOnlyAEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_approach_only_a.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-FFWSG2-PegApproachOnly-B-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblyApproachOnlyBEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_approach_only_b.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-FFWSG2-PegApproachOnly-B-InferCkpt256-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblyApproachOnlyBEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_approach_only_b_ckpt256.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblySmokeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
