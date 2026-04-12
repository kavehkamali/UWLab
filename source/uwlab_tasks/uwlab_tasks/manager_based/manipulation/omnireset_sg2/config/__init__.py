# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="OmniReset-FFWSG2-SpawnSmoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.ffw_sg2_spawn_env_cfg:FfwSg2SpawnEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.ffw_sg2_peg_partial_smoke_env_cfg:FfwSg2PegPartialAssemblySmokeEnvCfg"},
    disable_env_checker=True,
)

