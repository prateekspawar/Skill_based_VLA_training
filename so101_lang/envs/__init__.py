"""Isaac Lab task registrations. Import this module (inside a running Isaac Sim app) to register them."""

import gymnasium as gym

gym.register(
    id="SO101-PickPlace-Lang-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:SO101PickPlaceIkEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="SO101-PickPlace-Lang-Joint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:SO101PickPlaceJointEnvCfg"},
    disable_env_checker=True,
)
