# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Migration adapter: IsaacLab-compatible config classes for AirGym.

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class AirGymSceneCfg(InteractiveSceneCfg):
    """Base scene configuration for AirGym environments."""
    pass


@configclass
class AirGymBaseCfg(DirectRLEnvCfg):
    """Base configuration for AirGym environments using IsaacLab.

    This replaces the old BaseConfig + sim_params pattern with IsaacLab's
    DirectRLEnvCfg which uses configclass and SimulationCfg.
    """

    # Environment settings
    num_observations: int = 18
    num_actions: int = 4
    get_privileged_obs: bool = True
    env_spacing: float = 1.0
    episode_length_s: float = 24.0
    num_control_steps_per_env_step: int = 1
    reset_on_collision: bool = False
    create_ground_plane: bool = False
    ctl_mode: str | None = None

    # Target state (default: identity rotation + zero position/velocity)
    target_state: list[float] = None

    # Scene
    scene: AirGymSceneCfg = AirGymSceneCfg(num_envs=256, env_spacing=1.0)

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=0.01, gravity=(0.0, 0.0, -9.81))

    # Viewer
    viewer: Any = None

    # Asset configuration
    asset_config: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.target_state is None:
            # Default: identity rotation (3x3 flattened) + zero pos/vel (9+9=18)
            self.target_state = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
