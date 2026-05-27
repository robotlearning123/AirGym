# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible configuration for the Hovering environment.

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from airgym import AIRGYM_ROOT_DIR


@configclass
class HoveringSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the Hovering environment."""
    pass


@configclass
class HoveringIsaacLabCfg(DirectRLEnvCfg):
    """IsaacLab-compatible configuration for the Hovering environment.

    This replaces the old HoveringCfg + sim_params pattern with IsaacLab's
    DirectRLEnvCfg which uses configclass and SimulationCfg.
    """

    # Environment settings
    episode_length_s: float = 24.0
    decimation: int = 1
    num_observations: int = 18
    num_actions: int = 4
    get_privileged_obs: bool = True
    env_spacing: float = 1.0
    num_control_steps_per_env_step: int = 1
    reset_on_collision: bool = False
    create_ground_plane: bool = False
    ctl_mode: str | None = None

    # Target state (identity rotation + zero position/velocity)
    target_state: list = None

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=1.0, replicate_physics=True, clone_in_fabric=True
    )

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Robot asset
    robot: ArticulationCfg = MISSING

    # Viewer camera
    viewer: dict = None

    def __post_init__(self):
        super().__post_init__()
        if self.target_state is None:
            self.target_state = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if self.viewer is None:
            self.viewer = {"ref_env": 0, "pos": [-5, -5, 4], "lookat": [0, 0, 0]}
