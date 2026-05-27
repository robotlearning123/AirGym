# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible configuration for the MAPlanning environment.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab.envs.common import ViewerCfg

from airgym.assets.x152b_isaaclab import X152B_CFG


@configclass
class MAPlanningIsaacLabCfg(DirectRLEnvCfg):
    """IsaacLab-compatible configuration for the Multi-Agent Planning environment."""

    # Environment settings
    episode_length_s: float = 16.0
    decimation: int = 1
    num_observations: int = 24  # 16 + num_agents*2 (default 4 agents)
    num_actions: int = 4
    observation_space: int = 24
    action_space: int = 4
    get_privileged_obs: bool = True
    env_spacing: float = 14.0
    num_control_steps_per_env_step: int = 1
    reset_on_collision: bool = False
    create_ground_plane: bool = True
    ctl_mode: str | None = None

    # Camera settings
    cam_dt: float = 0.04
    enable_onboard_cameras: bool = True

    # Multi-agent settings
    num_robots: int = 4

    # Target state
    target_state: list = None

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=14.0, replicate_physics=True, clone_in_fabric=True
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
    robot: ArticulationCfg = X152B_CFG

    # Viewer camera
    viewer: ViewerCfg = ViewerCfg(eye=(-5, -5, 4), lookat=(0, 0, 0))

    def __post_init__(self):
        super().__post_init__()
        if self.target_state is None:
            self.target_state = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
