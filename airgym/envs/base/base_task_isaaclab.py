# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Migration adapter: wraps IsaacLab DirectRLEnv to provide
# an API compatible with the original AirGym BaseTask interface.

from __future__ import annotations

import torch
import numpy as np
from typing import Any

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


class BaseTaskIsaacLab(DirectRLEnv):
    """IsaacLab-based base task that provides backward-compatible API with original AirGym BaseTask.

    This class adapts IsaacLab's DirectRLEnv to match the original AirGym interface:
    - obs_buf, rew_buf, reset_buf, time_out_buf
    - root_states, root_positions, root_quats, root_linvels, root_angvels
    - pre_physics_step / post_physics_step pattern
    - compute_observations / compute_reward pattern
    """

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Allocate buffers matching original AirGym interface
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        # Will be populated by subclass after scene setup
        self.root_states = None
        self.root_positions = None
        self.root_quats = None
        self.root_linvels = None
        self.root_angvels = None
        self.initial_root_states = None

    def _setup_scene(self):
        """Setup the scene. Subclasses should override to add custom assets."""
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-physics step. Subclasses should override."""
        pass

    def _apply_action(self):
        """Apply actions to the simulation. Subclasses should override."""
        pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done signals. Subclasses should override."""
        terminated = self.reset_buf.bool()
        time_outs = self.progress_buf > self.max_episode_length
        return terminated, time_outs

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards. Subclasses should override."""
        return self.rew_buf

    def _get_observations(self) -> dict:
        """Compute observations. Subclasses should override."""
        return {"policy": self.obs_buf}

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments. Subclasses should override."""
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    # ------------------------------------------------------------------
    # Backward-compatible methods from original AirGym BaseTask
    # ------------------------------------------------------------------

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return getattr(self, 'privileged_obs_buf', None)

    def reset(self, seed=None, options=None):
        """Override reset to match AirGym interface."""
        obs, extras = super().reset(seed=seed, options=options)
        return obs.get("policy", self.obs_buf), self.get_privileged_observations()
