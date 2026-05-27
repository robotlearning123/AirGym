# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible Tracking environment for AirGym.

from __future__ import annotations

import torch
import numpy as np
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from airgym.envs.task.tracking_config_isaaclab import TrackingIsaacLabCfg


class TrackingIsaacLab(DirectRLEnv):
    """IsaacLab-compatible Tracking environment for AirGym.

    This environment implements trajectory tracking with:
    - Moving target tracking
    - Smooth trajectory following
    """

    cfg: TrackingIsaacLabCfg

    def __init__(self, cfg: TrackingIsaacLabCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Control mode
        assert cfg.ctl_mode is not None, "Please specify one control mode!"
        self.ctl_mode = cfg.ctl_mode
        self.num_actions = 5 if cfg.ctl_mode == "atti" else 4

        # Allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, cfg.num_observations, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        # Controller
        self.cmd_thrusts = torch.zeros((self.num_envs, 4), device=self.device)

        # Set action limits based on control mode
        if cfg.ctl_mode == "pos":
            self.action_upper_limits = torch.tensor([3, 3, 3, 6.0], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor([-3, -3, -3, -6.0], device=self.device, dtype=torch.float32)
        elif cfg.ctl_mode == "vel":
            self.action_upper_limits = torch.tensor([6, 6, 6, 6], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor([-6, -6, -6, -6], device=self.device, dtype=torch.float32)
        elif cfg.ctl_mode == "atti":
            self.action_upper_limits = torch.tensor([1, 1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor([-1, -1, -1, -1, 0.], device=self.device, dtype=torch.float32)
        elif cfg.ctl_mode == "rate":
            self.action_upper_limits = torch.tensor([6, 6, 6, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor([-6, -6, -6, 0], device=self.device, dtype=torch.float32)
        elif cfg.ctl_mode == "prop":
            self.action_upper_limits = torch.tensor([1, 1, 1, 1], device=self.device, dtype=torch.float32)
            self.action_lower_limits = torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown control mode: {cfg.ctl_mode}")

        # Forces and torques
        self.forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self.torques = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)

        # Control parameters
        self.thrusts = torch.zeros((self.num_envs, 4, 3), dtype=torch.float32, device=self.device)

        # Target states
        self.target_states = torch.tensor(cfg.target_state, device=self.device).repeat(self.num_envs, 1)

        # Actions
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # Get robot body indices
        self._body_id = self._robot.find_bodies("base_link")[0]

    def _setup_scene(self):
        """Setup the scene with robot."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = actions.to(self.device)

        if self.ctl_mode == 'rate' or self.ctl_mode == 'atti':
            self.actions[..., -1] = 0.5 + 0.5 * self.actions[..., -1]

        self.actions = torch.clamp(self.actions, self.action_lower_limits, self.action_upper_limits)

        # Get robot state
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]

        actions_cpu = self.actions.cpu().numpy()
        root_pos_cpu = root_pos.cpu().numpy()
        root_quat_cpu = root_quat_wxyz.cpu().numpy()
        lin_vel_cpu = root_linvel.cpu().numpy()
        ang_vel_cpu = root_angvel.cpu().numpy()

        if self.ctl_mode == "pos":
            self.cmd_thrusts = torch.tensor(
                self.parallel_pos_control.update(actions_cpu.astype(np.float64)),
                device=self.device
            )
        elif self.ctl_mode == "vel":
            self.cmd_thrusts = torch.tensor(
                self.parallel_vel_control.update(actions_cpu.astype(np.float64)),
                device=self.device
            )
        elif self.ctl_mode == "atti":
            self.cmd_thrusts = torch.tensor(
                self.parallel_atti_control.update(actions_cpu.astype(np.float64)),
                device=self.device
            )
        elif self.ctl_mode == "rate":
            self.cmd_thrusts = torch.tensor(
                self.parallel_rate_control.update(actions_cpu.astype(np.float64), ang_vel_cpu.astype(np.float64), 0.01),
                device=self.device
            )
        elif self.ctl_mode == "prop":
            self.cmd_thrusts = self.actions

        delta = 9.59
        thrusts = (self.cmd_thrusts.to(self.device) * delta)

        force_xy = torch.zeros(self.num_envs, 4, 2, device=self.device)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        self.thrusts = thrusts

        prop_rot = (self.cmd_thrusts * 0.2).to(self.device)
        self.torques[:, 0, 0] = -prop_rot[:, 0]
        self.torques[:, 0, 1] = -prop_rot[:, 1]
        self.torques[:, 0, 2] = prop_rot[:, 2] + prop_rot[:, 3]

        self.forces[:, 0, :] = self.thrusts.sum(dim=1)

    def _apply_action(self):
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            self.forces, self.torques, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        """Compute observations."""
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        rot_matrix = self._quat_to_rot_matrix(root_quat_wxyz)

        self.obs_buf[..., 0:9] = rot_matrix.reshape(self.num_envs, 9)
        self.obs_buf[..., 9:12] = root_pos
        self.obs_buf[..., 12:15] = root_linvel
        self.obs_buf[..., 15:18] = root_angvel

        self.obs_buf[..., 0:18] -= self.target_states

        return {"policy": self.obs_buf}

    def _quat_to_rot_matrix(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (wxyz) to rotation matrix."""
        w, x, y, z = quat_wxyz.unbind(-1)
        rot_matrix = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)
        return rot_matrix

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        self.rew_buf[:], self.reset_buf[:], self.item_reward_info = self._compute_tracking_reward()
        self.pre_actions = self.actions.clone()
        return self.rew_buf

    def _compute_tracking_reward(self):
        """Compute tracking reward."""
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        target_positions = self.target_states[..., 9:12]
        relative_positions = target_positions - root_pos
        pos_diff = torch.norm(relative_positions, dim=-1)
        pos_reward = 0.7 / (1.0 + torch.square(1.6 * pos_diff))

        tar_direction = relative_positions / torch.norm(relative_positions, dim=1, keepdim=True)
        vel_direction = root_linvel / torch.norm(root_linvel, dim=1, keepdim=True)
        dot_product = (tar_direction * vel_direction).sum(dim=1)
        angle_diff = torch.acos(dot_product.clamp(-1.0, 1.0)).abs()
        vel_direction_reward = 0.1 * torch.exp(-angle_diff / torch.pi)

        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        ups = self._quat_rotate(root_quat, torch.tensor([0, 0, 1.0], device=self.device).expand(self.num_envs, -1))
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        spinnage = torch.square(root_angvel[:, -1])
        spin_reward = 1.0 / (1.0 + torch.square(3 * spinnage))

        effort_reward = 0.1 * (1 - torch.clamp(self.cmd_thrusts, min=0.0, max=1.0).sum(-1) / 4)

        action_diff = self.actions - self.pre_actions
        thrust_reward = torch.zeros(self.num_envs, device=self.device)
        if self.ctl_mode == "pos" or self.ctl_mode == 'vel' or self.ctl_mode == 'prop':
            continous_action_reward = 0.2 * torch.exp(-torch.norm(action_diff[..., :], dim=-1))
        else:
            continous_action_reward = 0.2 * torch.exp(-torch.norm(action_diff[..., :-1], dim=-1)) + 0.5 / (1.0 + torch.square(3 * action_diff[..., -1]))
            thrust = self.actions[..., -1]
            thrust_reward = 0.1 * (1 - torch.abs(0.1533 - thrust))

        if self.ctl_mode == "pos" or self.ctl_mode == 'vel' or self.ctl_mode == 'prop':
            reward = (
                continous_action_reward
                + effort_reward
                + pos_reward
                + pos_reward * (vel_direction_reward + ups_reward + spin_reward)
            )
        else:
            reward = (
                continous_action_reward
                + effort_reward
                + thrust_reward
                + pos_reward
                + pos_reward * (vel_direction_reward + ups_reward + spin_reward)
            )

        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        reset = torch.where(self.episode_length_buf >= self.max_episode_length - 1, ones, die)
        reset = torch.where(pos_diff > 4, ones, reset)
        reset = torch.where(relative_positions[..., 2] < -2, ones, reset)
        reset = torch.where(relative_positions[..., 2] > 2, ones, reset)
        reset = torch.where(ups[..., 2] < 0.0, ones, reset)

        item_reward_info = {
            "continous_action_reward": continous_action_reward,
            "effort_reward": effort_reward,
            "thrust_reward": thrust_reward if self.ctl_mode == "atti" or self.ctl_mode == 'rate' else 0,
            "pos_reward": pos_reward,
            "vel_direction_reward": vel_direction_reward,
            "ups_reward": ups_reward,
            "spin_reward": spin_reward,
            "reward": reward,
        }

        return reward, reset, item_reward_info

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q (xyzw format)."""
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done signals."""
        terminated = self.reset_buf.bool()
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)

        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]

        default_root_pose[:, 0:2] = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        default_root_pose[:, 2:3] = torch.rand(len(env_ids), 1, device=self.device) * 2 - 1

        root_angle = torch.stack([
            0.01 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi),
            0.01 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi),
            0.05 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi)
        ], dim=-1)

        root_quat = quat_from_euler_xyz(root_angle[:, 0], root_angle[:, 1], root_angle[:, 2])
        default_root_pose[:, 3:7] = root_quat

        default_root_vel[:, 0:3] = 0.5 * (torch.rand(len(env_ids), 3, device=self.device) * 2 - 1)
        default_root_vel[:, 3:6] = 0.2 * (torch.rand(len(env_ids), 3, device=self.device) * 2 - 1)

        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

        self.actions[env_ids] = 0
        self.pre_actions[env_ids] = 0
        self.cmd_thrusts[env_ids] = 0
