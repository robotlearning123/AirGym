# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible Balloon environment for AirGym.

from __future__ import annotations

import torch
import numpy as np
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz

from airgym.envs.task.balloon_config_isaaclab import BalloonIsaacLabCfg


def _compute_yaw_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = b - a
    diff = torch.where(diff < -torch.pi, diff + 2 * torch.pi, diff)
    diff = torch.where(diff > torch.pi, diff - 2 * torch.pi, diff)
    return diff


class BalloonIsaacLab(DirectRLEnv):
    """IsaacLab-compatible Balloon environment for AirGym.

    This environment implements balloon popping task with:
    - Moving balloon target
    - Guidance reward
    - Yaw alignment
    - Hit detection
    """

    cfg: BalloonIsaacLabCfg

    def __init__(self, cfg: BalloonIsaacLabCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        assert cfg.ctl_mode is not None, "Please specify one control mode!"
        self.ctl_mode = cfg.ctl_mode
        self.num_actions = 5 if cfg.ctl_mode == "atti" else 4

        self._obs_tensor = torch.zeros(self.num_envs, cfg.num_observations, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}

        self.cmd_thrusts = torch.zeros((self.num_envs, 4), device=self.device)

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

        self.forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self.torques = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self.thrusts = torch.zeros((self.num_envs, 4, 3), dtype=torch.float32, device=self.device)

        self.target_states = torch.tensor(cfg.target_state, device=self.device).repeat(self.num_envs, 1)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.collisions = torch.zeros(self.num_envs, device=self.device)

        # Balloon state (virtual target)
        self.balloon_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_root_positions = torch.zeros((self.num_envs, 3), device=self.device)

        self._body_id = self._robot.find_bodies("base_link")[0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.to(self.device)

        if self.ctl_mode == 'rate' or self.ctl_mode == 'atti':
            self.actions[..., -1] = 0.5 + 0.5 * self.actions[..., -1]

        self.actions = torch.clamp(self.actions, self.action_lower_limits, self.action_upper_limits)

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
            self.cmd_thrusts = torch.tensor(self.parallel_pos_control.update(actions_cpu.astype(np.float64)), device=self.device)
        elif self.ctl_mode == "vel":
            self.cmd_thrusts = torch.tensor(self.parallel_vel_control.update(actions_cpu.astype(np.float64)), device=self.device)
        elif self.ctl_mode == "atti":
            self.cmd_thrusts = torch.tensor(self.parallel_atti_control.update(actions_cpu.astype(np.float64)), device=self.device)
        elif self.ctl_mode == "rate":
            self.cmd_thrusts = torch.tensor(self.parallel_rate_control.update(actions_cpu.astype(np.float64), ang_vel_cpu.astype(np.float64), 0.01), device=self.device)
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
        self._robot.set_external_force_and_torque(self.forces, self.torques, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        rot_matrix = self._quat_to_rot_matrix(root_quat_wxyz)

        self._obs_tensor[..., 0:9] = rot_matrix.reshape(self.num_envs, 9)
        self._obs_tensor[..., 9:12] = root_pos
        self._obs_tensor[..., 12:15] = root_linvel
        self._obs_tensor[..., 15:18] = root_angvel

        # Subtract balloon state (relative observations)
        self._obs_tensor[..., 9:12] -= self.balloon_positions

        return {"policy": self._obs_tensor}

    def _quat_to_rot_matrix(self, quat_wxyz):
        w, x, y, z = quat_wxyz.unbind(-1)
        return torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)

    def _matrix_to_euler_xyz(self, rot_matrix: torch.Tensor) -> torch.Tensor:
        sy = torch.sqrt(rot_matrix[:, 0, 0] ** 2 + rot_matrix[:, 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])
        y = torch.atan2(-rot_matrix[:, 2, 0], sy)
        z = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])

        x_s = torch.atan2(-rot_matrix[:, 1, 2], rot_matrix[:, 1, 1])
        y_s = torch.atan2(-rot_matrix[:, 2, 0], sy)
        z_s = torch.zeros_like(z)

        x = torch.where(singular, x_s, x)
        y = torch.where(singular, y_s, y)
        z = torch.where(singular, z_s, z)

        return torch.stack([x, y, z], dim=-1)

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    def _get_rewards(self) -> torch.Tensor:
        self.rew_buf[:], self._terminated_buf[:], self.item_reward_info = self._compute_balloon_reward()
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self._robot.data.root_pos_w.torch.clone()
        return self.rew_buf

    def _compute_balloon_reward(self):
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch

        relative_positions = self.balloon_positions - root_pos

        # Guidance reward
        guidance_reward = 30 * (torch.norm(self.balloon_positions - self.pre_root_positions, dim=-1) -
                                torch.norm(self.balloon_positions - root_pos, dim=-1))

        # Yaw reward
        direction_vector = F.normalize(relative_positions, dim=-1)
        direction_yaw = torch.atan2(direction_vector[..., 1], direction_vector[..., 0])
        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        rot_matrix = self._quat_to_rot_matrix(root_quat_wxyz)
        root_euler = self._matrix_to_euler_xyz(rot_matrix)
        relative_heading = _compute_yaw_diff(root_euler[..., 2], direction_yaw)
        yaw_distance = torch.norm(relative_heading.unsqueeze(-1), dim=1)
        yaw_reward = 1.0 / (1.0 + torch.square(1.6 * yaw_distance))

        # Uprightness reward
        ups = self._quat_rotate(root_quat, torch.tensor([0, 0, 1.0], device=self.device).expand(self.num_envs, -1))
        ups_reward = 0.5 * torch.square((ups[..., 2] + 1) / 2)

        # Hit reward
        check = torch.norm(relative_positions, dim=-1)
        hit_reward = 800 * torch.where(check < 0.1, torch.ones_like(check), torch.zeros_like(check))

        # Effort reward
        effort_reward = 0.1 * torch.exp(-self.actions.pow(2).sum(-1))

        # Action smoothness reward
        action_diff = torch.norm(self.actions - self.pre_actions, dim=-1)
        action_smoothness_reward = 0.1 * torch.exp(-action_diff)

        reward = (
            guidance_reward
            + yaw_reward
            + hit_reward
            + action_smoothness_reward
            + ups_reward
            + effort_reward
        )

        # Reset conditions
        ones = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        die = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        reset = torch.where(self.actions[..., -1] < -1, ones, die)
        reset = torch.where(self.actions[..., -1] > 1, ones, reset)
        reset = torch.where(relative_positions[..., 0] < -0.2, ones, reset)
        reset = torch.where(root_linvel[..., 0] < 0, ones, reset)
        reset = torch.where(torch.norm(relative_positions, dim=1) > 4, ones, reset)
        reset = torch.where(root_pos[..., 2] < 0.5, ones, reset)
        reset = torch.where(root_pos[..., 2] > 1.5, ones, reset)
        reset = torch.where(check < 0.1, ones, reset)

        item_reward_info = {
            "guidance_reward": guidance_reward,
            "hit_reward": hit_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "effort_reward": effort_reward,
            "ups_reward": ups_reward,
            "yaw_reward": yaw_reward,
            "reward": reward,
        }

        return reward, reset, item_reward_info

    def _get_dones(self):
        terminated = self._terminated_buf.bool()
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # Randomize balloon position
        self.balloon_positions[env_ids, 0:1] = 0.5 * (torch.rand(num_resets, 1, device=self.device) * 2 - 1) + 2.5
        self.balloon_positions[env_ids, 1:2] = 2.0 * (torch.rand(num_resets, 1, device=self.device) * 2 - 1)
        self.balloon_positions[env_ids, 2:3] = 0.3 * (torch.rand(num_resets, 1, device=self.device) * 2 - 1) + 1.0

        # Reset robot state
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]

        default_root_pose[:, 0:2] = 0.1 * (torch.rand(num_resets, 2, device=self.device) * 2 - 1)
        default_root_pose[:, 2:3] = 0.2 * (torch.rand(num_resets, 1, device=self.device) * 2 - 1) + 1.0

        root_angle = torch.stack([
            0.1 * (torch.rand(num_resets, device=self.device) * 2 * torch.pi - torch.pi),
            0.1 * (torch.rand(num_resets, device=self.device) * torch.pi),
            0.2 * (torch.rand(num_resets, device=self.device) * 2 * torch.pi - torch.pi)
        ], dim=-1)
        root_quat = quat_from_euler_xyz(root_angle[:, 0], root_angle[:, 1], root_angle[:, 2])
        default_root_pose[:, 3:7] = root_quat

        default_root_vel[:, 0:3] = 0.5 * (torch.rand(num_resets, 3, device=self.device) * 2 - 1)
        default_root_vel[:, 3:6] = 0.2 * (torch.rand(num_resets, 3, device=self.device) * 2 - 1)

        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

        self.actions[env_ids] = 0
        self.pre_actions[env_ids] = 0
        self.cmd_thrusts[env_ids] = 0
        self.pre_root_positions[env_ids] = 0
