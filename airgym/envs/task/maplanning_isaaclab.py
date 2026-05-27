# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible MAPlanning (Multi-Agent Planning) environment for AirGym.

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz

from airgym.envs.task.maplanning_config_isaaclab import MAPlanningIsaacLabCfg


LENGTH = 8.0
WIDTH = 4.0
FLY_HEIGHT = 1.5


def _compute_yaw_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = b - a
    diff = torch.where(diff < -torch.pi, diff + 2 * torch.pi, diff)
    diff = torch.where(diff > torch.pi, diff - 2 * torch.pi, diff)
    return diff


class MAPlanningIsaacLab(DirectRLEnv):
    """IsaacLab-compatible Multi-Agent Planning environment for AirGym.

    This environment implements multi-robot path planning with:
    - Multiple robots per environment
    - Inter-agent observations
    - Goal-reaching with ESDF-based obstacle avoidance
    - Per-robot PID controllers
    """

    cfg: MAPlanningIsaacLabCfg

    def __init__(self, cfg: MAPlanningIsaacLabCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        assert cfg.ctl_mode is not None, "Please specify one control mode!"
        self.ctl_mode = cfg.ctl_mode
        self.num_actions = 5 if cfg.ctl_mode == "atti" else 4
        self.num_robots = cfg.num_robots

        # Multi-agent buffers: (num_envs, num_robots, ...)
        self.obs_buf = torch.zeros(self.num_envs, self.num_robots, cfg.num_observations, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, self.num_robots, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_robot = torch.zeros(self.num_envs, self.num_robots, device=self.device, dtype=torch.long)
        self.extras = {}

        self.cmd_thrusts = torch.zeros((self.num_envs, self.num_robots, 4), device=self.device)

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

        self.forces = torch.zeros((self.num_envs, self.num_robots, 3), dtype=torch.float32, device=self.device)
        self.torques = torch.zeros((self.num_envs, self.num_robots, 3), dtype=torch.float32, device=self.device)
        self.thrusts = torch.zeros((self.num_envs, self.num_robots, 4, 3), dtype=torch.float32, device=self.device)

        self.target_states = torch.tensor(cfg.target_state, device=self.device).view(1, 1, -1).expand(self.num_envs, self.num_robots, -1)

        self.actions = torch.zeros((self.num_envs, self.num_robots, self.num_actions), device=self.device)
        self.pre_actions = torch.zeros((self.num_envs, self.num_robots, self.num_actions), device=self.device)

        self.collisions = torch.zeros(self.num_envs, self.num_robots, device=self.device)
        self.all_collisions = torch.zeros(self.num_envs, device=self.device)
        self.counter = 0

        # Goal state
        self.goal_positions = torch.zeros((self.num_envs, 1, 3), device=self.device)

        # Previous state for reward computation
        self.pre_root_positions = torch.zeros((self.num_envs, self.num_robots, 3), device=self.device)
        self.prev_related_dist = torch.zeros(self.num_envs, device=self.device)
        self.related_dist = torch.zeros(self.num_envs, device=self.device)

        # World to local rotation matrices
        self.world_to_local = torch.zeros((self.num_envs, self.num_robots, 3, 3), device=self.device)

        # ESDF distance from depth cameras
        self.esdf_dist = torch.ones((self.num_envs, self.num_robots, 1), device=self.device) * 10

        self._body_ids = []
        for i in range(self.num_robots):
            self._body_ids.append(self._robot.find_bodies("base_link")[0])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.counter += 1
        self.actions = actions.to(self.device)

        if self.ctl_mode == 'rate' or self.ctl_mode == 'atti':
            self.actions[..., -1] = 0.5 + 0.5 * self.actions[..., -1]

        self.actions = torch.clamp(self.actions, self.action_lower_limits, self.action_upper_limits)

        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        # Process each robot
        for i in range(self.num_robots):
            root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
            actions_cpu = self.actions[:, i].cpu().numpy()
            root_pos_cpu = root_pos.cpu().numpy()
            root_quat_cpu = root_quat_wxyz.cpu().numpy()
            lin_vel_cpu = root_linvel.cpu().numpy()
            ang_vel_cpu = root_angvel.cpu().numpy()

            if self.ctl_mode == "pos":
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_pos_control[i].update(actions_cpu.astype(np.float64)), device=self.device)
            elif self.ctl_mode == "vel":
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_vel_control[i].update(actions_cpu.astype(np.float64)), device=self.device)
            elif self.ctl_mode == "atti":
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_atti_control[i].update(actions_cpu.astype(np.float64)), device=self.device)
            elif self.ctl_mode == "rate":
                self.cmd_thrusts[:, i] = torch.tensor(self.parallel_rate_control[i].update(actions_cpu.astype(np.float64), ang_vel_cpu.astype(np.float64), 0.01), device=self.device)
            elif self.ctl_mode == "prop":
                self.cmd_thrusts[:, i] = self.actions[:, i]

        delta = 9.59
        thrusts = (self.cmd_thrusts.to(self.device) * delta)
        force_xy = torch.zeros(self.num_envs, self.num_robots, 4, 2, device=self.device)
        thrusts = thrusts.reshape(self.num_envs, self.num_robots, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 3)
        self.thrusts = thrusts

        prop_rot = (self.cmd_thrusts * 0.2).to(self.device)
        self.torques[:, :, 0] = -prop_rot[:, :, 0]
        self.torques[:, :, 1] = -prop_rot[:, :, 1]
        self.torques[:, :, 2] = prop_rot[:, :, 2] + prop_rot[:, :, 3]
        self.forces[:, :, :] = self.thrusts.sum(dim=2)

    def _apply_action(self):
        for i in range(self.num_robots):
            self._robot.set_external_force_and_torque(
                self.forces[:, i:i+1, :],
                self.torques[:, i:i+1, :],
                body_ids=self._body_ids[i]
            )

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        forward_global = self.goal_positions - root_pos

        q_global = root_quat[..., [3, 0, 1, 2]]
        rot_matrix_global = self._quat_to_rot_matrix_batch(q_global)

        yaw = torch.atan2(rot_matrix_global[..., 1, 0], rot_matrix_global[..., 0, 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        self.world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-2)

        rot_matrix_local = torch.matmul(self.world_to_local, rot_matrix_global)
        euler_angles_local = self._matrix_to_euler_xyz_batch(rot_matrix_local)

        pos_diff_local = torch.einsum("bnij,bnj->bni", self.world_to_local, forward_global)
        vel_local = torch.einsum("bnij,bnj->bni", self.world_to_local, root_linvel)
        ang_vel_local = torch.einsum("bnij,bnj->bni", self.world_to_local, root_angvel)

        goal_dir = pos_diff_local / torch.norm(pos_diff_local, dim=-1, keepdim=True)
        self.related_dist = torch.norm(forward_global, dim=-1)

        self.obs_buf[..., 0:3] = goal_dir
        self.obs_buf[..., 3:6] = euler_angles_local
        self.obs_buf[..., 6:9] = vel_local
        self.obs_buf[..., 9:12] = ang_vel_local
        self.obs_buf[..., 12:16] = self.actions

        # Inter-agent observations
        for idx in range(self.num_robots):
            pos_x = root_pos[:, idx, 0:1]
            vel_x = root_linvel[:, idx, 0:1]
            for j in range(self.num_robots):
                self.obs_buf[:, idx, 16+j*2:16+(j+1)*2-1] = root_pos[:, j, 0:1] - pos_x
                self.obs_buf[:, idx, 16+(j+1)*2-1:16+(j+1)*2] = root_linvel[:, j, 0:1] - vel_x

        self.obs_buf[..., 16:] = 0

        # Flatten for return
        flat_obs = self.obs_buf.reshape(-1, self.cfg.num_observations)
        return {"policy": flat_obs}

    def _quat_to_rot_matrix_batch(self, quat_wxyz):
        w, x, y, z = quat_wxyz.unbind(-1)
        return torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)

    def _matrix_to_euler_xyz_batch(self, rot_matrix: torch.Tensor) -> torch.Tensor:
        sy = torch.sqrt(rot_matrix[..., 0, 0] ** 2 + rot_matrix[..., 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(rot_matrix[..., 2, 1], rot_matrix[..., 2, 2])
        y = torch.atan2(-rot_matrix[..., 2, 0], sy)
        z = torch.atan2(rot_matrix[..., 1, 0], rot_matrix[..., 0, 0])

        x_s = torch.atan2(-rot_matrix[..., 1, 2], rot_matrix[..., 1, 1])
        y_s = torch.atan2(-rot_matrix[..., 2, 0], sy)
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
        self.rew_buf[:], self.reset_robot[:], self.reset_buf[:], self.item_reward_info = self._compute_reward()
        self.pre_actions = self.actions.clone()
        self.pre_root_positions = self._robot.data.root_pos_w.torch.clone()
        self.prev_related_dist = self.related_dist.clone()

        # Flatten for return
        flat_rew = self.rew_buf.reshape(-1)
        return flat_rew

    def _compute_reward(self):
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch

        # Continuous action reward
        action_diff = self.actions - self.pre_actions
        continous_action_reward = 0.2 * torch.norm(action_diff, dim=-1)
        thrust_reward = 0.5 * (1 - torch.abs(0.1533 - self.actions[..., -1]))

        # Guidance reward
        forward_reward = 0.1 * (torch.norm(self.goal_positions - self.pre_root_positions, dim=-1) -
                                torch.norm(self.goal_positions - root_pos, dim=-1))

        # Heading reward
        forward_vec = self.goal_positions - root_pos
        forward_vec = forward_vec / torch.norm(forward_vec, dim=-1, keepdim=True)
        heading_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 1, -1).expand(self.num_envs, self.num_robots, -1)
        heading_reward = torch.sum(forward_vec * heading_vec, dim=-1)

        # Speed reward
        vel_local = torch.einsum("bnij,bnj->bni", self.world_to_local, self._robot.data.root_lin_vel_w.torch)
        speed_reward = -0.5 * (1 - torch.exp(-2 * torch.square(vel_local[..., 0] - 1.0)))

        # Height reward
        z_reward = torch.min(torch.min(root_pos[..., 2] - (FLY_HEIGHT + 0.3), torch.tensor(0.0)),
                             (FLY_HEIGHT - 0.3) - root_pos[..., 2])

        # Uprightness reward
        ups = self._quat_rotate(root_quat.reshape(-1, 4), torch.tensor([0, 0, 1.0], device=self.device).expand(self.num_envs * self.num_robots, -1))
        ups = ups.view(self.num_envs, self.num_robots, 3)
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        # ESDF reward
        esdf_reward = 0.5 * (1 - torch.exp(-0.5 * torch.square(self.esdf_dist))).squeeze(-1)

        # Collision reward
        alive_reward = torch.where(self.esdf_dist > 0.3, torch.tensor(0.0), torch.tensor(-1.0)).squeeze(-1)

        # Reach goal
        reach_goal = self.related_dist < 0.3
        reach_goal_reward = torch.where(reach_goal, torch.tensor(200.0), torch.tensor(0.0))

        reward = (
            continous_action_reward
            + forward_reward
            + alive_reward + esdf_reward
            + ups_reward
            + z_reward
            + speed_reward
            + heading_reward
            + thrust_reward
            + reach_goal_reward
        )

        # Per-robot resets
        ones = torch.ones_like(self.reset_robot)
        die = torch.zeros_like(self.reset_robot)

        reset_robot = torch.where(root_pos[..., 2] > FLY_HEIGHT + 0.3, ones, die)
        reset_robot = torch.where(self.collisions > 0, ones, reset_robot)
        reset_robot = torch.where(reach_goal, ones, reset_robot)

        # Env reset
        reset_env = torch.any(reset_robot, dim=-1)
        reset_env = torch.where(self.episode_length_buf >= self.max_episode_length - 1,
                                torch.ones_like(self.reset_buf), reset_env)

        item_reward_info = {
            "continous_action_reward": continous_action_reward,
            "heading_reward": heading_reward,
            "speed_reward": speed_reward,
            "forward_reward": forward_reward,
            "alive_reward": alive_reward,
            "ups_reward": ups_reward,
            "z_reward": z_reward,
            "esdf_reward": esdf_reward,
            "thrust_reward": thrust_reward,
            "reach_goal_reward": reach_goal_reward,
            "reward": reward,
        }

        return reward, reset_robot, reset_env, item_reward_info

    def _get_dones(self):
        terminated = self.reset_buf.bool()
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # Randomize goal position
        self.goal_positions[env_ids, :, 0:1] = LENGTH + 0.5
        self.goal_positions[env_ids, :, 1:2] = 1.5 * (torch.rand(num_resets, 1, 1, device=self.device) * 2 - 1)
        self.goal_positions[env_ids, :, 2:3] = FLY_HEIGHT

        # Reset robot states
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]

        default_root_pose[:, 0:1] = -LENGTH - 0.5
        default_root_pose[:, 1:2] = 2.0 * (torch.rand(num_resets, 1, device=self.device) * 2 - 1)
        default_root_pose[:, 2:3] = FLY_HEIGHT

        # Compute initial yaw toward goal
        init_yaw = torch.atan2(
            self.goal_positions[env_ids, 0, 1] - default_root_pose[:, 1],
            self.goal_positions[env_ids, 0, 0] - default_root_pose[:, 0]
        )

        root_angle = torch.stack([
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            init_yaw
        ], dim=-1)
        root_quat = quat_from_euler_xyz(root_angle[:, 0], root_angle[:, 1], root_angle[:, 2])
        default_root_pose[:, 3:7] = root_quat

        default_root_vel[:, :] = 0

        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

        self.actions[env_ids] = 0
        self.pre_actions[env_ids] = 0
        self.cmd_thrusts[env_ids] = 0
        self.collisions[env_ids] = 0
        self.prev_related_dist[env_ids] = 0
        self.pre_root_positions[env_ids] = 0

        # Initialize world_to_local
        q_global = self._robot.data.root_quat_w.torch[env_ids][:, [3, 0, 1, 2]]
        rot_matrix_global = self._quat_to_rot_matrix_batch(q_global)
        yaw = torch.atan2(rot_matrix_global[..., 1, 0], rot_matrix_global[..., 0, 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        self.world_to_local[env_ids] = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-2)

        self.esdf_dist[env_ids] = 10.0
