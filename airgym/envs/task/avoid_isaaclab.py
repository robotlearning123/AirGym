# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible Avoid environment for AirGym.

from __future__ import annotations

import torch
import numpy as np
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import convert_quat, quat_from_euler_xyz, quat_mul, quat_rotate

from airgym.envs.task.avoid_config_isaaclab import AvoidIsaacLabCfg


class AvoidIsaacLab(DirectRLEnv):
    """IsaacLab-compatible Avoid environment for AirGym.

    This environment implements obstacle avoidance with:
    - Random obstacle placement
    - Collision detection
    - Camera-based observations (optional)
    """

    cfg: AvoidIsaacLabCfg

    def __init__(self, cfg: AvoidIsaacLabCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Control mode
        assert cfg.ctl_mode is not None, "Please specify one control mode!"
        self.ctl_mode = cfg.ctl_mode
        self.num_actions = 5 if cfg.ctl_mode == "atti" else 4

        # Allocate buffers
        obs_size = 12 + self.num_actions
        self._obs_tensor = torch.zeros(self.num_envs, obs_size, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
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

        # Collision tracking
        self.collisions = torch.zeros(self.num_envs, device=self.device)

        # Get robot body indices
        self._body_id = self._robot.find_bodies("base_link")[0]

    def _setup_scene(self):
        """Setup the scene with robot, obstacles, and sensors."""
        # Create robot articulation
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Create obstacle if configured
        if self.cfg.obstacle is not None:
            self._obstacle = RigidObject(self.cfg.obstacle)
            self.scene.rigid_objects["obstacle"] = self._obstacle

        # Add contact sensor
        # Resolve {ENV_REGEX_NS} since the scene doesn't do this for ContactSensorCfg.prim_path
        env_regex_ns = self.scene.env_regex_ns
        contact_cfg = ContactSensorCfg(
            prim_path=f"{env_regex_ns}/Robot/Geometry/.*",
            history_length=3,
            track_air_time=True,
        )
        self._contact_sensor = ContactSensor(contact_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
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
        root_quat = self._robot.data.root_quat_w.torch  # xyzw format in IsaacLab
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        # Convert quaternion to numpy for controller (controller expects wxyz)
        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        actions_cpu = self.actions.cpu().numpy()
        root_pos_cpu = root_pos.cpu().numpy()
        root_quat_cpu = root_quat_wxyz.cpu().numpy()
        lin_vel_cpu = root_linvel.cpu().numpy()
        ang_vel_cpu = root_angvel.cpu().numpy()

        # Apply control based on mode
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

        # Compute thrust
        delta = 9.59
        thrusts = (self.cmd_thrusts.to(self.device) * delta)

        # Create force/torque vectors
        force_xy = torch.zeros(self.num_envs, 4, 2, device=self.device)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        self.thrusts = thrusts

        # Compute torques from propeller rotation
        prop_rot = (self.cmd_thrusts * 0.2).to(self.device)
        self.torques[:, 0, 0] = -prop_rot[:, 0]
        self.torques[:, 0, 1] = -prop_rot[:, 1]
        self.torques[:, 0, 2] = prop_rot[:, 2] + prop_rot[:, 3]

        # Sum forces from all 4 propellers
        self.forces[:, 0, :] = self.thrusts.sum(dim=1)

    def _apply_action(self):
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            self.forces, self.torques, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        """Compute observations."""
        # Get robot state
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch  # xyzw
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        # Convert to local frame
        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        rot_matrix = self._quat_to_rot_matrix(root_quat_wxyz)

        # Compute yaw
        yaw = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # World to local rotation
        world_to_local = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([sin_yaw, cos_yaw, torch.zeros_like(yaw)], dim=1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
        ], dim=2)

        # Transform to local frame
        rot_matrix_local = torch.bmm(world_to_local, rot_matrix)
        euler_angles_local = self._matrix_to_euler_xyz(rot_matrix_local)

        vel_local = torch.einsum("bij,bj->bi", world_to_local, root_linvel)
        ang_vel_local = torch.einsum("bij,bj->bi", world_to_local, root_angvel)

        # Fill observation buffer
        self._obs_tensor[..., 0:3] = root_pos - self.target_states[..., 9:12]
        self._obs_tensor[..., 3:6] = euler_angles_local
        self._obs_tensor[..., 6:9] = vel_local
        self._obs_tensor[..., 9:12] = ang_vel_local
        self._obs_tensor[..., 12:12+self.num_actions] = self.actions

        return {"policy": self._obs_tensor}

    def _quat_to_rot_matrix(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (wxyz) to rotation matrix."""
        w, x, y, z = quat_wxyz.unbind(-1)
        rot_matrix = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).reshape(-1, 3, 3)
        return rot_matrix

    def _matrix_to_euler_xyz(self, rot_matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to Euler angles (XYZ convention)."""
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

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        reward, terminated, self.item_reward_info = self._compute_quadcopter_reward()
        self._terminated_buf[:] = terminated.bool()
        self.rew_buf[:] = reward
        self.pre_actions = self.actions.clone()
        return self.rew_buf

    def _compute_quadcopter_reward(self):
        """Compute quadcopter reward with obstacle avoidance."""
        # Get robot state
        root_pos = self._robot.data.root_pos_w.torch
        root_quat = self._robot.data.root_quat_w.torch  # xyzw
        root_linvel = self._robot.data.root_lin_vel_w.torch
        root_angvel = self._robot.data.root_ang_vel_w.torch

        # Target positions
        target_positions = self.target_states[..., 9:12]
        relative_positions = target_positions - root_pos

        # Distance reward
        distance = torch.norm(relative_positions, dim=1)
        pose_reward = 1.0 / (1.0 + torch.square(1.6 * distance))

        # Uprightness reward
        root_quat_wxyz = root_quat[:, [3, 0, 1, 2]]
        ups = self._quat_rotate(root_quat, torch.tensor([0, 0, 1.0], device=self.device).expand(self.num_envs, -1))
        ups_reward = torch.square((ups[..., 2] + 1) / 2)

        # Spin reward
        spinnage = torch.square(root_angvel[:, -1])
        spin_reward = 1.0 / (1.0 + torch.square(spinnage))

        # Effort reward
        effort_reward = 0.1 * torch.exp(-self.actions.pow(2).sum(-1))

        # Action smoothness reward
        action_diff = torch.norm(self.actions[..., :-1] - self.pre_actions[..., :-1], dim=-1)
        action_smoothness_reward = 0.1 * torch.exp(-action_diff)

        # Thrust reward
        thrust_reward = 0.05 * (1 - torch.abs(0.1533 - self.actions[..., -1]))

        # Collision penalty
        alive_reward = torch.where(self.collisions > 0, -500.0, 0.5)

        # Total reward
        reward = (
            pose_reward
            + pose_reward * (ups_reward + spin_reward)
            + effort_reward
            + action_smoothness_reward
            + thrust_reward
            + alive_reward
        )

        # Reset conditions
        ones = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        die = torch.zeros_like(ones)

        reset = torch.where(root_pos[..., 2] < 0.3, ones, die)
        reset = torch.where(root_pos[..., 2] > 1.7, ones, reset)
        reset = torch.where(distance > 2.0, ones, reset)
        reset = torch.where(ups[..., 2] < 0.0, ones, reset)

        item_reward_info = {
            "pose_reward": pose_reward,
            "ups_reward": ups_reward,
            "spin_reward": spin_reward,
            "effort_reward": effort_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "thrust_reward": thrust_reward,
            "alive_reward": alive_reward,
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
        terminated = self._terminated_buf.bool()
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        super()._reset_idx(env_ids)

        # Reset robot state
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]

        # Randomize position
        default_root_pose[:, 0:2] = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        default_root_pose[:, 2:3] = torch.rand(len(env_ids), 1, device=self.device) * 2 - 1

        # Randomize orientation
        root_angle = torch.stack([
            0.01 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi),
            0.01 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi),
            0.05 * (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi - torch.pi)
        ], dim=-1)

        # Convert euler to quaternion (xyzw)
        root_quat = quat_from_euler_xyz(root_angle[:, 0], root_angle[:, 1], root_angle[:, 2])
        default_root_pose[:, 3:7] = root_quat

        # Randomize velocities
        default_root_vel[:, 0:3] = 0.0
        default_root_vel[:, 3:6] = 0.0

        # Write to simulation
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)

        # Reset buffers
        self.actions[env_ids] = 0
        self.pre_actions[env_ids] = 0
        self.cmd_thrusts[env_ids] = 0
        self.collisions[env_ids] = 0
