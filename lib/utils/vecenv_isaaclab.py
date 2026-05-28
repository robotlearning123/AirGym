# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible vectorized environment wrapper for AirGym.

import numpy as np
import torch
from argparse import Namespace

try:
    import gymnasium as gym
    from gymnasium import spaces
    USE_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USE_GYMNASIUM = False

from airgym.envs import __init___isaaclab
from airgym.utils.task_registry_isaaclab import task_registry_isaaclab

from lib.utils.ivecenv import IVecEnv


class ExtractObsWrapperIsaacLab(gym.Wrapper):
    """Wrapper that extracts observations from IsaacLab environment."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        if USE_GYMNASIUM:
            obs, info = super().reset(**kwargs)
            if isinstance(obs, dict) and "policy" in obs:
                obs = obs["policy"]
            return obs
        else:
            observations, _privileged_observations = super().reset(**kwargs)
            return observations

    def step(self, action):
        if USE_GYMNASIUM:
            obs, rewards, terminated, truncated, infos = super().step(action)
            if isinstance(obs, dict) and "policy" in obs:
                obs = obs["policy"]
            dones = terminated | truncated
            return obs, rewards, dones, infos
        else:
            observations, _privileged_observations, rewards, dones, infos = super().step(action)
            return observations, rewards, dones, infos


class AirGymRLGPUEnvIsaacLab(IVecEnv):
    """IsaacLab-compatible vectorized environment for RL training."""

    def __init__(self, config_name, num_actors, **kwargs):
        print("AirGymRLGPUEnvIsaacLab:", config_name, num_actors, kwargs)
        self.use_image = kwargs.get('use_image', False)

        # Get task from registry
        if config_name not in task_registry_isaaclab.get_registered_tasks():
            raise ValueError(f"Task '{config_name}' not registered. "
                           f"Available: {task_registry_isaaclab.get_registered_tasks()}")

        # Create environment
        from argparse import Namespace
        args = Namespace(**kwargs) if kwargs else Namespace(headless=True, num_envs=num_actors)
        self.env, self.env_info = task_registry_isaaclab.make_env(config_name, args=args)

        self.env = ExtractObsWrapperIsaacLab(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {
            'action_space': spaces.Box(
                np.ones(self.env.unwrapped.num_actions) * -1.,
                np.ones(self.env.unwrapped.num_actions) * 1.
            ),
        }

        if self.use_image and hasattr(self.env.unwrapped, 'cam_resolution'):
            info['observation_space'] = spaces.Dict({
                'image': spaces.Box(
                    0, 1,
                    shape=(self.env.unwrapped.cam_channel,
                           self.env.unwrapped.cam_resolution[0],
                           self.env.unwrapped.cam_resolution[1])
                ),
                'observation': spaces.Box(
                    np.ones(self.env.unwrapped.cfg.num_observations) * -np.inf,
                    np.ones(self.env.unwrapped.cfg.num_observations) * np.inf
                )
            })
        else:
            info['observation_space'] = spaces.Box(
                np.ones(self.env.unwrapped.cfg.num_observations) * -np.inf,
                np.ones(self.env.unwrapped.cfg.num_observations) * np.inf
            )

        return info


# Auto-register IsaacLab environments
def register_isaaclab_envs():
    """Register all IsaacLab environments with the vecenv system."""
    try:
        from lib.utils import env_configurations

        for task_name in task_registry_isaaclab.get_registered_tasks():
            env_configurations.register(
                task_name,
                {
                    'env_creator': lambda task_name=task_name, **kwargs: \
                        task_registry_isaaclab.make_env(task_name, args=Namespace(headless=True, ctl_mode='prop')),
                    'vecenv_type': 'AirGym-RLGPU-IsaacLab'
                }
            )

        # Register the vec environment
        from lib.utils.vecenv import register
        register('AirGym-RLGPU-IsaacLab',
                lambda config_name, num_actors, **kwargs: AirGymRLGPUEnvIsaacLab(config_name, num_actors, **kwargs))

        print("Registered IsaacLab environments with vecenv system")
    except ImportError as e:
        print(f"Could not register IsaacLab envs with vecenv: {e}")


# Auto-register on import
register_isaaclab_envs()
