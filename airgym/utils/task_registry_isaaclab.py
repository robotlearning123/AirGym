# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible task registry for AirGym.

import os
import torch
import numpy as np

from isaaclab.utils.configclass import configclass


class TaskRegistryIsaacLab:
    """Task registry that supports both old AirGym and new IsaacLab environments."""

    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.is_isaaclab = {}  # Track which tasks use IsaacLab

    def register(self, name: str, task_class, env_cfg, is_isaaclab: bool = False):
        """Register a task.

        Args:
            name: Task name
            task_class: Task class
            env_cfg: Environment configuration
            is_isaaclab: Whether this task uses IsaacLab
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.is_isaaclab[name] = is_isaaclab

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name):
        return self.env_cfgs[name]

    def get_registered_tasks(self):
        return list(self.task_classes.keys())

    def is_isaaclab_task(self, name: str) -> bool:
        return self.is_isaaclab.get(name, False)

    def make_env(self, name, args=None, env_cfg=None):
        """Create an environment.

        Args:
            name: Task name
            args: Command line arguments
            env_cfg: Environment config override

        Returns:
            Environment instance and config
        """
        if name not in self.task_classes:
            raise ValueError(f"Task with name: {name} was not registered")

        task_class = self.get_task_class(name)

        if env_cfg is None:
            env_cfg = self.get_cfgs(name)

        if self.is_isaaclab_task(name):
            # IsaacLab environment
            headless = getattr(args, 'headless', True) if args is not None else True

            # Update config from args
            if args is not None:
                num_envs = getattr(args, 'num_envs', None)
                if num_envs is not None:
                    env_cfg.scene.num_envs = num_envs
                ctl_mode = getattr(args, 'ctl_mode', None)
                if ctl_mode is not None:
                    env_cfg.ctl_mode = ctl_mode

            env = task_class(cfg=env_cfg, render_mode=None if headless else "human")
        else:
            # Old AirGym environment
            from airgym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, parse_sim_params

            if args is None:
                args = get_args()

            env_cfg = update_cfg_from_args(env_cfg, args)

            seed = env_cfg.seed
            if seed == -1:
                seed = np.random.randint(0, 10000)
                print("Setting seed: {}".format(seed))
            np.random.seed(seed)
            torch.manual_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            sim_params = {"sim": class_to_dict(env_cfg.sim)}
            try:
                sim_params = parse_sim_params(args, sim_params)
            except AttributeError:
                print("Ignore! sim_params is not required in real robot inferencing.")
                sim_params = None

            env = task_class(
                cfg=env_cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                sim_device=args.sim_device,
                headless=args.headless
            )

        return env, env_cfg


# Global task registry
task_registry_isaaclab = TaskRegistryIsaacLab()
