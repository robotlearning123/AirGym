# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible helper utilities for AirGym.

import argparse
import torch
import numpy as np


def get_args_isaaclab():
    """Parse command line arguments for IsaacLab mode.

    This replaces the old get_args() that used gymutil.parse_arguments()
    which is not available in IsaacLab.
    """
    parser = argparse.ArgumentParser(description="AirGym IsaacLab Runner")

    parser.add_argument("--task", type=str, default="hovering_isaaclab",
                       help="Task name (e.g., hovering_isaaclab, avoid_isaaclab)")
    parser.add_argument("--num_envs", type=int, default=256,
                       help="Number of environments")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (0 for random)")
    parser.add_argument("--headless", action="store_true", default=False,
                       help="Run without rendering")
    parser.add_argument("--ctl_mode", type=str, default="vel",
                       choices=["pos", "vel", "atti", "rate", "prop"],
                       help="Control mode")
    parser.add_argument("--train", action="store_true",
                       help="Training mode")
    parser.add_argument("--play", action="store_true",
                       help="Play/test mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for play mode")
    parser.add_argument("--experiment_name", type=str, default="airgym_isaaclab",
                       help="Experiment name")
    parser.add_argument("--rl_device", type=str, default="cuda:0",
                       help="Device for RL algorithm")
    parser.add_argument("--sim_device", type=str, default="cuda:0",
                       help="Device for simulation")

    return parser.parse_args()


def update_cfg_from_args_isaaclab(env_cfg, args):
    """Update IsaacLab config from command line arguments.

    Args:
        env_cfg: IsaacLab DirectRLEnvCfg instance
        args: Parsed arguments

    Returns:
        Updated config
    """
    if env_cfg is None:
        return env_cfg

    if hasattr(args, 'num_envs') and args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    if hasattr(args, 'ctl_mode') and args.ctl_mode is not None:
        env_cfg.ctl_mode = args.ctl_mode

    return env_cfg


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed. If 0, a random seed is generated.
    """
    if seed <= 0:
        seed = np.random.randint(0, 10000)
        print(f"Setting random seed: {seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed
