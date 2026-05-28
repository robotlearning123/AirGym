# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible runner for AirGym.

import argparse
import os
import sys
import yaml
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_args_isaaclab():
    """Parse command line arguments for IsaacLab mode."""
    parser = argparse.ArgumentParser(description="AirGym IsaacLab Runner")

    parser.add_argument("--task", type=str, default="hovering_isaaclab", help="Task name")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    parser.add_argument("--ctl_mode", type=str, default="vel",
                       choices=["pos", "vel", "atti", "rate", "prop"],
                       help="Control mode")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--play", action="store_true", help="Play/test mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--experiment_name", type=str, default="airgym_isaaclab", help="Experiment name")

    return parser.parse_args()


def main():
    args = get_args_isaaclab()

    # Import IsaacLab-compatible environment registration
    from airgym.envs import __init___isaaclab
    from airgym.utils.task_registry_isaaclab import task_registry_isaaclab

    # Check if task is registered
    if args.task not in task_registry_isaaclab.get_registered_tasks():
        print(f"Error: Task '{args.task}' not found.")
        print(f"Available tasks: {task_registry_isaaclab.get_registered_tasks()}")
        sys.exit(1)

    # Create environment
    env, env_cfg = task_registry_isaaclab.make_env(args.task, args=args)

    print(f"Environment created: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Device: {env.device}")

    # Determine action shape
    num_robots = getattr(env, 'num_robots', 1)
    num_actions = env.num_actions
    if num_robots > 1:
        action_shape = (env.num_envs, num_robots, num_actions)
    else:
        action_shape = (env.num_envs, num_actions)

    # Run environment
    if args.train:
        # Training loop
        print("Starting training...")
        obs, _ = env.reset()

        for step in range(1000000):
            # Random actions for testing
            actions = torch.randn(action_shape, device=env.device)
            obs, rewards, terminated, truncated, infos = env.step(actions)

            if step % 100 == 0:
                print(f"Step {step}, Mean Reward: {rewards.mean().item():.4f}")

    elif args.play:
        # Play/test mode
        print("Starting play mode...")
        obs, _ = env.reset()

        for step in range(1000):
            # Random actions for testing
            actions = torch.randn(action_shape, device=env.device)
            obs, rewards, terminated, truncated, infos = env.step(actions)

            if step % 100 == 0:
                print(f"Step {step}, Mean Reward: {rewards.mean().item():.4f}")

    else:
        print("Please specify --train or --play mode.")
        sys.exit(1)

    env.close()


if __name__ == "__main__":
    main()
