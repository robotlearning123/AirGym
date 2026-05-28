# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible runner for AirGym.
#
# Usage:
#   /mnt/storage/isaacsim-6.0-official/venv/bin/python scripts/runner_isaaclab.py --task AirGymHovering --headless
#   /mnt/storage/isaacsim-6.0-official/venv/bin/python scripts/runner_isaaclab.py --task AirGymHovering --steps 200

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Step 1: Initialize Isaac Sim BEFORE any other imports
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="AirGym IsaacLab Runner")
parser.add_argument("--task", type=str, default="hovering_isaaclab", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--ctl_mode", type=str, default="prop",
                    choices=["pos", "vel", "atti", "rate", "prop"],
                    help="Control mode")
parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
args = parser.parse_args()

app = SimulationApp({"headless": args.headless})

# Step 2: Now import IsaacLab and AirGym (omni modules available)
import torch
from airgym.envs.__init___isaaclab import register_tasks_isaaclab
from airgym.utils.task_registry_isaaclab import task_registry_isaaclab


def log(msg):
    print(msg, flush=True, file=sys.stderr)

register_tasks_isaaclab()

if args.task not in task_registry_isaaclab.get_registered_tasks():
    log(f"Error: Task '{args.task}' not found.")
    log(f"Available tasks: {task_registry_isaaclab.get_registered_tasks()}")
    app.close()
    sys.exit(1)

env, env_cfg = task_registry_isaaclab.make_env(args.task, args=args)

log(f"Environment created: {args.task}")
log(f"Number of environments: {env.num_envs}")
log(f"Device: {env.device}")

num_robots = getattr(env, 'num_robots', 1)
num_actions = env.num_actions
if num_robots > 1:
    action_shape = (env.num_envs, num_robots, num_actions)
else:
    action_shape = (env.num_envs, num_actions)

obs, _ = env.reset()
total_reward = 0.0

for step in range(args.steps):
    actions = torch.rand(action_shape, device=env.device) * 2 - 1
    obs, rewards, terminated, truncated, infos = env.step(actions)
    total_reward += rewards.sum().item()
    if step % 20 == 0:
        log(f"Step {step}/{args.steps}  Mean reward: {rewards.mean().item():.4f}")

log(f"\nDone. {args.steps} steps, total reward: {total_reward:.2f}")
env.close()
app.close()
