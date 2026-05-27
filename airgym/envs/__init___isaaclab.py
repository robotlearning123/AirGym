# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab-compatible environment registration for AirGym.

import os
import traceback
from airgym.utils.task_registry_isaaclab import task_registry_isaaclab

# Try to import IsaacLab
try:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.utils.configclass import configclass
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    print("WARNING: IsaacLab not available. Falling back to legacy IsaacGym mode.")

# Import IsaacLab-compatible environments
if ISAACLAB_AVAILABLE:
    try:
        from airgym.envs.base.hovering_config_isaaclab import HoveringIsaacLabCfg
        from airgym.envs.base.hovering_isaaclab import HoveringIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab hovering environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.task.avoid_config_isaaclab import AvoidIsaacLabCfg
        from airgym.envs.task.avoid_isaaclab import AvoidIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab avoid environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.base.customized_config_isaaclab import CustomizedIsaacLabCfg
        from airgym.envs.base.customized_isaaclab import CustomizedIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab customized environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.base.depthgen_config_isaaclab import DepthGenIsaacLabCfg
        from airgym.envs.base.depthgen_isaaclab import DepthGenIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab depthgen environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.task.balloon_config_isaaclab import BalloonIsaacLabCfg
        from airgym.envs.task.balloon_isaaclab import BalloonIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab balloon environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.task.tracking_config_isaaclab import TrackingIsaacLabCfg
        from airgym.envs.task.tracking_isaaclab import TrackingIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab tracking environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.task.planning_config_isaaclab import PlanningIsaacLabCfg
        from airgym.envs.task.planning_isaaclab import PlanningIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab planning environment: {e}")
        traceback.print_exc()

    try:
        from airgym.envs.task.maplanning_config_isaaclab import MAPlanningIsaacLabCfg
        from airgym.envs.task.maplanning_isaaclab import MAPlanningIsaacLab
    except ImportError as e:
        print(f"WARNING: Failed to import IsaacLab maplanning environment: {e}")
        traceback.print_exc()

# Define task configurations
TASK_CONFIGS_ISAACLAB = [
    {
        'name': 'hovering_isaaclab',
        'config_class': HoveringIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': HoveringIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'avoid_isaaclab',
        'config_class': AvoidIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': AvoidIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'customized_isaaclab',
        'config_class': CustomizedIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': CustomizedIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'depthgen_isaaclab',
        'config_class': DepthGenIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': DepthGenIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'balloon_isaaclab',
        'config_class': BalloonIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': BalloonIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'tracking_isaaclab',
        'config_class': TrackingIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': TrackingIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'planning_isaaclab',
        'config_class': PlanningIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': PlanningIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
    {
        'name': 'maplanning_isaaclab',
        'config_class': MAPlanningIsaacLabCfg if ISAACLAB_AVAILABLE else None,
        'task_class': MAPlanningIsaacLab if ISAACLAB_AVAILABLE else None,
        'is_isaaclab': True,
    },
]


def register_tasks_isaaclab():
    """Register IsaacLab-compatible tasks."""
    if not ISAACLAB_AVAILABLE:
        print("IsaacLab not available. Skipping IsaacLab task registration.")
        return

    for config in TASK_CONFIGS_ISAACLAB:
        try:
            if config['config_class'] is None or config['task_class'] is None:
                continue

            task_registry_isaaclab.register(
                config['name'],
                config['task_class'],
                config['config_class'](),
                is_isaaclab=config['is_isaaclab']
            )
            print(f"Registered IsaacLab task: {config['name']}")
        except Exception as e:
            print(f"WARNING: Failed to register IsaacLab task {config['name']}: {e}")
            traceback.print_exc()


# Register tasks on import
register_tasks_isaaclab()
