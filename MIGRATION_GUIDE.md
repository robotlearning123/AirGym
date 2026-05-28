# AirGym Migration Guide: IsaacSim 6 & IsaacLab 3

## Overview

This document describes the migration of AirGym from IsaacGym to IsaacSim 6 & IsaacLab 3.

## Key Changes

### 1. Quaternion Format
- **Old (IsaacGym)**: WXYZ format `(w, x, y, z)`
- **New (IsaacLab)**: XYZW format `(x, y, z, w)`

### 2. Data Access
- **Old**: Direct tensor access via `gymtorch.wrap_tensor()`
- **New**: Use `.torch` accessor on ProxyArray objects
  ```python
  # Old
  root_pos = robot.data.root_pos_w  # torch.Tensor

  # New
  root_pos = robot.data.root_pos_w.torch  # ProxyArray -> torch.Tensor
  ```

### 3. Write Methods
- **Old**: `write_root_pose_to_sim(pose, env_ids)`
- **New**: `write_root_pose_to_sim_index(pose, env_ids)` or `write_root_pose_to_sim_mask(pose, env_mask)`

### 4. Configuration System
- **Old**: Custom `BaseConfig` class with nested classes
- **New**: IsaacLab's `@configclass` decorator with `DirectRLEnvCfg`

### 5. Environment Base Class
- **Old**: Custom `BaseTask` class using raw IsaacGym API
- **New**: IsaacLab's `DirectRLEnv` class

## New File Structure

```
airgym/
├── envs/
│   ├── base/
│   │   ├── base_task_isaaclab.py          # New IsaacLab base task
│   │   ├── base_config_isaaclab.py        # New IsaacLab config
│   │   ├── hovering_isaaclab.py           # Hovering environment
│   │   ├── hovering_config_isaaclab.py    # Hovering config
│   │   ├── customized_isaaclab.py         # Customized environment
│   │   ├── customized_config_isaaclab.py  # Customized config
│   │   ├── depthgen_isaaclab.py           # Depth generation environment
│   │   └── depthgen_config_isaaclab.py    # Depth generation config
│   ├── task/
│   │   ├── avoid_isaaclab.py              # Obstacle avoidance environment
│   │   ├── avoid_config_isaaclab.py       # Obstacle avoidance config
│   │   ├── tracking_isaaclab.py           # Trajectory tracking environment
│   │   ├── tracking_config_isaaclab.py    # Trajectory tracking config
│   │   ├── planning_isaaclab.py           # Path planning environment
│   │   ├── planning_config_isaaclab.py    # Path planning config
│   │   ├── balloon_isaaclab.py            # Balloon popping environment
│   │   ├── balloon_config_isaaclab.py     # Balloon popping config
│   │   ├── maplanning_isaaclab.py         # Multi-agent planning environment
│   │   └── maplanning_config_isaaclab.py  # Multi-agent planning config
│   └── __init___isaaclab.py               # Environment registration
├── utils/
│   ├── task_registry_isaaclab.py          # Task registry
│   ├── helpers_isaaclab.py                # Helper utilities
│   └── isaacgym_compat.py                 # Compatibility layer
└── ...
```

## Available IsaacLab Environments

| Environment | Task Name | Description |
|------------|-----------|-------------|
| Hovering | `hovering_isaaclab` | Basic hovering control |
| Avoid | `avoid_isaaclab` | Obstacle avoidance with obstacles |
| Customized | `customized_isaaclab` | Base environment with depth cameras |
| DepthGen | `depthgen_isaaclab` | Depth data generation |
| Tracking | `tracking_isaaclab` | Trajectory tracking |
| Planning | `planning_isaaclab` | Path planning (goal reaching) |
| Balloon | `balloon_isaaclab` | Balloon popping task |
| MAPlanning | `maplanning_isaaclab` | Multi-agent planning (4 robots) |

## Running IsaacLab Environments

### Training
```bash
python scripts/runner_isaaclab.py --task hovering_isaaclab --train --ctl_mode vel
```

### Testing
```bash
python scripts/runner_isaaclab.py --task hovering_isaaclab --play --ctl_mode vel
```

### Available Control Modes
- `pos`: Position control
- `vel`: Velocity control
- `atti`: Attitude control (w, x, y, z, thrust)
- `rate`: Rate control
- `prop`: Direct propeller control

## Migration Steps for Custom Environments

1. **Create new config class** using `@configclass` decorator
2. **Create new environment class** extending `DirectRLEnv`
3. **Implement required methods**:
   - `_setup_scene()`: Setup scene with articulations
   - `_pre_physics_step()`: Process actions
   - `_apply_action()`: Apply actions to simulation
   - `_get_observations()`: Compute observations
   - `_get_rewards()`: Compute rewards
   - `_get_dones()`: Compute done signals
   - `_reset_idx()`: Reset environments

4. **Update quaternion handling** from WXYZ to XYZW
5. **Update data access** to use `.torch` accessor
6. **Update write methods** to use `_index` or `_mask` variants

## Dependencies

### IsaacLab Mode
```bash
pip install -e ".[isaaclab]"
```

### Legacy IsaacGym Mode (deprecated)
```bash
pip install -e ".[isaacgym]"
```

## Known Issues

1. Some environments may need additional tuning for IsaacLab
2. Camera sensors require different setup in IsaacLab
3. ROS integration may need updates for IsaacLab
4. The `rlPx4Controller` dependency needs to be installed separately
5. Multi-agent environments (maplanning) may need additional testing

## References

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [IsaacLab Migration Guide](https://isaac-sim.github.io/IsaacLab/source/migration/migrating_to_isaaclab_3-0.html)
