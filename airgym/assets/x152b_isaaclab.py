# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# IsaacLab ArticulationCfg for the X152b quadrotor.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from airgym import AIRGYM_ROOT_DIR

X152B_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{AIRGYM_ROOT_DIR}/airgym/assets/robots/X152b/model.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            angular_damping=0.0,
            linear_damping=0.0,
            max_angular_velocity=100.0,
            max_linear_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
"""Configuration for the X152b quadrotor."""
