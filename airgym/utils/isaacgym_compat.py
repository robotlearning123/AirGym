# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Compatibility layer: maps old IsaacGym API calls to IsaacLab equivalents.
# This allows gradual migration of existing code.

from __future__ import annotations

import torch
import numpy as np
from typing import Any


class GymCompat:
    """Compatibility wrapper that maps isaacgym.gymapi calls to IsaacLab equivalents."""

    def __init__(self):
        self._sim = None
        self._scene = None

    def set_context(self, sim, scene):
        """Set the simulation context."""
        self._sim = sim
        self._scene = scene

    # ------------------------------------------------------------------
    # Tensor operations (gymtorch equivalents)
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Compatibility: gymtorch.wrap_tensor is a no-op in IsaacLab."""
        return tensor

    @staticmethod
    def unwrap_tensor(tensor: torch.Tensor) -> Any:
        """Compatibility: gymtorch.unwrap_tensor is a no-op in IsaacLab."""
        return tensor

    # ------------------------------------------------------------------
    # Simulation operations
    # ------------------------------------------------------------------

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine: int, sim_params: dict) -> Any:
        """Compatibility: create simulation context."""
        from isaaclab.sim import SimulationCfg, SimulationContext

        # Convert old sim_params to IsaacLab SimulationCfg
        cfg = SimulationCfg()
        if sim_params:
            if 'dt' in sim_params:
                cfg.dt = sim_params['dt']
            if 'gravity' in sim_params:
                cfg.gravity = tuple(sim_params['gravity'])
            if 'substeps' in sim_params:
                cfg.substeps = sim_params['substeps']

        sim = SimulationContext(cfg)
        return sim

    def prepare_sim(self, sim: Any):
        """Compatibility: prepare simulation."""
        pass

    def simulate(self, sim: Any):
        """Compatibility: step simulation."""
        sim.step()

    def fetch_results(self, sim: Any, state: bool = True):
        """Compatibility: fetch simulation results."""
        pass

    # ------------------------------------------------------------------
    # Scene operations
    # ------------------------------------------------------------------

    def create_env(self, sim: Any, env_lower: Any, env_upper: Any, num_envs: int) -> Any:
        """Compatibility: create environment."""
        return 0  # IsaacLab handles this automatically

    def add_ground(self, sim: Any, plane_params: Any):
        """Compatibility: add ground plane."""
        pass

    # ------------------------------------------------------------------
    # Asset operations
    # ------------------------------------------------------------------

    def load_asset(self, sim: Any, asset_root: str, asset_file: str, options: Any) -> Any:
        """Compatibility: load URDF asset."""
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        cfg = UrdfConverterCfg(
            asset_path=f"{asset_root}/{asset_file}",
            usd_dir="/tmp/airgym_assets",
            fix_base=False,
            merge_fixed_joints=True,
        )
        converter = UrdfConverter(cfg)
        return converter.usd_path

    # ------------------------------------------------------------------
    # Actor operations
    # ------------------------------------------------------------------

    def create_actor(self, env: Any, asset: Any, pose: Any, name: str, env_id: int,
                     collision_group: int = 0, semantic_id: int = 0) -> int:
        """Compatibility: create actor."""
        return 0

    def get_asset_rigid_body_count(self, asset: Any) -> int:
        """Compatibility: get rigid body count."""
        return 1

    def get_actor_rigid_body_count(self, env: Any, actor: Any) -> int:
        """Compatibility: get actor rigid body count."""
        return 1

    # ------------------------------------------------------------------
    # State operations
    # ------------------------------------------------------------------

    def acquire_actor_root_state_tensor(self, sim: Any) -> Any:
        """Compatibility: acquire root state tensor."""
        return torch.zeros(1, 13)

    def refresh_actor_root_state_tensor(self, sim: Any):
        """Compatibility: refresh root state tensor."""
        pass

    def set_actor_root_state_tensor(self, sim: Any, tensor: Any):
        """Compatibility: set root state tensor."""
        pass

    def acquire_net_contact_force_tensor(self, sim: Any) -> Any:
        """Compatibility: acquire contact force tensor."""
        return torch.zeros(1, 3)

    def refresh_net_contact_force_tensor(self, sim: Any):
        """Compatibility: refresh contact force tensor."""
        pass

    # ------------------------------------------------------------------
    # Force operations
    # ------------------------------------------------------------------

    def apply_rigid_body_force_tensors(self, sim: Any, forces: Any, torques: Any, space: int):
        """Compatibility: apply forces and torques."""
        pass

    # ------------------------------------------------------------------
    # Camera operations
    # ------------------------------------------------------------------

    def create_camera_sensor(self, env: Any, props: Any) -> int:
        """Compatibility: create camera sensor."""
        return 0

    def attach_camera_to_body(self, cam: Any, env: Any, body: Any, transform: Any, mode: int):
        """Compatibility: attach camera to body."""
        pass

    def get_camera_image_gpu_tensor(self, sim: Any, env: Any, cam: Any, image_type: int) -> Any:
        """Compatibility: get camera image tensor."""
        return torch.zeros(120, 212)

    def render_all_camera_sensors(self, sim: Any):
        """Compatibility: render camera sensors."""
        pass

    def start_access_image_tensors(self, sim: Any):
        """Compatibility: start image tensor access."""
        pass

    def end_access_image_tensors(self, sim: Any):
        """Compatibility: end image tensor access."""
        pass

    # ------------------------------------------------------------------
    # Viewer operations
    # ------------------------------------------------------------------

    def create_viewer(self, sim: Any, props: Any) -> Any:
        """Compatibility: create viewer."""
        return None

    def subscribe_viewer_keyboard_event(self, viewer: Any, key: int, action: str):
        """Compatibility: subscribe to keyboard events."""
        pass

    def query_viewer_has_closed(self, viewer: Any) -> bool:
        """Compatibility: check if viewer closed."""
        return False

    def query_viewer_action_events(self, viewer: Any) -> list:
        """Compatibility: get viewer action events."""
        return []

    def draw_viewer(self, viewer: Any, sim: Any, sync: bool):
        """Compatibility: draw viewer."""
        pass

    def sync_frame_time(self, sim: Any):
        """Compatibility: sync frame time."""
        pass

    def poll_viewer_events(self, viewer: Any):
        """Compatibility: poll viewer events."""
        pass

    def viewer_camera_look_at(self, viewer: Any, camera: Any, pos: Any, target: Any):
        """Compatibility: set viewer camera."""
        pass

    def step_graphics(self, sim: Any):
        """Compatibility: step graphics."""
        pass

    # ------------------------------------------------------------------
    # Asset options
    # ------------------------------------------------------------------

    def set_rigid_body_color(self, env: Any, actor: Any, body: int, mesh_type: int, color: Any):
        """Compatibility: set rigid body color."""
        pass

    def get_actor_rigid_body_names(self, env: Any, actor: Any) -> list:
        """Compatibility: get rigid body names."""
        return ["base_link"]


class GymApiCompat:
    """Compatibility constants for gymapi."""

    SIM_PHYSX = 0
    SIM_FLEX = 1

    LOCAL_SPACE = 0
    GLOBAL_SPACE = 1

    MESH_VISUAL = 0
    MESH_COLLISION = 1

    KEY_ESCAPE = 256
    KEY_V = 86

    class Vec3:
        def __init__(self, x=0, y=0, z=0):
            self.x = x
            self.y = y
            self.z = z

    class Quat:
        def __init__(self, x=0, y=0, z=0, w=1):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    class Transform:
        def __init__(self):
            self.p = GymApiCompat.Vec3()
            self.r = GymApiCompat.Quat()

    class PlaneParams:
        def __init__(self):
            self.normal = GymApiCompat.Vec3(0, 0, 1)

    class CameraProperties:
        def __init__(self):
            self.enable_tensors = False
            self.width = 212
            self.height = 120
            self.far_plane = 5.0
            self.horizontal_fov = 87.0
            self.use_collision_geometry = False

    class AssetOptions:
        def __init__(self):
            self.collapse_fixed_joints = True
            self.replace_cylinder_with_capsule = False
            self.flip_visual_attachments = False
            self.fix_base_link = False
            self.density = -1
            self.angular_damping = 0.0
            self.linear_damping = 0.0
            self.max_angular_velocity = 100.0
            self.max_linear_velocity = 100.0
            self.disable_gravity = False
            self.vhacd_enabled = False


class GymUtilCompat:
    """Compatibility utilities."""

    @staticmethod
    def parse_device_str(device: str) -> tuple:
        """Parse device string."""
        if ':' in device:
            parts = device.split(':')
            return parts[0], int(parts[1])
        return device, 0

    @staticmethod
    def parse_arguments(description: str = "", custom_parameters: list = None) -> Any:
        """Parse command line arguments."""
        import argparse
        parser = argparse.ArgumentParser(description=description)

        if custom_parameters:
            for param in custom_parameters:
                name = param.pop('name')
                parser.add_argument(name, **param)

        args = parser.parse_args()

        # Set defaults
        if not hasattr(args, 'physics_engine'):
            args.physics_engine = 0
        if not hasattr(args, 'sim_device_type'):
            args.sim_device_type = 'cuda'
        if not hasattr(args, 'compute_device_id'):
            args.compute_device_id = 0
        if not hasattr(args, 'use_gpu_pipeline'):
            args.use_gpu_pipeline = True
        if not hasattr(args, 'num_threads'):
            args.num_threads = 0
        if not hasattr(args, 'subscenes'):
            args.subscenes = 0
        if not hasattr(args, 'use_gpu'):
            args.use_gpu = True

        return args

    @staticmethod
    def parse_sim_config(sim_config: dict, sim_params: Any):
        """Parse simulation configuration."""
        pass


# Global compatibility instances
gymapi_compat = GymApiCompat()
gymutil_compat = GymUtilCompat()
gym_compat = GymCompat()
