import os

AIRGYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AIRGYM_ENVS_DIR = os.path.join(AIRGYM_ROOT_DIR, 'airgym', 'envs')

# Check if IsaacLab is available
try:
    import isaaclab
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

# Monkey-patch wp.launch to handle two IsaacSim 6.0 compatibility issues:
# 1. ProxyArray unwrapping: omni.warp.core-1.12.0 sometimes fails to convert
#    ProxyArray via __cuda_array_interface__.
# 2. CPU/CUDA device mismatch: The PhysX tensor API (device_ordinal=-1) returns
#    CPU arrays even when GPU dynamics are enabled. Warp kernels expect CUDA
#    inputs when launched on CUDA. This patch transparently copies CPU arrays
#    to the target device before launch and restores outputs afterward.
if ISAACLAB_AVAILABLE:
    try:
        import warp as wp
        _original_launch = wp.launch

        def _patched_launch(kernel, dim, inputs=None, outputs=None, device=None, **kwargs):
            from isaaclab.utils.warp.proxy_array import ProxyArray
            if inputs is None:
                inputs = []
            else:
                inputs = [x.warp if isinstance(x, ProxyArray) else x for x in inputs]
            if outputs is None:
                outputs = []
            else:
                outputs = [x.warp if isinstance(x, ProxyArray) else x for x in outputs]

            # Resolve target device
            target = device or (inputs[0].device if inputs else None)
            target_str = str(target) if target else ""

            # Only patch if target is CUDA — CPU kernels don't need device transfer
            if "cuda" in target_str:
                final_inputs = []
                for arr in inputs:
                    if arr is not None and hasattr(arr, 'device') and "cpu" in str(arr.device):
                        final_inputs.append(wp.clone(arr, device=target))
                    else:
                        final_inputs.append(arr)
                inputs = final_inputs

                final_outputs = []
                cpu_output_refs = []  # (index, original_cpu_array) pairs
                for i, arr in enumerate(outputs):
                    if arr is not None and hasattr(arr, 'device') and "cpu" in str(arr.device):
                        cpu_output_refs.append((i, arr))
                        final_outputs.append(wp.zeros_like(arr, device=target))
                    else:
                        final_outputs.append(arr)
                outputs = final_outputs
            else:
                cpu_output_refs = []

            result = _original_launch(kernel, dim, inputs=inputs, outputs=outputs, device=device, **kwargs)

            # Copy CUDA outputs back to the original CPU arrays
            for idx, cpu_arr in cpu_output_refs:
                cpu_arr.assign(wp.clone(outputs[idx], device="cpu"))

            return result

        wp.launch = _patched_launch
    except Exception:
        pass

    # Monkey-patch ArticulationView write methods to copy CUDA data/indices to CPU.
    # The PhysX tensor API in IsaacSim 6.0 has device_ordinal=-1 (CPU) even when
    # GPU dynamics are enabled, so all write operations expect CPU data.
    try:
        import warp as wp

        def _to_cpu_warp(arr):
            """Copy a warp array to CPU if it's on CUDA."""
            if arr is None:
                return None
            if hasattr(arr, 'device') and "cuda" in str(arr.device):
                return wp.clone(arr, device="cpu")
            return arr

        from omni.physics.tensors.impl.api import ArticulationView

        WRITE_METHODS = [
            'set_root_transforms',
            'set_root_velocities',
            'set_dof_positions',
            'set_dof_velocities',
            'set_dof_position_targets',
            'set_dof_velocity_targets',
            'set_dof_actuation_forces',
            'apply_forces_and_torques_at_position',
            'set_spatial_tendon_properties',
            'set_dof_stiffnesses',
            'set_dof_dampings',
            'set_dof_limits',
            'set_dof_max_velocities',
            'set_dof_max_forces',
            'set_dof_armatures',
        ]

        for method_name in WRITE_METHODS:
            if not hasattr(ArticulationView, method_name):
                continue

            original_method = getattr(ArticulationView, method_name)

            def _make_wrapper(orig):
                def wrapper(self, *args, **kwargs):
                    new_args = []
                    for arg in args:
                        if isinstance(arg, wp.array):
                            new_args.append(_to_cpu_warp(arg))
                        else:
                            new_args.append(arg)
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, wp.array):
                            new_kwargs[k] = _to_cpu_warp(v)
                        else:
                            new_kwargs[k] = v
                    return orig(self, *new_args, **new_kwargs)
                return wrapper

            setattr(ArticulationView, method_name, _make_wrapper(original_method))
    except Exception:
        pass
