import os

AIRGYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
AIRGYM_ENVS_DIR = os.path.join(AIRGYM_ROOT_DIR, 'airgym', 'envs')

# Check if IsaacLab is available
try:
    import isaaclab
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

print("AIRGYM_ROOT_DIR", AIRGYM_ROOT_DIR)
print("IsaacLab available:", ISAACLAB_AVAILABLE)