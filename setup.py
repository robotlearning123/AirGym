from setuptools import find_packages, setup

setup(
    name='airgym',
    version='0.2.0',
    author='emNavi Tech',
    license="BSD 3-Clause",
    packages=find_packages(),
    author_email='',
    description='IsaacLab Drone RL Project (IsaacSim 6 / IsaacLab 3 compatible)',
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "scipy",
        "pyyaml",
        "pillow",
        "imageio",
        "ninja",
        'matplotlib',
        'torch>=2.0.0',
        'gymnasium>=0.29.0',
        'tensorboardX',
    ],
    extras_require={
        # IsaacLab dependencies (required for IsaacLab mode)
        'isaaclab': [
            'isaaclab',
            'isaacsim',
        ],
        # Legacy IsaacGym dependencies (deprecated)
        'isaacgym': [
            'gym==0.23.1',
            'rospkg',
            'rlpx4controller',
            'usd-core',
            'pytorch3d',
        ],
        # ROS dependencies (optional, for real robot deployment)
        'ros': [
            'rospkg',
            'rospy',
        ],
    },
)