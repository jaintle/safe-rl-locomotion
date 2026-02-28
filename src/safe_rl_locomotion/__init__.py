"""
safe_rl_locomotion
==================
Top-level package for the Safe-RL Locomotion project.

Exposes the two main algorithm modules (PPO and Constrained-PPO) and the
shared utilities so that scripts can do simple package-level imports.

Nothing is instantiated at import time; all heavy objects (networks, envs)
are constructed inside the training scripts.
"""

__version__ = "1.0.0"
