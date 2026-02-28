"""Minimal setup.py for editable installs (pip install -e .)."""
from setuptools import setup, find_packages

setup(
    name="safe-rl-locomotion",
    version="0.1.0",
    description="Baseline PPO and Lagrangian-constrained C-PPO on MuJoCo locomotion tasks",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
