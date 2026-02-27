"""
utils.py
========
Shared utility functions used across training scripts and algorithm modules.

Includes:
    - Seed initialisation (Python, NumPy, PyTorch, Gymnasium).
    - MLP factory (returns a nn.Sequential with configurable hidden layers).
    - Config loading from YAML files.
    - MetricLogger: writes per-step and per-episode metrics to a CSV file
      and optionally to stdout.
    - Checkpoint save / load helpers.

Design notes:
    - No algorithm logic here; pure utility code.
    - MetricLogger is the single point of I/O for training metrics so that
      plotting scripts have a consistent CSV schema to consume.

Status: STUB — implementations not yet written.
"""

from __future__ import annotations

import pathlib
# import csv
# import random
# import numpy as np
# import torch
# import yaml


def set_seeds(seed: int, env=None) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for: Python built-in `random`, NumPy, PyTorch (CPU and CUDA),
    and optionally the Gymnasium environment via `env.reset(seed=seed)`.

    Args:
        seed: Integer seed value.
        env : Optional Gymnasium environment instance.
    """
    raise NotImplementedError


def make_mlp(in_dim: int, out_dim: int, hidden_sizes: list[int],
             activation: str = "tanh"):
    """
    Construct a fully-connected MLP as a torch.nn.Sequential.

    Args:
        in_dim       : Input feature dimensionality.
        out_dim      : Output dimensionality.
        hidden_sizes : List of hidden layer widths, e.g. [64, 64].
        activation   : Activation function name ("tanh" or "relu").

    Returns:
        torch.nn.Sequential
    """
    raise NotImplementedError


def load_config(path: str | pathlib.Path) -> dict:
    """
    Load a YAML config file and return it as a plain dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    raise NotImplementedError


class MetricLogger:
    """
    Logs scalar metrics to a CSV file row by row.

    Each call to `log()` appends one row.  The CSV header is written on the
    first call (columns inferred from the keys of the first dict passed).

    Usage::

        logger = MetricLogger(save_dir / "metrics.csv")
        logger.log({"step": 1000, "episode_return": 42.3, ...})

    Status: STUB — not yet implemented.
    """

    def __init__(self, csv_path: str | pathlib.Path) -> None:
        """
        Args:
            csv_path: Destination path for the metrics CSV.
        """
        raise NotImplementedError

    def log(self, metrics: dict) -> None:
        """
        Append one row of metrics.

        Args:
            metrics: Dict mapping column names to scalar values.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Flush and close the underlying CSV file handle."""
        raise NotImplementedError


def save_checkpoint(agent, path: str | pathlib.Path, metadata: dict) -> None:
    """
    Persist agent network weights and training metadata to disk.

    Args:
        agent   : Agent instance with a `state_dict()` method (or equivalent).
        path    : File path for the checkpoint (e.g. ``checkpoints/step_10000.pt``).
        metadata: Dict of scalar metadata to embed in the checkpoint
                  (e.g. step count, seed, eval_return).
    """
    raise NotImplementedError


def load_checkpoint(path: str | pathlib.Path) -> dict:
    """
    Load a checkpoint from disk.

    Args:
        path: Path to a ``.pt`` checkpoint file.

    Returns:
        Checkpoint dictionary with keys ``state_dict`` and ``metadata``.
    """
    raise NotImplementedError
