"""
utils.py
========
Shared utility functions used across training scripts and algorithm modules.

Includes:
    - Seed initialisation (Python, NumPy, PyTorch, Gymnasium).
    - MLP factory (returns a nn.Sequential with configurable hidden layers).
    - Config loading from YAML files.
    - MetricLogger: writes per-step and per-episode metrics to a CSV file.
    - Checkpoint save / load helpers.

Design notes:
    - No algorithm logic here; pure utility code.
    - MetricLogger is the single point of I/O for training metrics so that
      plotting scripts have a consistent CSV schema to consume.
    - File is opened at MetricLogger construction so metrics.csv exists on disk
      immediately (important for smoke tests that check file existence).
"""

from __future__ import annotations

import csv
import pathlib
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Seed initialisation
# ---------------------------------------------------------------------------

def set_seeds(seed: int, env=None) -> None:
    """
    Set random seeds for full reproducibility.

    Sets seeds for: Python built-in ``random``, NumPy, PyTorch (CPU + CUDA),
    and optionally the Gymnasium environment.

    Args:
        seed: Integer seed value.
        env : Optional Gymnasium environment; if provided, ``env.reset(seed=seed)``
              is called to seed the environment's internal RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.reset(seed=seed)


# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, out_dim: int, hidden_sizes: List[int],
             activation: str = "tanh") -> nn.Sequential:
    """
    Construct a fully-connected MLP as a torch.nn.Sequential.

    Args:
        in_dim       : Input feature dimensionality.
        out_dim      : Output dimensionality.
        hidden_sizes : List of hidden layer widths, e.g. [64, 64].
        activation   : Activation function name — ``"tanh"`` or ``"relu"``.

    Returns:
        torch.nn.Sequential with Linear → Activation → ... → Linear layers.

    Raises:
        ValueError: If activation is not ``"tanh"`` or ``"relu"``.
    """
    _act_map = {"tanh": nn.Tanh, "relu": nn.ReLU}
    if activation not in _act_map:
        raise ValueError(f"Unknown activation '{activation}'. Choose 'tanh' or 'relu'.")
    act_cls = _act_map[activation]

    layers: list = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict:
    """
    Load a YAML config file and return it as a plain dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

class MetricLogger:
    """
    Logs scalar metrics to a CSV file, one row per ``log()`` call.

    The CSV file is created (and the header written) at construction time if
    ``fieldnames`` is provided — this ensures the file exists on disk
    immediately, which matters for tests that check file existence.

    If ``fieldnames`` is None, the header is inferred from the keys of the
    first ``log()`` call.

    Usage::

        logger = MetricLogger(save_dir / "metrics.csv", fieldnames=[...])
        logger.log({"step": 1000, "episode_return": 42.3, ...})
        logger.close()
    """

    def __init__(self, csv_path: str | pathlib.Path,
                 fieldnames: Optional[List[str]] = None) -> None:
        """
        Args:
            csv_path  : Destination path for the metrics CSV.
            fieldnames: Optional list of column names.  If provided, the CSV
                        header is written immediately.  Rows may contain a
                        subset of these keys; missing values are written as
                        ``nan``.  Extra keys in a row are silently ignored.
        """
        self.csv_path = pathlib.Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = fieldnames
        self._file = open(self.csv_path, "w", newline="")  # creates file immediately
        if fieldnames is not None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=fieldnames,
                extrasaction="ignore",
                restval="nan",
            )
            self._writer.writeheader()
            self._file.flush()
        else:
            self._writer = None

    def log(self, metrics: dict) -> None:
        """
        Append one row of metrics.

        Args:
            metrics: Dict mapping column names to scalar values.
        """
        if self._writer is None:
            # First call: infer schema from keys.
            self._fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
                restval="nan",
            )
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying CSV file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(agent, path: str | pathlib.Path, metadata: dict) -> None:
    """
    Persist agent network weights and training metadata to disk.

    Saves actor weights, critic weights, log_std, and optimiser state into a
    single ``.pt`` file using ``torch.save``.

    Args:
        agent   : PPOAgent instance (must have ``.actor``, ``.critic``,
                  ``.log_std``, and ``.optimizer`` attributes).
        path    : File path for the checkpoint, e.g. ``checkpoints/step_010000.pt``.
        metadata: Dict of scalar metadata embedded in the checkpoint
                  (e.g. step count, seed, eval_return_mean).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "log_std": agent.log_std.data.clone(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | pathlib.Path) -> dict:
    """
    Load a checkpoint from disk.

    Args:
        path: Path to a ``.pt`` checkpoint file produced by ``save_checkpoint``.

    Returns:
        Dictionary with keys:
            - ``actor_state_dict``
            - ``critic_state_dict``
            - ``log_std``
            - ``optimizer_state_dict``
            - ``metadata``
    """
    return torch.load(str(path), map_location="cpu")
