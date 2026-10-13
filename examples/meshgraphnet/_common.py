"""Backward-compatible re-exports for the MeshGraphNet examples.

The helpers now live in the library:

* DeepMind ``meshgraphnets`` download, TFRecord reader and conversion: :mod:`pinneapple_data.datasets.deepmind_mgn`
* training and rollout evaluation: :mod:`pinneapple_neural.architectures.graphnn.mgn_dynamics`
* device choice: :func:`pinneapple_neural.trainer.parallel.best_device`
"""
from __future__ import annotations

import torch

from pinneapple_data.datasets.deepmind_mgn import (  # noqa: F401
    CYLINDER_TYPE_MAP,
    DM_URL,
    cylinder_flow_to_tensors,
    download_prefix,
    read_trajectories,
)
from pinneapple_neural.architectures.graphnn.mgn_dynamics import (  # noqa: F401
    eval_rollout,
    train_dynamics,
)
from pinneapple_neural.trainer.parallel import best_device


def pick_device() -> torch.device:
    """CUDA if available, else CPU (tiny graphs: MPS launch latency dominates)."""
    return best_device(allow_mps=False)
