"""Monotonicity constraints for regression surrogates: a hinge penalty on input-output derivatives.

Many physical responses are monotone in some inputs (a more conductive fin never raises a thermal resistance, more
airflow never lowers a pressure drop). When the effect is small compared with the fit error, a plain regression
gets its sign wrong on part of the input space even though the training data are monotone. Adding

    penalty = mean( relu( sign_j * d f / d x_j * scale_j + margin )^2 )

to the loss, evaluated on the training inputs and on random in-range inputs, enforces
``sign_j * d f / d x_j <= -margin`` wherever it is sampled. ``scale_j`` converts the derivative from the network's
(standardised) coordinates to the physical ones, so ``margin`` is in physical units.

    from pinneapple_neural.trainer.monotonic import monotonicity_penalty
    loss = mse + lam * monotonicity_penalty(model, z, cols=[7, 8], signs=[-1, -1], scale=s, margin=3e-4)
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

__all__ = ["monotonicity_penalty", "monotone_violation_share"]


def _out(model: Callable, z: torch.Tensor) -> torch.Tensor:
    y = model(z)
    return y.y if hasattr(y, "y") else y


def monotonicity_penalty(model: Callable, z: torch.Tensor, cols: Sequence[int], signs: Sequence[float], *,
                         scale: Sequence[float] | torch.Tensor | None = None, margin: float = 0.0,
                         output: int = 0, create_graph: bool = True) -> torch.Tensor:
    """Mean squared hinge on ``signs[j] * d out / d z[:, cols[j]] * scale[j] + margin`` (zero when every sampled
    derivative has the required sign with at least ``margin`` to spare).

    ``signs[j]`` is the required sign of the derivative (+1 increasing, -1 decreasing). The penalty is
    differentiable in the model parameters (``create_graph=True``), so it can be added to a training loss.
    """
    z = z.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(_out(model, z)[:, output].sum(), z, create_graph=create_graph)[0][:, list(cols)]
    s = torch.as_tensor(np.asarray(signs, dtype=np.float32), device=z.device)
    if scale is not None:
        g = g * torch.as_tensor(np.asarray(scale, dtype=np.float32), device=z.device)
    # the hinge is active when the derivative points against the required sign
    return torch.mean(torch.relu(-s * g + margin) ** 2)


def monotone_violation_share(model: Callable, z: torch.Tensor, cols: Sequence[int], signs: Sequence[float],
                             output: int = 0) -> list[float]:
    """Share of the rows of ``z`` where ``d out / d z[:, col]`` has the wrong sign, per constrained column."""
    z = z.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(_out(model, z)[:, output].sum(), z)[0][:, list(cols)].detach().cpu().numpy()
    return [float(np.mean(np.sign(g[:, j]) != np.sign(s))) for j, s in enumerate(signs)]
