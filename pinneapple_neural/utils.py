"""Small helpers shared by trainers, solvers and examples.

``as_tensor`` turns whatever a model's ``forward`` returns (a plain Tensor, ``PINNOutput``, ``ModelOutput``,
``OperatorOutput``, ``GraphOutput``, ``AEOutput``, a dict of fields or a ``(prediction, extras)`` tuple) into the
prediction Tensor.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

__all__ = ["OUTPUT_TENSOR_ATTRS", "as_tensor"]

#: Attribute names searched, in this order, on a model-output object.
OUTPUT_TENSOR_ATTRS: tuple[str, ...] = ("y", "pred", "out", "logits", "recon", "x_hat", "y_hat", "x")


def _join_fields(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Join per-field tensors along the last dim: ``(N,)`` fields become columns, ``(N, k)`` fields are concatenated."""
    cols = [v.unsqueeze(-1) if v.ndim <= 1 else v for v in values]
    return torch.cat(cols, dim=-1)


def _is_tensor_seq(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(torch.is_tensor(t) for t in v)


def as_tensor(out: Any, *, attrs: Sequence[str] = OUTPUT_TENSOR_ATTRS) -> torch.Tensor:
    """Return the prediction Tensor of a model output.

    Rules, in order:

    1. a Tensor is returned unchanged;
    2. a dict of Tensors (one per field) is joined along the last dim, keeping insertion order
       (``(N,)`` values become columns, ``(N, k)`` values are concatenated);
    3. a tuple or list returns ``as_tensor`` of its first element (the ``(prediction, extras)`` convention);
    4. otherwise the first attribute in ``attrs`` (default ``y, pred, out, logits, recon, x_hat, y_hat, x``)
       that holds a Tensor is returned; an attribute holding a list of Tensors (``PINNOutput.y`` with one
       Tensor per field) is joined as in rule 2.

    Raises ``TypeError`` when no Tensor can be found.
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, Mapping):
        vals = list(out.values())
        if vals and all(torch.is_tensor(v) for v in vals):
            return _join_fields(vals)
        raise TypeError(f"dict model output must hold only Tensors, got {[type(v).__name__ for v in vals]}")
    if isinstance(out, (tuple, list)):
        if not out:
            raise TypeError("empty tuple/list model output")
        return as_tensor(out[0], attrs=attrs)
    for name in attrs:
        val = getattr(out, name, None)
        if torch.is_tensor(val):
            return val
        if _is_tensor_seq(val):
            return _join_fields(val)
    raise TypeError(f"cannot extract a Tensor from {type(out).__name__}: no Tensor attribute among {list(attrs)}")
