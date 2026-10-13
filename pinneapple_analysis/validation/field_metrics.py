"""PDEBench-style error metrics for gridded fields (NumPy, no torch).

Given a prediction and a reference on the same regular grid, report the
error the way neural-operator / surrogate benchmarks do, so a PINNeAPPle
model can be compared with published PDEBench numbers:

``rmse``      root-mean-square error over space, averaged over the rest
``nrmse``     ``rmse / rms(target)`` per sample/channel/time, then averaged
``max_error`` worst absolute pointwise error
``crmse``     error of the spatial *mean* (conserved-quantity error)
``brmse``     RMSE restricted to the boundary faces of the box
``frmse``     RMSE split into low / mid / high radial-wavenumber bands
``vrmse``     RMSE / std(target) -- The Well's headline metric (variance-scaled)
``pearson``   spatial Pearson correlation between prediction and target

Definitions follow PDEBench (Takamoto et al., NeurIPS 2022 Datasets &
Benchmarks; code ``pdebench/models/metrics.py``, MIT), adapted:

- **Full spectrum.** PDEBench's 2-D/3-D Fourier metric accumulates only the
  ``i, j < n/2`` quadrant of the FFT (negative frequencies are skipped), so
  half (2-D) or three quarters (3-D) of the spectral error is ignored. Here
  every FFT coefficient is binned by its integer radial wavenumber
  ``floor(|m|)``, with ``m`` the signed mode index, so nothing is dropped and
  Parseval holds: the shell energies sum to ``rmse^2`` (tested).
- **Normalisation.** ``frmse`` is reported per shell in the same units as
  ``rmse`` (``sqrt(shell_energy)``), then averaged within each band, instead
  of PDEBench's ``/n * L`` scaling, so the numbers are comparable across
  resolutions.
- **Boundary RMSE** averages over every boundary node once per face, as in
  PDEBench, for any dimension.

``vrmse`` and ``pearson`` follow The Well (Ohana et al., NeurIPS 2024 Datasets &
Benchmarks; ``the_well/benchmark/metrics/spatial.py``, BSD-2-Clause), so
numbers can be compared with its leaderboard: VRMSE = sqrt(MSE / Var(target)),
computed per sample/channel/time and then averaged. Unlike nRMSE it does not
reward predicting the mean of a field with a large offset.

Layout: arrays are ``(..., *space)`` -- any leading axes (batch, time,
channel) followed by ``ndim`` spatial axes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["FieldErrorReport", "field_error_metrics", "spectral_error_shells"]


@dataclass
class FieldErrorReport:
    rmse: float
    nrmse: float
    max_error: float
    crmse: float
    brmse: float
    frmse_low: float
    frmse_mid: float
    frmse_high: float
    bands: Tuple[int, int]
    vrmse: float = float("nan")
    pearson: float = float("nan")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _space_axes(ndim: int) -> Tuple[int, ...]:
    return tuple(range(-ndim, 0))


def spectral_error_shells(pred: np.ndarray, target: np.ndarray, ndim: int) -> np.ndarray:
    """Mean-square error per integer radial wavenumber shell.

    Returns ``(..., n_shells)`` with ``sum(axis=-1) == mean((pred-target)**2)``
    over the spatial axes (Parseval), shells ``0 .. floor(max |m|)``.
    """
    err = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    axes = _space_axes(ndim)
    shape = err.shape[-ndim:]
    n_total = int(np.prod(shape))
    E = np.abs(np.fft.fftn(err, axes=axes)) ** 2 / n_total ** 2   # sum(E) == mean(err^2)
    grids = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    shell = np.floor(np.sqrt(sum(g ** 2 for g in grids)) + 1e-9).astype(int)
    n_shells = int(shell.max()) + 1
    lead = E.shape[:-ndim]
    flatE = E.reshape(*lead, n_total)
    idx = shell.reshape(-1)
    out = np.zeros((*lead, n_shells))
    for s in range(n_shells):
        out[..., s] = flatE[..., idx == s].sum(axis=-1)
    return out


def _boundary_mse(err2: np.ndarray, ndim: int) -> np.ndarray:
    total, count = 0.0, 0
    for ax in _space_axes(ndim):
        for idx in (0, -1):
            face = np.take(err2, idx, axis=ax)
            total = total + face.reshape(*face.shape[: face.ndim - (ndim - 1)], -1).sum(axis=-1)
            count += int(np.prod(face.shape[face.ndim - (ndim - 1):])) if ndim > 1 else 1
    return total / count


def field_error_metrics(pred: np.ndarray, target: np.ndarray, *, ndim: int,
                        bands: Sequence[int] = (4, 12), eps: float = 1e-12) -> FieldErrorReport:
    """Compute :class:`FieldErrorReport` for ``pred`` vs ``target``.

    ``bands = (i_low, i_high)``: shells ``[0, i_low)`` are "low",
    ``[i_low, i_high)`` "mid", ``[i_high, ...)`` "high" (PDEBench defaults 4, 12).
    Every metric is computed per leading index and then averaged.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    if not 1 <= ndim <= pred.ndim:
        raise ValueError("ndim must be between 1 and pred.ndim")
    axes = _space_axes(ndim)
    err = pred - target
    err2 = err ** 2
    rmse_i = np.sqrt(err2.mean(axis=axes))
    nrm_i = np.sqrt((target ** 2).mean(axis=axes))
    crmse = np.sqrt(np.mean((pred.mean(axis=axes) - target.mean(axis=axes)) ** 2))
    brmse_i = np.sqrt(_boundary_mse(err2, ndim))
    shells = spectral_error_shells(pred, target, ndim)
    i_low, i_high = int(bands[0]), int(bands[1])
    per_shell = np.sqrt(shells)

    def band(a: int, b: int) -> float:
        sel = per_shell[..., a:b]
        return float(sel.mean()) if sel.shape[-1] else 0.0

    var_i = target.var(axis=axes)
    vrmse_i = np.sqrt(err2.mean(axis=axes) / (var_i + eps))
    pc = pred - pred.mean(axis=axes, keepdims=True)
    tc = target - target.mean(axis=axes, keepdims=True)
    pearson_i = (pc * tc).sum(axis=axes) / np.sqrt((pc ** 2).sum(axis=axes) * (tc ** 2).sum(axis=axes) + eps)

    return FieldErrorReport(
        rmse=float(rmse_i.mean()),
        nrmse=float((rmse_i / (nrm_i + eps)).mean()),
        max_error=float(np.abs(err).max()),
        crmse=float(crmse),
        brmse=float(brmse_i.mean()),
        frmse_low=band(0, i_low),
        frmse_mid=band(i_low, i_high),
        frmse_high=band(i_high, per_shell.shape[-1]),
        bands=(i_low, i_high),
        vrmse=float(vrmse_i.mean()),
        pearson=float(pearson_i.mean()),
    )
