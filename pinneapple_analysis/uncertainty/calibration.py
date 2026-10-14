"""Calibration metrics for probabilistic regression models.

A well-calibrated model should assign uncertainty estimates that match
empirical frequencies.  This module provides standard metrics for diagnosing
and reporting calibration quality.

References
----------
Kuleshov, V., Fenner, N. & Ermon, S. (2018). *Accurate Uncertainties for Deep
Learning Using Calibrated Regression*. ICML.

Levi, D. et al. (2022). *Evaluating and Calibrating Uncertainty Prediction in
Regression Tasks*. Sensors, 22(15).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

# Optional scipy import for precise normal CDF.
try:
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def _normal_cdf(z: Tensor) -> Tensor:
        return torch.from_numpy(_scipy_norm.cdf(z.cpu().numpy())).to(z.device)

    def _normal_ppf(p: float) -> float:
        return float(_scipy_norm.ppf(p))

    _HAS_SCIPY = True

except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

    def _normal_cdf(z: Tensor) -> Tensor:  # type: ignore[misc]
        """Approximation of the standard normal CDF using the error function."""
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    def _normal_ppf(p: float) -> float:  # type: ignore[misc]
        """Approximate inverse normal CDF via bisection (fallback)."""
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            cdf_mid = float(_normal_cdf(torch.tensor(mid)).item())
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


def _prepare(y_pred, y_true, y_std=None):
    """Float64 tensors of one shape; a clear error for mismatched shapes, NaN/Inf or a negative std (a NaN target
    used to count as "outside every interval" and silently changed the ECE)."""
    out = [torch.as_tensor(v.detach() if hasattr(v, "detach") else v, dtype=torch.float64).cpu()
           for v in ((y_pred, y_true) if y_std is None else (y_pred, y_true, y_std))]
    if any(t.shape != out[0].shape for t in out):
        raise ValueError(f"y_pred, y_true and y_std must have the same shape; got {[tuple(t.shape) for t in out]}")
    if out[0].numel() == 0:
        raise ValueError("calibration metrics need at least one sample")
    for name, t in zip(("y_pred", "y_true", "y_std"), out, strict=False):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} contains NaN or Inf; remove or impute those samples first")
    if y_std is not None:
        if bool((out[2] < 0).any()):
            raise ValueError("y_std must be non-negative")
        out[2] = out[2].clamp(min=1e-9)
    return out


class CalibrationMetrics:
    """Collection of calibration and sharpness metrics for regression UQ.

    All methods are static/class methods and accept plain ``Tensor`` inputs.
    No instantiation is required; use the class directly as a namespace:

    .. code-block:: python

        ece = CalibrationMetrics.expected_calibration_error(
            y_pred, y_true, y_std, n_bins=15
        )
        cov = CalibrationMetrics.coverage_at_level(y_pred, y_true, y_std, alpha=0.1)

    Notes
    -----
    Inputs are tensors or arrays of one shape, ``(N,)`` or ``(N, D)``; they are
    computed in float64. For multi-output models, metrics are computed
    element-wise and then averaged across output dimensions (so a calibrated
    model has ``coverage_at_level`` close to ``1 - alpha`` whatever ``D``).
    NaN or Inf inputs raise ``ValueError``.
    """

    # ------------------------------------------------------------------
    # Expected Calibration Error
    # ------------------------------------------------------------------

    @staticmethod
    def expected_calibration_error(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute the Expected Calibration Error (ECE) for regression.

        For each confidence level ``p`` in a uniform grid over ``(0, 1)``,
        the empirical coverage is computed and compared to the nominal
        coverage.  ECE is the area-weighted average absolute difference.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        n_bins:
            Number of equally-spaced confidence levels at which to evaluate
            calibration.

        Returns
        -------
        float
            ECE in ``[0, 1]``.  A perfectly calibrated model returns 0.
        """
        y_pred, y_true, y_std = _prepare(y_pred, y_true, y_std)

        # Probability integral transform of the standardised residuals: Uniform(0, 1) when calibrated.
        pit = _normal_cdf((y_true - y_pred) / y_std)

        levels = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1]  # exclude 0 and 1
        ece_sum = 0.0
        for p in levels.tolist():
            empirical = float((pit <= p).double().mean().item())
            ece_sum += abs(empirical - p)

        return ece_sum / n_bins

    # ------------------------------------------------------------------
    # Coverage at a given level
    # ------------------------------------------------------------------

    @staticmethod
    def coverage_at_level(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        alpha: float = 0.1,
    ) -> float:
        """Compute empirical coverage of the (1 - alpha) Gaussian interval.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        alpha:
            Miscoverage level; e.g. ``0.1`` targets 90 % coverage.

        Returns
        -------
        float
            Fraction of values (samples times output dimensions) that lie
            within their predicted interval. For a calibrated model this is
            close to ``1 - alpha`` whatever the number of outputs. (The joint
            coverage of all ``D`` outputs at once is a different quantity, about
            ``(1 - alpha)^D`` for independent errors.)
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        y_pred, y_true, y_std = _prepare(y_pred, y_true, y_std)

        z_alpha = _normal_ppf(1.0 - alpha / 2.0)
        lower = y_pred - z_alpha * y_std
        upper = y_pred + z_alpha * y_std
        within = (y_true >= lower) & (y_true <= upper)
        return float(within.double().mean().item())

    # ------------------------------------------------------------------
    # Calibration plot data
    # ------------------------------------------------------------------

    @staticmethod
    def calibration_plot_data(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        n_bins: int = 15,
    ) -> dict[str, list[float]]:
        """Compute data for a reliability / calibration diagram.

        Returns a dict with ``"expected"`` and ``"observed"`` confidence
        levels that can be passed directly to a plotting library.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        n_bins:
            Number of confidence levels.

        Returns
        -------
        dict with keys:
            ``"expected"`` — list of nominal confidence levels.
            ``"observed"`` — list of empirical coverages at each level.
            ``"ece"`` — scalar ECE (float wrapped in a single-element list for
            consistency).
        """
        y_pred, y_true, y_std = _prepare(y_pred, y_true, y_std)

        pit = _normal_cdf((y_true - y_pred) / y_std)
        levels = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1].tolist()

        observed: list[float] = []
        for p in levels:
            empirical = float((pit <= p).double().mean().item())
            observed.append(empirical)

        ece = sum(abs(o - e) for o, e in zip(observed, levels, strict=True)) / n_bins

        return {
            "expected": list(levels),
            "observed": observed,
            "ece": [ece],
        }

    # ------------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------------

    @staticmethod
    def sharpness(y_std: Tensor) -> float:
        """Compute sharpness: the mean predictive standard deviation.

        Lower sharpness indicates more confident (narrower) predictions.
        Sharpness should be interpreted alongside calibration — a model can
        be sharp but miscalibrated, or well-calibrated but not sharp.

        Parameters
        ----------
        y_std:
            Predictive standard deviation, shape ``(N,)`` or ``(N, D)``.

        Returns
        -------
        float
            Mean predictive std, averaged over all elements.
        """
        return float(torch.as_tensor(y_std.detach() if hasattr(y_std, "detach") else y_std,
                                     dtype=torch.float64).mean().item())

    # ------------------------------------------------------------------
    # Gaussian NLL
    # ------------------------------------------------------------------

    @staticmethod
    def nll_gaussian(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
    ) -> float:
        """Compute the mean Gaussian negative log-likelihood.

        Under the assumption that ``p(y | x) = N(y_pred, y_std²)``, the NLL
        for a single sample is:

        .. math::

            \\text{NLL} = \\frac{1}{2} \\log(2\\pi\\sigma^2)
                        + \\frac{(y - \\mu)^2}{2\\sigma^2}

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.

        Returns
        -------
        float
            Mean NLL over all samples and output dimensions.
        """
        y_pred, y_true, y_std = _prepare(y_pred, y_true, y_std)

        var = y_std ** 2
        nll = 0.5 * (torch.log(2.0 * math.pi * var) + ((y_true - y_pred) ** 2) / var)
        return float(nll.mean().item())
