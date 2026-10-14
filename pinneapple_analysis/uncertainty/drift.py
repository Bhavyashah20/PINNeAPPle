"""Drift monitoring of a deployed surrogate: accuracy, calibration, input shift and residual trend in one report.

A surrogate trained on one input distribution degrades silently when the operating point moves. Given a batch of new
predictions with their measured (or high-fidelity) truth, :func:`assess_drift` computes

* the normalised RMSE ``sqrt(mean(r^2)) / std(y_true)`` and the bias ``mean(r)`` of the residuals
  ``r = y_pred - y_true``, with a t-type test of ``bias = 0``;
* the empirical coverage of the central Gaussian interval ``y_pred +- z sigma`` at ``level``, compared with the
  binomial standard error ``sqrt(level (1 - level) / n)`` (Gneiting, Balabdaoui & Raftery 2007);
* per input feature, the two-sample Kolmogorov-Smirnov statistic and p-value between the reference (training) inputs
  and the new inputs (``scipy.stats.ks_2samp``; Massey 1951), Bonferroni-corrected over the features;
* the fraction of new inputs outside the training box ``bounds``;
* a Mann-Kendall trend test of the residuals against their order of arrival (Mann 1945; Kendall 1975), computed as
  Kendall's tau between the time index and the residual (``scipy.stats.kendalltau``).

The decision is a screening heuristic, not a statistical guarantee; every threshold is a field of
:class:`DriftThresholds`. A *flag* is raised by each of: a significant bias, an nRMSE above ``nrmse_review``, an
interval coverage more than ``coverage_z`` binomial standard errors below ``level``, a KS p-value below
``ks_alpha / d`` on any feature, an out-of-range fraction above ``oob_review`` and a significant residual trend.
The decision is ``"retrain"`` when a hard limit is crossed (nRMSE above ``nrmse_retrain`` or out-of-range fraction
above ``oob_retrain``) or when ``n_flags_retrain`` or more flags are raised, ``"review"`` with at least one flag, and
``"ok"`` otherwise.

References
----------
Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *JASA* 46(253), 68-78.

Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica* 13(3), 245-259.

Kendall, M. G. (1975). *Rank Correlation Methods*, 4th ed. Griffin.

Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
*J. R. Stat. Soc. B* 69(2), 243-268.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["DriftThresholds", "DriftReport", "assess_drift"]


@dataclass(frozen=True)
class DriftThresholds:
    """Thresholds of the ``ok`` / ``review`` / ``retrain`` decision of :func:`assess_drift`.

    nrmse_review, nrmse_retrain : nRMSE (RMSE / std of the truth) that raises a flag / forces retraining.
    bias_z : a flag when ``|mean(r)| > bias_z * std(r) / sqrt(n)``.
    coverage_z : a flag when coverage ``< level - coverage_z * sqrt(level (1 - level) / n)``.
    ks_alpha : family-wise level of the per-feature KS tests (Bonferroni: each feature at ``ks_alpha / d``).
    oob_review, oob_retrain : out-of-range fraction that raises a flag / forces retraining.
    trend_alpha : level of the Mann-Kendall trend test.
    n_flags_retrain : number of simultaneous flags that turns ``review`` into ``retrain``.
    """
    nrmse_review: float = 0.2
    nrmse_retrain: float = 0.5
    bias_z: float = 3.0
    coverage_z: float = 3.0
    ks_alpha: float = 0.01
    oob_review: float = 0.01
    oob_retrain: float = 0.10
    trend_alpha: float = 0.01
    n_flags_retrain: int = 3


@dataclass
class DriftReport:
    """Result of :func:`assess_drift`. Per-output arrays have shape ``(m,)``; per-feature arrays ``(d,)``."""
    n: int
    nrmse: np.ndarray
    bias: np.ndarray
    bias_z: np.ndarray
    level: float
    coverage: np.ndarray | None = None
    ks_stat: np.ndarray | None = None
    ks_pvalue: np.ndarray | None = None
    out_of_bounds: float | None = None
    trend_tau: np.ndarray | None = None
    trend_pvalue: np.ndarray | None = None
    decision: str = "ok"
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line per quantity, then the decision and its reasons."""
        lines = [f"n = {self.n}", f"nRMSE = {np.array2string(self.nrmse, precision=4)}",
                 f"bias = {np.array2string(self.bias, precision=4)}"]
        if self.coverage is not None:
            lines.append(f"coverage@{self.level:g} = {np.array2string(self.coverage, precision=3)}")
        if self.ks_pvalue is not None:
            lines.append(f"KS p-value = {np.array2string(self.ks_pvalue, precision=3)}")
        if self.out_of_bounds is not None:
            lines.append(f"out of training bounds = {self.out_of_bounds:.3%}")
        if self.trend_pvalue is not None:
            lines.append(f"trend p-value = {np.array2string(self.trend_pvalue, precision=3)}")
        lines.append(f"decision: {self.decision}" + (f" ({'; '.join(self.reasons)})" if self.reasons else ""))
        return "\n".join(lines)


def _as_2d(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {a.shape}")
    return a


def assess_drift(y_true, y_pred, sigma=None, X_ref=None, X_new=None, bounds=None, level: float = 0.95,
                 thresholds: DriftThresholds | None = None) -> DriftReport:
    """Check a batch of surrogate predictions for accuracy loss, miscalibration, input shift and residual trend.

    Parameters
    ----------
    y_true, y_pred : arrays (n,) or (n, m), in order of arrival (the trend test uses this order).
    sigma : optional predictive standard deviation, same shape as ``y_pred``; enables the coverage check.
    X_ref : optional reference (training) inputs (n_ref, d); with ``X_new`` enables the per-feature KS tests.
    X_new : optional inputs (n_new, d) of the new batch.
    bounds : optional training box, a tuple ``(lo, hi)`` of arrays (d,) or an array (d, 2); with ``X_new`` gives the
        out-of-range fraction. If omitted and ``X_ref`` is given, the min/max of ``X_ref`` is used.
    level : central coverage of the Gaussian interval ``y_pred +- z sigma``, ``z = Phi^-1((1 + level) / 2)``.
    thresholds : :class:`DriftThresholds` (defaults when omitted).

    Returns
    -------
    DriftReport
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    th = thresholds or DriftThresholds()
    yt, yp = _as_2d(y_true, "y_true"), _as_2d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} differ")
    n = yt.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples")
    r = yp - yt
    rmse = np.sqrt(np.mean(r ** 2, axis=0))
    scale = np.std(yt, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nrmse = np.where(scale > 0, rmse / scale, np.nan)
        bias = r.mean(axis=0)
        sd = r.std(axis=0, ddof=1)
        bias_z = np.where(sd > 0, bias / (sd / np.sqrt(n)), np.where(bias == 0, 0.0, np.inf))
    rep = DriftReport(n=n, nrmse=nrmse, bias=bias, bias_z=bias_z, level=level)
    flags: list[str] = []
    hard: list[str] = []

    if np.any(np.abs(bias_z) > th.bias_z):
        flags.append(f"bias significant (|z| = {np.max(np.abs(bias_z)):.1f})")
    if np.any(nrmse > th.nrmse_retrain):
        hard.append(f"nRMSE {np.nanmax(nrmse):.3f} > {th.nrmse_retrain}")
    elif np.any(nrmse > th.nrmse_review):
        flags.append(f"nRMSE {np.nanmax(nrmse):.3f} > {th.nrmse_review}")

    if sigma is not None:
        sg = _as_2d(sigma, "sigma")
        if sg.shape != yp.shape:
            raise ValueError(f"sigma {sg.shape} and y_pred {yp.shape} differ")
        z = stats.norm.ppf(0.5 + level / 2.0)
        rep.coverage = np.mean(np.abs(r) <= z * sg, axis=0)
        floor = level - th.coverage_z * np.sqrt(level * (1.0 - level) / n)
        if np.any(rep.coverage < floor):
            flags.append(f"coverage {np.min(rep.coverage):.3f} < {floor:.3f}")

    Xn = None if X_new is None else _as_2d(X_new, "X_new")
    if X_ref is not None and Xn is not None:
        Xr = _as_2d(X_ref, "X_ref")
        if Xr.shape[1] != Xn.shape[1]:
            raise ValueError(f"X_ref has {Xr.shape[1]} features, X_new {Xn.shape[1]}")
        res = [stats.ks_2samp(Xr[:, j], Xn[:, j]) for j in range(Xr.shape[1])]
        rep.ks_stat = np.array([q.statistic for q in res])
        rep.ks_pvalue = np.array([q.pvalue for q in res])
        alpha = th.ks_alpha / Xr.shape[1]
        if np.any(rep.ks_pvalue < alpha):
            j = int(np.argmin(rep.ks_pvalue))
            flags.append(f"input shift in feature {j} (KS p = {rep.ks_pvalue[j]:.2g})")
        if bounds is None:
            bounds = (Xr.min(axis=0), Xr.max(axis=0))
    if bounds is not None and Xn is not None:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
        else:
            b = np.asarray(bounds, dtype=float).reshape(-1, 2)
            lo, hi = b[:, 0], b[:, 1]
        outside = np.any((Xn < lo) | (Xn > hi), axis=1)
        rep.out_of_bounds = float(outside.mean())
        if rep.out_of_bounds > th.oob_retrain:
            hard.append(f"{rep.out_of_bounds:.1%} of inputs outside training bounds")
        elif rep.out_of_bounds > th.oob_review:
            flags.append(f"{rep.out_of_bounds:.1%} of inputs outside training bounds")

    if n >= 3:
        t = np.arange(n)
        res = [stats.kendalltau(t, r[:, k]) for k in range(r.shape[1])]
        rep.trend_tau = np.array([q.statistic for q in res])
        rep.trend_pvalue = np.array([1.0 if not np.isfinite(q.pvalue) else q.pvalue for q in res])
        if np.any(rep.trend_pvalue < th.trend_alpha):
            flags.append(f"residual trend (Mann-Kendall p = {np.min(rep.trend_pvalue):.2g})")

    rep.reasons = hard + flags
    if hard or len(flags) >= th.n_flags_retrain:
        rep.decision = "retrain"
    elif flags:
        rep.decision = "review"
    return rep
