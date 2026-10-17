"""Forecast skill against a reference forecast, with confidence intervals that respect serial correlation.

Forecasts verified from many start times along one or a few trajectories are not independent: two starts 20 time
units apart share most of their future. The intervals here come from a **moving-block bootstrap** over the start
times (Kuensch 1989), with blocks about as long as the correlation time of the errors.

    from pinneapple_analysis.verification.forecast_skill import skill_vs_lead
    res = skill_vs_lead(err_model, err_persistence, leads, block=10)   # errors: (starts, leads)
    res.horizon                     # last lead up to which the model is better with 95 % confidence
    res.skill_score[:, 0]           # 1 - E_model / E_ref per lead (lo / hi in columns 1, 2)

Statistics of time series from a simulation (time means and their intervals) use **batch means**
(:func:`batch_means_ci`), the usual estimator for correlated samples in turbulence statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["block_bootstrap_indices", "block_bootstrap_mean", "block_bootstrap_ratio", "batch_means_ci",
           "SkillResult", "skill_vs_lead"]


def block_bootstrap_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) resampled indices: moving blocks of ``block`` consecutive samples, concatenated and cut to n."""
    block = max(1, min(int(block), n))
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, (n_boot, nb))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return idx


def block_bootstrap_mean(d: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0,
                         level: float = 0.95) -> tuple[float, float, float]:
    """Mean of ``d`` along axis 0 and its ``level`` interval (moving-block bootstrap)."""
    d = np.asarray(d, float)
    idx = block_bootstrap_indices(len(d), block, n_boot, np.random.default_rng(seed))
    means = d[idx].mean(1)
    a = 100 * (1 - level) / 2
    return float(d.mean()), float(np.percentile(means, a)), float(np.percentile(means, 100 - a))


def block_bootstrap_ratio(a: np.ndarray, b: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0,
                          level: float = 0.95) -> tuple[float, float, float]:
    """Skill score 1 - mean(a) / mean(b) and its interval (paired moving-block bootstrap)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = block_bootstrap_indices(len(a), block, n_boot, np.random.default_rng(seed))
    vals = 1 - a[idx].mean(1) / b[idx].mean(1)
    q = 100 * (1 - level) / 2
    return float(1 - a.mean() / b.mean()), float(np.percentile(vals, q)), float(np.percentile(vals, 100 - q))


def batch_means_ci(x: np.ndarray, n_batches: int = 8, level: float = 0.95) -> tuple[float, float]:
    """Mean of a correlated time series and the half-width of its interval from ``n_batches`` batch means
    (Student t with n_batches - 1 degrees of freedom). The batches must be longer than the correlation time."""
    from scipy import stats
    x = np.asarray(x, float)
    m = np.array([b.mean() for b in np.array_split(x, n_batches)])
    t = stats.t.ppf(0.5 + level / 2, n_batches - 1)
    return float(x.mean()), float(t * m.std(ddof=1) / math.sqrt(n_batches))


@dataclass
class SkillResult:
    leads: np.ndarray               # (L,)
    diff: np.ndarray                # (L, 3): mean of E_model - E_ref per lead, lower, upper
    skill_score: np.ndarray         # (L, 3): 1 - E_model / E_ref, lower, upper
    horizon: float                  # last lead with diff upper bound < 0 at that lead and all earlier ones
    n_starts: int
    block: int

    def to_dict(self) -> dict:
        return {"leads": self.leads.tolist(), "diff": self.diff.tolist(), "skill_score": self.skill_score.tolist(),
                "horizon": self.horizon, "n_starts": self.n_starts, "block": self.block}


def skill_vs_lead(err_model: np.ndarray, err_ref: np.ndarray, leads: np.ndarray, block: int, *,
                  n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> SkillResult:
    """Paired comparison of two forecasts verified from the same starts. ``err_*``: (starts, leads), e.g. the MAE of
    each forecast at each lead. The **horizon** is the last lead up to which the model is better than the reference
    at every lead with ``level`` confidence (0 if it is not better at the first lead)."""
    em, er = np.asarray(err_model, float), np.asarray(err_ref, float)
    if em.shape != er.shape:
        raise ValueError(f"shapes differ: {em.shape} vs {er.shape}")
    leads = np.asarray(leads, float)
    diff = np.array([block_bootstrap_mean(em[:, j] - er[:, j], block, n_boot, seed + j, level)
                     for j in range(em.shape[1])])
    ss = np.array([block_bootstrap_ratio(em[:, j], er[:, j], block, n_boot, seed + j, level)
                   for j in range(em.shape[1])])
    better = diff[:, 2] < 0
    k = len(better) if better.all() else int(np.argmin(better))
    horizon = float(leads[k - 1]) if k > 0 else 0.0
    return SkillResult(leads, diff, ss, horizon, em.shape[0], int(block))
