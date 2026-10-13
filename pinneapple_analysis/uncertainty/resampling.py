"""Resampling confidence intervals for dependent data: moving-block and group bootstrap, skill scores, jackknife.

Forecast errors are rarely independent: consecutive time steps are autocorrelated and the cases of one event (a
storm, an experiment, a sequence) move together. Resampling single cases then gives confidence intervals that are
too narrow. The functions here resample whole blocks or whole groups instead. NumPy only.

* :func:`block_bootstrap` -- moving-block bootstrap (Kunsch 1989) for a statistic of a time series.
* :func:`group_bootstrap` -- cluster bootstrap that resamples whole groups (Davison & Hinkley 1997, sec. 3.8).
* :func:`skill_score` -- ``1 - E|e_model| / E|e_ref|`` with a paired bootstrap interval (blocks, groups or iid).
* :func:`jackknife` -- delete-one-block jackknife standard error (Efron & Tibshirani 1993, ch. 11).

Every bootstrap returns ``(estimate, lo, hi)``: the statistic on the full sample and the equal-tailed percentile
interval at ``level`` (Efron & Tibshirani 1993, ch. 13). Non-finite entries are dropped before resampling.

References
----------
Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*
17(3), 1217-1241.

Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["block_bootstrap", "group_bootstrap", "skill_score", "jackknife"]


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a.reshape(len(a), -1)).all(axis=1)
    return ok


def _interval(boots: np.ndarray, level: float) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    a = (1.0 - level) / 2.0
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1.0 - a))


def _block_indices(n_obs: int, block: int, n: int, rng: np.random.Generator):
    """Yield ``n`` moving-block resamples of ``range(n_obs)``: ``ceil(n_obs / block)`` blocks of ``block``
    consecutive indices with starts uniform on ``{0, ..., n_obs - block}``, concatenated and cut to ``n_obs``."""
    block = int(min(max(block, 1), n_obs))
    nb = -(-n_obs // block)
    starts = rng.integers(0, n_obs - block + 1, size=(n, nb))
    offs = np.arange(block)
    for s in starts:
        yield (s[:, None] + offs).ravel()[:n_obs]


def block_bootstrap(x, stat: Callable = np.mean, block: int = 20, n: int = 2000, seed: int = 0,
                    level: float = 0.95) -> tuple[float, float, float]:
    """Moving-block bootstrap of ``stat(x)`` for a serially dependent sample (Kunsch 1989).

    Each replicate concatenates ``ceil(N / block)`` overlapping blocks of ``block`` consecutive observations, with
    block starts drawn uniformly from ``{0, ..., N - block}``, and cuts the result to ``N``. ``block=1`` is the
    ordinary iid bootstrap. Choose ``block`` longer than the correlation time of ``x``.

    Parameters
    ----------
    x : array (N,) or (N, ...); resampling is along axis 0, rows with a non-finite entry are dropped.
    stat : function of an array like ``x`` returning a scalar (default: mean).
    block : block length in samples.
    n : number of bootstrap replicates.
    seed : seed of ``numpy.random.default_rng``.
    level : coverage of the equal-tailed percentile interval.

    Returns
    -------
    (estimate, lo, hi) : ``stat(x)`` and the percentile interval.
    """
    x = np.asarray(x, dtype=float)
    x = x[_finite_rows(x)]
    if len(x) == 0:
        raise ValueError("block_bootstrap: no finite samples")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[i]) for i in _block_indices(len(x), block, n, rng)], dtype=float)
    return (float(stat(x)), *_interval(boots, level))


def _group_members(groups: np.ndarray):
    _, inv = np.unique(groups, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    return np.split(order, np.cumsum(counts)[:-1]), inv, len(counts)


def group_bootstrap(x, groups, stat: Callable = np.mean, n: int = 2000, seed: int = 0,
                    level: float = 0.95) -> tuple[float, float, float]:
    """Cluster bootstrap of ``stat(x)``: resample whole groups with replacement.

    Cases inside one group (a storm, an event, a week, an experiment) are strongly dependent while groups are taken
    as independent, so each replicate draws ``G`` group labels with replacement out of the ``G`` groups and
    concatenates all their cases (Davison & Hinkley 1997, sec. 3.8). With one case per group this is the iid
    bootstrap.

    Parameters
    ----------
    x : array (N,) or (N, ...); rows with a non-finite entry are dropped (with their group label).
    groups : (N,) group label of every row (any sortable type). Groups are ordered as ``numpy.unique`` sorts them.
    stat, n, seed, level : as in :func:`block_bootstrap`.

    Returns
    -------
    (estimate, lo, hi)
    """
    x, g = np.asarray(x, dtype=float), np.asarray(groups)
    if len(g) != len(x):
        raise ValueError(f"groups has {len(g)} labels for {len(x)} samples")
    ok = _finite_rows(x)
    x, g = x[ok], g[ok]
    if len(x) == 0:
        raise ValueError("group_bootstrap: no finite samples")
    members, _, n_groups = _group_members(g)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_groups, size=(n, n_groups))
    boots = np.array([stat(x[np.concatenate([members[k] for k in p])]) for p in pick], dtype=float)
    return (float(stat(x)), *_interval(boots, level))


def skill_score(err_model, err_ref, *, block: int | None = None, groups=None, n: int = 2000, seed: int = 0,
                level: float = 0.95) -> tuple[float, float, float]:
    """Skill score ``1 - E|e_model| / E|e_ref|`` with a paired bootstrap interval.

    Positive means the model beats the reference (persistence, climatology, ...), 0 means no skill, 1 a perfect
    model: the generic skill score ``(A - A_ref) / (A_perfect - A_ref)`` with ``A`` the mean absolute error and
    ``A_perfect = 0`` (Wilks 2011, *Statistical Methods in the Atmospheric Sciences*, 3rd ed., ch. 8).
    Both error series are resampled with the same indices so their correlation is kept:

    * ``groups`` given: whole groups are resampled (:func:`group_bootstrap`);
    * else ``block`` given: moving blocks of that length (:func:`block_bootstrap`, Kunsch 1989);
    * else: iid cases (Efron & Tibshirani 1993).

    Parameters
    ----------
    err_model, err_ref : (N,) errors of the model and of the reference on the same cases (absolute values are
        taken). Cases where either is non-finite are dropped.
    block : moving-block length in cases (mutually exclusive with ``groups``).
    groups : (N,) group label per case.
    n, seed, level : as in :func:`block_bootstrap`.

    Returns
    -------
    (skill, lo, hi)
    """
    if block is not None and groups is not None:
        raise ValueError("give either block or groups, not both")
    em = np.abs(np.asarray(err_model, dtype=float)).ravel()
    er = np.abs(np.asarray(err_ref, dtype=float)).ravel()
    if em.shape != er.shape:
        raise ValueError(f"err_model and err_ref differ in length: {em.shape} vs {er.shape}")
    ok = np.isfinite(em) & np.isfinite(er)
    if groups is not None:
        g = np.asarray(groups).ravel()
        if len(g) != len(ok):
            raise ValueError(f"groups has {len(g)} labels for {len(ok)} cases")
        g = g[ok]
    em, er = em[ok], er[ok]
    if len(em) == 0:
        raise ValueError("skill_score: no finite pairs")
    est = float(1.0 - em.mean() / er.mean())
    rng = np.random.default_rng(seed)
    if groups is not None:
        _, inv, n_groups = _group_members(g)
        sm, sr = np.bincount(inv, em, n_groups), np.bincount(inv, er, n_groups)
        pick = rng.integers(0, n_groups, size=(n, n_groups))
        boots = 1.0 - sm[pick].sum(axis=1) / sr[pick].sum(axis=1)
    else:
        boots = np.array([1.0 - em[i].mean() / er[i].mean()
                          for i in _block_indices(len(em), 1 if block is None else block, n, rng)])
    return (est, *_interval(boots, level))


def jackknife(samples, fn: Callable, n_blocks: int = 20):
    """Delete-one-block jackknife estimate and standard error of ``fn(samples)``.

    ``samples`` (axis 0) is cut into ``n_blocks`` contiguous blocks (``numpy.array_split``); ``theta_i`` is ``fn``
    on the samples without block ``i``, and

        stderr = sqrt((n_blocks - 1) / n_blocks * sum_i (theta_i - mean(theta))**2)

    (Efron & Tibshirani 1993, eq. 11.5). Blocking makes the error honest for autocorrelated Monte Carlo chains when
    blocks are longer than the autocorrelation time; ``n_blocks = len(samples)`` is the classic delete-one jackknife.
    ``fn`` may return an array (e.g. a correlator per time separation): the error is element-wise.

    Returns
    -------
    (estimate, stderr) : ``fn(samples)`` on the full sample (not bias-corrected) and its jackknife standard error.
    """
    samples = np.asarray(samples)
    if not 2 <= n_blocks <= len(samples):
        raise ValueError(f"n_blocks must be in [2, {len(samples)}], got {n_blocks}")
    blocks = np.array_split(np.arange(len(samples)), n_blocks)
    full = fn(samples)
    jk = np.array([fn(np.delete(samples, b, axis=0)) for b in blocks])
    err = np.sqrt((n_blocks - 1) * ((jk - jk.mean(axis=0)) ** 2).mean(axis=0))
    return full, err
