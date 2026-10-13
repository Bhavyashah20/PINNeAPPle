"""Posterior-matching metrics for generative / probabilistic inverse solvers (PosteriorBench).

A solver whose posterior *mean* is right can still have the wrong posterior: too narrow (the
classic MC-dropout failure), too wide, or with the wrong spatial texture. PosteriorBench
(neuraloperator/PosteriorBench, MIT; Wang et al., "PosteriorBench: From Point Estimates to
Posterior Matching in Evaluating Generative Inverse Solvers", NeurIPS 2026, arXiv:2609.20794)
compares predicted and reference posterior *ensembles* of 2D fields with:

- relative L2 of the mean and of the pointwise standard deviation;
- MMD with a multi-scale Gaussian kernel (bandwidths 0.2..5 x the median cross distance);
- sliced Wasserstein distance with Gaussian-random-field projection directions (smooth
  directions, drawn in the DCT domain) instead of isotropic white-noise directions;
- geometric mean of the relative error of the radially averaged power spectrum.

The implementation follows ``posteriorbench/evaluation/metrics.py`` (MIT, Copyright the
PosteriorBench authors); all quantities accept weighted ensembles (e.g. importance-sampled
reference posteriors). Its datasets are on the Hugging Face Hub (``pinneapple_catalog``:
``posteriorbench``).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.fft import idctn


def normalize_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"weights must be {n} finite non-negative numbers with a positive sum")
    return w / w.sum()


def _sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0.0)


def weighted_mmd(pred, ref, pred_weights=None, ref_weights=None) -> float:
    x = np.asarray(pred, np.float64).reshape(len(pred), -1)
    y = np.asarray(ref, np.float64).reshape(len(ref), -1)
    wx, wy = normalize_weights(pred_weights, len(x)), normalize_weights(ref_weights, len(y))
    dxx, dyy, dxy = _sq_dist(x, x), _sq_dist(y, y), _sq_dist(x, y)
    bw = float(np.median(dxy)) or 1.0
    k = lambda d: sum(np.exp(-d / (s * bw)) for s in (0.2, 0.5, 1.0, 2.0, 5.0)) / 5.0
    v = wx @ k(dxx) @ wx + wy @ k(dyy) @ wy - 2 * wx @ k(dxy) @ wy
    return float(np.sqrt(max(v, 0.0)))


def weighted_wasserstein_1d(x, y, wx=None, wy=None) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    wx, wy = normalize_weights(wx, len(x)), normalize_weights(wy, len(y))
    ox, oy = np.argsort(x), np.argsort(y)
    x, wx, y, wy = x[ox], wx[ox], y[oy], wy[oy]
    vals = np.unique(np.concatenate([x, y]))
    if len(vals) < 2:
        return 0.0
    cx = np.concatenate([[0.0], np.cumsum(wx)])[np.searchsorted(x, vals[:-1], side="right")]
    cy = np.concatenate([[0.0], np.cumsum(wy)])[np.searchsorted(y, vals[:-1], side="right")]
    return float(np.sum(np.abs(cx - cy) * np.diff(vals)))


def grf_direction(h: int, w: int, rng: np.random.Generator, alpha: float = 2.0, tau: float = 3.0) -> np.ndarray:
    """Unit-norm smooth random direction: GRF with spectrum ~ (pi^2 |k|^2 + tau^2)^(-alpha/2)."""
    ky, kx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2)
    c = np.sqrt(h * w) * coef * rng.normal(size=(h, w))
    c[0, 0] = 0.0  # zero-mean direction
    d = idctn(c, type=2, norm="ortho")
    return d / (np.linalg.norm(d) + 1e-12)


def sliced_wasserstein(pred, ref, pred_weights=None, ref_weights=None, n_projections: int = 128, seed: int = 0,
                       alpha: float = 2.0, tau: float = 3.0) -> float:
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    _, h, w = pred.shape
    x, y = pred.reshape(len(pred), -1), ref.reshape(len(ref), -1)
    rng = np.random.default_rng(seed)
    return float(np.mean([
        weighted_wasserstein_1d(x @ d, y @ d, pred_weights, ref_weights)
        for d in (grf_direction(h, w, rng, alpha, tau).ravel() for _ in range(n_projections))
    ]))


def radial_power_spectrum(samples, weights=None) -> np.ndarray:
    samples = np.asarray(samples, np.float64)
    w = normalize_weights(weights, len(samples))
    _, h, wd = samples.shape
    fy, fx = np.fft.fftshift(np.fft.fftfreq(h)), np.fft.fftshift(np.fft.fftfreq(wd))
    gx, gy = np.meshgrid(fx, fy)
    step = 1 / min(h, wd)
    bins = np.arange(0, 0.5 + step, step)
    idx = np.digitize(np.sqrt(gx ** 2 + gy ** 2).ravel(), bins) - 1
    nb = len(bins) - 1
    ok = (idx >= 0) & (idx < nb)
    counts = np.maximum(np.bincount(idx[ok], minlength=nb), 1)
    spectra = [np.bincount(idx[ok], weights=(np.abs(np.fft.fftshift(np.fft.fft2(s))) ** 2).ravel()[ok],
                           minlength=nb) / counts for s in samples]
    return np.average(np.asarray(spectra), axis=0, weights=w)


def spectral_relative_geomean(pred_spectrum, ref_spectrum, eps: float = 1e-10) -> float:
    rel = np.abs(pred_spectrum - ref_spectrum) / (np.abs(ref_spectrum) + eps)
    ok = np.isfinite(rel) & np.isfinite(ref_spectrum) & (ref_spectrum > eps)
    return float(np.exp(np.mean(np.log(rel[ok] + eps)))) if ok.any() else float("nan")


def evaluate_posterior(pred, ref, pred_weights=None, ref_weights=None, *, n_projections: int = 128,
                       seed: int = 0, swd_mean: Optional[float] = None, swd_std: Optional[float] = None) -> Dict[str, float]:
    """Compare a predicted posterior ensemble (N,H,W) with a reference one (M,H,W). Lower is better."""
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    if pred.ndim != 3 or ref.ndim != 3 or pred.shape[1:] != ref.shape[1:]:
        raise ValueError(f"expected [N,H,W] ensembles on the same grid, got {pred.shape} and {ref.shape}")
    if not (np.isfinite(pred).all() and np.isfinite(ref).all()):
        raise ValueError("ensembles must be finite")
    wp, wr = normalize_weights(pred_weights, len(pred)), normalize_weights(ref_weights, len(ref))
    mp, mr = np.average(pred, 0, wp), np.average(ref, 0, wr)
    sp = np.sqrt(np.average((pred - mp) ** 2, 0, wp))
    sr = np.sqrt(np.average((ref - mr) ** 2, 0, wr))
    sw_p, sw_r = pred, ref
    if swd_mean is not None:
        sw_p, sw_r = (pred - swd_mean) / swd_std, (ref - swd_mean) / swd_std
    return {
        "mean_rel_l2": float(np.linalg.norm(mp - mr) / (np.linalg.norm(mr) + 1e-12)),
        "std_rel_l2": float(np.linalg.norm(sp - sr) / (np.linalg.norm(sr) + 1e-12)),
        "mmd": weighted_mmd(pred, ref, wp, wr),
        "swd": sliced_wasserstein(sw_p, sw_r, wp, wr, n_projections=n_projections, seed=seed),
        "spectral_rel_geomean": spectral_relative_geomean(radial_power_spectrum(pred, wp), radial_power_spectrum(ref, wr)),
    }


# ---------------------------------------------------------------------------
# CRPS -- continuous ranked probability score (scoring rule for probabilistic
# surrogates; the standard ensemble score in Earth-2 Studio / WeatherBench).
# CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx; lower is better, it reduces
# to |x - y| for a deterministic forecast and is in the units of y.
# ---------------------------------------------------------------------------

def crps_gaussian(mu, sigma, y) -> np.ndarray:
    """Closed form for a Gaussian predictive N(mu, sigma^2) (Gneiting et al.,
    Mon. Weather Rev. 133 (2005) 1098-1118, Eq. 5). Elementwise."""
    from scipy.stats import norm
    mu, sigma, y = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sigma, float), np.asarray(y, float))
    shape = mu.shape
    mu, sigma, y = (np.atleast_1d(a).ravel() for a in (mu, sigma, y))
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")
    out = np.abs(y - mu)
    pos = sigma > 0
    z = (y[pos] - mu[pos]) / sigma[pos]
    out[pos] = sigma[pos] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return out.reshape(shape)


def crps_ensemble(samples, y, *, fair: bool = False) -> np.ndarray:
    """Ensemble CRPS, ``E|X - y| - 0.5 E|X - X'|`` with members on axis 0
    (Gneiting & Raftery, JASA 102 (2007) 359-378, Eq. 21). ``fair=True``
    uses the M(M-1) normalisation of the spread term (Ferro, QJRMS 2014),
    unbiased for the CRPS of the distribution the members are drawn from.
    Computed in O(M log M) per point through the sorted-member identity."""
    x = np.sort(np.asarray(samples, float), axis=0)
    m = x.shape[0]
    if m < 1 or (fair and m < 2):
        raise ValueError("not enough ensemble members")
    y = np.asarray(y, float)
    skill = np.abs(x - y[None]).mean(axis=0)
    i = np.arange(1, m + 1).reshape((m,) + (1,) * (x.ndim - 1))
    # sum_{i,j} |x_i - x_j| = 2 sum_i (2i - m - 1) x_(i)
    pair_sum = 2 * ((2 * i - m - 1) * x).sum(axis=0)
    spread = pair_sum / (m * (m - 1) if fair else m * m)
    return skill - 0.5 * spread
