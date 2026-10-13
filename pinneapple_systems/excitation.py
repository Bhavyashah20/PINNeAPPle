"""Actuator excitation signals for system identification and surrogate training (numpy).

Every ``shape_*`` function maps normalised time ``tau`` in ``[0, 1]`` (an array) to a signal in ``[0, 1]``;
:func:`actuator_series` scales one to ``[lo, hi]`` and optionally rate-limits it with :func:`rate_limit`.
Random shapes take an ``rng`` (``numpy.random.Generator``) so a seed reproduces the signal.

Families: step, ramp, sinusoid, piecewise constant (random staircase), PRBS (pseudo-random binary sequence),
linear chirp, sawtooth, smoothed random walk, Gaussian bumps and an amplitude-modulated two-tone. Persistently
exciting inputs (PRBS, chirp/swept sine) are the standard identification signals of Ljung, *System Identification:
Theory for the User*, 2nd ed., Prentice Hall (1999), ch. 13.

    >>> import numpy as np
    >>> from pinneapple_systems.excitation import actuator_series
    >>> u = actuator_series("prbs", 500, lo=0.75, hi=1.25, seed=3, dwell=0.05)
    >>> set(np.unique(u)) <= {0.75, 1.25}
    True
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "SHAPES", "TRAIN_FAMILIES", "HELDOUT_FAMILIES",
    "shape_step", "shape_ramp", "shape_sinusoid", "shape_piecewise_constant", "shape_prbs", "shape_chirp",
    "shape_sawtooth", "shape_smoothed_random_walk", "shape_gaussian_bumps", "shape_am_two_tone",
    "rate_limit", "actuator_series",
]


def shape_step(tau, t0=0.3, **_):
    """Unit step at ``tau = t0``."""
    return (tau >= t0).astype(np.float64)


def shape_ramp(tau, t1=0.8, **_):
    """Linear ramp from 0 at ``tau = 0`` to 1 at ``tau = t1``, then held."""
    return np.clip(tau / t1, 0.0, 1.0)


def shape_sinusoid(tau, freq=2.0, phase=0.0, **_):
    """``0.5 + 0.5 sin(2 pi freq tau + phase)`` (``freq`` in cycles per unit ``tau``)."""
    return 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * tau + phase)


def shape_piecewise_constant(tau, rng=None, n_seg=5, **_):
    """Random staircase: ``n_seg`` levels ~ U(0, 1) between sorted random edges in ``[0.05, 0.95]``."""
    rng = rng or np.random.default_rng(0)
    edges = np.sort(rng.uniform(0.05, 0.95, n_seg - 1))
    levels = rng.uniform(0.0, 1.0, n_seg)
    return levels[np.searchsorted(edges, tau)].astype(np.float64)


def shape_prbs(tau, rng=None, dwell=0.08, **_):
    """Pseudo-random binary sequence in ``{0, 1}``.

    The clock period is ``hold = max(1, round(dwell * len(tau)))`` samples; at every clock tick (sample index a
    multiple of ``hold``) the level is redrawn by an independent fair coin flip, so it can only change on the clock.
    """
    rng = rng or np.random.default_rng(0)
    n = len(tau)
    hold = max(1, int(round(dwell * n)))
    out = np.empty(n)
    val = float(rng.integers(0, 2))
    for i in range(n):
        if i % hold == 0:
            val = float(rng.integers(0, 2))
        out[i] = val
    return out


def shape_chirp(tau, f0=0.5, f1=6.0, **_):
    """Linear chirp ``0.5 + 0.5 sin(phi)``, instantaneous frequency ``f0 + (f1 - f0) tau`` (cycles per unit tau).

    Equals ``0.5 + 0.5 * scipy.signal.chirp(tau, f0, 1, f1, method="linear", phi=-90)``.
    """
    ph = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) * tau ** 2)
    return 0.5 + 0.5 * np.sin(ph)


def shape_sawtooth(tau, freq=3.0, **_):
    """Rising sawtooth ``(freq * tau) mod 1``."""
    return (tau * freq) % 1.0


def shape_smoothed_random_walk(tau, rng=None, **_):
    """Gaussian random walk smoothed by a 15-sample moving average, rescaled to ``[0, 1]``."""
    rng = rng or np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0.0, 1.0, len(tau)))
    w = np.convolve(walk, np.ones(15) / 15.0, mode="same")
    w = w - w.min()
    return w / (w.max() + 1e-9)


def shape_gaussian_bumps(tau, rng=None, n_bumps=4, width=0.05, **_):
    """Sum of ``n_bumps`` Gaussians (centres ~ U(0.1, 0.9), amplitudes ~ U(0.5, 1), std ``width``), peak-normalised."""
    rng = rng or np.random.default_rng(0)
    centres = rng.uniform(0.1, 0.9, n_bumps)
    amps = rng.uniform(0.5, 1.0, n_bumps)
    v = np.zeros_like(tau, dtype=np.float64)
    for c, a in zip(centres, amps, strict=True):
        v += a * np.exp(-0.5 * ((tau - c) / width) ** 2)
    return np.clip(v / (v.max() + 1e-9), 0.0, 1.0)


def shape_am_two_tone(tau, f_carrier=5.0, f_mod=0.7, **_):
    """Amplitude modulation: ``(0.5 + 0.5 sin 2 pi f_carrier tau) (0.5 + 0.5 sin 2 pi f_mod tau)``."""
    return (0.5 + 0.5 * np.sin(2 * np.pi * f_carrier * tau)) * (0.5 + 0.5 * np.sin(2 * np.pi * f_mod * tau))


SHAPES: dict[str, Callable[..., np.ndarray]] = {
    "step": shape_step, "ramp": shape_ramp, "sinusoid": shape_sinusoid,
    "piecewise_constant": shape_piecewise_constant, "prbs": shape_prbs,
    "chirp": shape_chirp, "sawtooth": shape_sawtooth,
    "smoothed_random_walk": shape_smoothed_random_walk,
    "gaussian_bumps": shape_gaussian_bumps, "am_two_tone": shape_am_two_tone,
}
#: Classic identification inputs, and families with a different generating process (useful as held-out tests).
TRAIN_FAMILIES = ["step", "ramp", "sinusoid", "piecewise_constant", "prbs"]
HELDOUT_FAMILIES = ["chirp", "sawtooth", "smoothed_random_walk", "gaussian_bumps", "am_two_tone"]


def rate_limit(series: np.ndarray, tau_steps: float) -> np.ndarray:
    """First-order low-pass ``y[i] = y[i-1] + (x[i] - y[i-1]) / tau_steps`` (time constant ``tau_steps`` samples).

    Models an actuator lag: the output moves at most ``|x[i] - y[i-1]| / tau_steps`` per sample and never
    overshoots the input range. ``tau_steps <= 1`` returns the input unchanged.
    """
    if tau_steps <= 1:
        return series
    a = 1.0 / float(tau_steps)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = out[i - 1] + a * (series[i] - out[i - 1])
    return out


def actuator_series(shape: str | Callable[..., np.ndarray], n_steps: int, *, lo: float = 0.0, hi: float = 1.0,
                    seed: int = 0, rate_limit_steps: float = 0, **kwargs) -> np.ndarray:
    """Sample one excitation on ``n_steps`` uniform points of ``tau`` in ``[0, 1]``, scaled to ``[lo, hi]``.

    ``shape`` is a key of :data:`SHAPES` or a callable ``f(tau, rng=..., **kwargs) -> [0, 1]``; ``seed`` seeds the
    ``rng`` of the random families; ``rate_limit_steps > 1`` applies :func:`rate_limit`. Returns ``(n_steps,)`` float64.
    """
    fn = SHAPES[shape] if isinstance(shape, str) else shape
    tau = np.linspace(0.0, 1.0, n_steps)
    s = fn(tau, rng=np.random.default_rng(seed), **kwargs)
    u = (lo + (hi - lo) * s).astype(np.float64)
    return rate_limit(u, rate_limit_steps) if rate_limit_steps > 1 else u
