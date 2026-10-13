"""Dimensionless numbers that tell which physics a case is in.

Every function takes plain floats (SI units) and returns a float, so the result can go straight into the feature
dictionary that the assumption auditor, the router and the knowledge graph read (``{"Re": ..., "Ma": ...}``).
:func:`regime_features` computes every number whose inputs are given and skips the rest.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

__all__ = ["reynolds", "mach", "peclet", "knudsen", "richardson", "rayleigh", "grashof", "prandtl", "biot", "froude",
           "strouhal", "dominant_frequency", "damkohler", "roughness_reynolds", "regime_features", "FEATURE_NAMES"]

FEATURE_NAMES = {
    "Re": "Reynolds number (inertia / viscosity)",
    "Ma": "Mach number (flow speed / speed of sound)",
    "Pe": "Peclet number (advection / diffusion)",
    "Kn": "Knudsen number (mean free path / length)",
    "Ri": "Richardson number (buoyancy / shear)",
    "Ra": "Rayleigh number (buoyancy-driven convection)",
    "Gr": "Grashof number (buoyancy / viscosity)",
    "Pr": "Prandtl number (momentum / thermal diffusivity)",
    "Bi": "Biot number (surface / internal heat transfer)",
    "Fr": "Froude number (inertia / gravity)",
    "St": "Strouhal number (oscillation / advection time)",
    "Da": "Damkohler number (reaction / transport rate)",
    "beta_dT": "Boussinesq parameter beta * delta T",
    "dT_over_T": "relative temperature difference",
    "strain": "maximum strain",
    "stress_ratio": "maximum stress / yield stress",
    "alpha_deg": "angle of attack in degrees",
    "H_over_L": "depth / horizontal length scale",
    "eps_plus": "roughness Reynolds number (roughness height in wall units)",
}


def reynolds(velocity: float, length: float, nu: float | None = None, *, rho: float | None = None,
             mu: float | None = None) -> float:
    """Re = U L / nu (or rho U L / mu)."""
    if nu is None:
        if rho is None or mu is None:
            raise ValueError("give nu, or rho and mu")
        nu = mu / rho
    return abs(velocity) * length / nu


def mach(velocity: float, sound_speed: float | None = None, *, gamma: float = 1.4, R: float = 287.05,
         T: float | None = None) -> float:
    """Ma = U / a, with a given or a = sqrt(gamma R T) for an ideal gas."""
    if sound_speed is None:
        if T is None:
            raise ValueError("give sound_speed or T")
        sound_speed = math.sqrt(gamma * R * T)
    return abs(velocity) / sound_speed


def peclet(velocity: float, length: float, diffusivity: float) -> float:
    """Pe = U L / D (mass) or U L / alpha (heat)."""
    return abs(velocity) * length / diffusivity


def knudsen(mean_free_path: float, length: float) -> float:
    return mean_free_path / length


def grashof(g: float, beta: float, dT: float, length: float, nu: float) -> float:
    return g * beta * abs(dT) * length ** 3 / nu ** 2


def prandtl(nu: float, alpha: float) -> float:
    return nu / alpha


def rayleigh(g: float, beta: float, dT: float, length: float, nu: float, alpha: float) -> float:
    return grashof(g, beta, dT, length, nu) * prandtl(nu, alpha)


def richardson(g: float, beta: float, dT: float, length: float, velocity: float) -> float:
    """Ri = Gr / Re^2 = g beta dT L / U^2."""
    return g * beta * abs(dT) * length / max(velocity ** 2, 1e-300)


def biot(h: float, length: float, k: float) -> float:
    return h * length / k


def froude(velocity: float, length: float, g: float = 9.81) -> float:
    return abs(velocity) / math.sqrt(g * length)


def strouhal(frequency: float, length: float, velocity: float) -> float:
    return frequency * length / abs(velocity)


def dominant_frequency(signal, dt: float = 1.0, discard: float = 0.5, window: bool = True, refine: bool = True,
                       pad: int = 1) -> float:
    """Frequency (1 / time unit of ``dt``) of the strongest spectral peak of a sampled signal, e.g. the
    vortex-shedding frequency of a wake probe or a lift history, for :func:`strouhal`.

    The first ``discard`` fraction of the record (start-up transient) is dropped and the mean removed; the rest is
    optionally multiplied by a Hann window (Harris, Proc. IEEE 66, 1978) and zero-padded to ``pad`` times its
    length, and the largest non-DC bin of ``|rfft|`` is taken. ``refine`` fits a parabola through the logarithms of
    the peak bin and its two neighbours (Gaussian interpolation; Smith & Serra, "PARSHL", ICMC 1987), which brings the
    estimate to a small fraction of a bin for a Hann-windowed sinusoid. Returns 0.0 for records too short to have
    a non-DC bin.
    """
    x = np.asarray(signal).ravel()
    if not np.issubdtype(x.dtype, np.floating):     # keep float32 records in float32, as numpy's FFT does
        x = x.astype(np.float64)
    x = x[int(discard * len(x)):]
    x = x - x.mean()
    if window:
        x = x * np.hanning(len(x))
    n = max(int(pad), 1) * len(x)
    spec = np.abs(np.fft.rfft(x, n))
    if len(spec) <= 2:
        return 0.0
    k: float = int(np.argmax(spec[1:]) + 1)
    if refine and 1 <= k < len(spec) - 1:
        a, b, g = (np.log(spec[i] + 1e-30) for i in (k - 1, k, k + 1))
        den = a - 2 * b + g
        if den != 0:
            k = k + 0.5 * (a - g) / den
    return float(k * (1.0 / (n * dt)))  # same arithmetic as k * np.fft.rfftfreq(n, dt)[1]


def damkohler(reaction_rate: float, length: float, velocity: float) -> float:
    return reaction_rate * length / abs(velocity)


def roughness_reynolds(re: float, eps_d: float, f: float | None = None) -> float:
    """eps+ = Re (eps / D) sqrt(f / 8): roughness height in wall units. ``f`` (Darcy) defaults to the Haaland
    estimate, so no solution is needed."""
    if f is None:
        f = (-1.8 * math.log10((eps_d / 3.7) ** 1.11 + 6.9 / re)) ** -2
    return re * eps_d * math.sqrt(f / 8.0)


def regime_features(*, velocity: float | None = None, length: float | None = None, nu: float | None = None,
                    sound_speed: float | None = None, diffusivity: float | None = None, alpha: float | None = None,
                    mean_free_path: float | None = None, g: float = 9.81, beta: float | None = None,
                    dT: float | None = None, T: float | None = None, h: float | None = None, k: float | None = None,
                    extra: Mapping[str, float] | None = None) -> dict[str, float]:
    """Every dimensionless number whose inputs are given (SI units). ``extra`` adds features computed elsewhere
    (``strain``, ``stress_ratio``, ``alpha_deg``...)."""
    f: dict[str, float] = {}
    if velocity is not None and length is not None:
        if nu is not None:
            f["Re"] = reynolds(velocity, length, nu)
        if diffusivity is not None:
            f["Pe"] = peclet(velocity, length, diffusivity)
        elif alpha is not None:
            f["Pe"] = peclet(velocity, length, alpha)
        f["Fr"] = froude(velocity, length, g)
    if velocity is not None and (sound_speed is not None or T is not None):
        f["Ma"] = mach(velocity, sound_speed, T=T)
    if mean_free_path is not None and length is not None:
        f["Kn"] = knudsen(mean_free_path, length)
    if beta is not None and dT is not None:
        f["beta_dT"] = beta * abs(dT)
        if length is not None and nu is not None:
            f["Gr"] = grashof(g, beta, dT, length, nu)
            if alpha is not None:
                f["Ra"] = rayleigh(g, beta, dT, length, nu, alpha)
        if length is not None and velocity is not None:
            f["Ri"] = richardson(g, beta, dT, length, velocity)
    if nu is not None and alpha is not None:
        f["Pr"] = prandtl(nu, alpha)
    if dT is not None and T is not None:
        f["dT_over_T"] = abs(dT) / T
    if h is not None and k is not None and length is not None:
        f["Bi"] = biot(h, length, k)
    f.update(dict(extra or {}))
    return f
