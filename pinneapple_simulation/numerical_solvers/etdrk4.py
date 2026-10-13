"""ETDRK4: fourth-order exponential time differencing Runge-Kutta for stiff
semilinear problems ``du/dt = L u + N(u, t)`` with a *diagonal* linear part.

Typical use is a spectral discretisation where ``L`` is diagonal in Fourier
space (diffusion ``-nu k^2``, hyperdiffusion ``-k^4``, dispersion ``i k^3``)
and ``N`` is the nonlinearity evaluated pseudo-spectrally -- Kuramoto-
Sivashinsky, Allen-Cahn, Burgers, KdV, Gray-Scott/Swift-Hohenberg pattern
formation. The stiff linear part is integrated exactly, so the time step is
limited by the nonlinearity, not by ``max |L|``.

Method: Cox & Matthews (2002), "Exponential time differencing for stiff
systems", J. Comput. Phys. 176:430-455, scheme ETDRK4; with the
phi-function coefficients evaluated by a contour integral as in Kassam &
Trefethen (2005), "Fourth-order time-stepping for stiff PDEs",
SIAM J. Sci. Comput. 26(4):1214-1233, which avoids the cancellation error
of the closed-form coefficients for small ``|h L|``.

Clean-room note: implemented from the two papers above. GPL-licensed
reference implementations (e.g. Dedalus) were *not* used as a source -- see
``docs/dev/rea_harvest/ledger.json``.

The blow-up check (non-finite values or ``|u|`` above a limit stops the run
instead of producing garbage) follows the behaviour observed in VisualPDE's
``checkForNaN`` (MIT), found with REA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

__all__ = ["ETDRK4", "ETDRK4Result", "SimulationBlowUp", "check_finite", "fourier_wavenumbers"]


class SimulationBlowUp(FloatingPointError):
    """The state became non-finite or exceeded the magnitude limit."""

    def __init__(self, message: str, *, step: int, t: float):
        super().__init__(message)
        self.step = step
        self.t = t


def check_finite(u: np.ndarray, *, limit: float = 1e37, step: int = -1, t: float = float("nan")) -> None:
    """Raise :class:`SimulationBlowUp` if ``u`` has a NaN/inf or ``|u| > limit``."""
    a = np.abs(u)
    if not np.all(np.isfinite(a)):
        raise SimulationBlowUp(f"non-finite state at step {step} (t={t:g})", step=step, t=t)
    if a.size and float(a.max()) > limit:
        raise SimulationBlowUp(f"|u| = {float(a.max()):.3g} > {limit:g} at step {step} (t={t:g})",
                               step=step, t=t)


def fourier_wavenumbers(n: int, length: float = 2 * np.pi, *, real: bool = False) -> np.ndarray:
    """Angular wavenumbers ``k`` of an ``n``-point periodic grid of the given
    length, in FFT order (``np.fft.rfft`` order when ``real``)."""
    d = length / n
    return 2 * np.pi * (np.fft.rfftfreq(n, d) if real else np.fft.fftfreq(n, d))


@dataclass
class ETDRK4Result:
    t: np.ndarray
    u: np.ndarray            # (n_saved, *state_shape)
    steps: int
    blew_up: Optional[SimulationBlowUp] = None
    meta: dict = field(default_factory=dict)


class ETDRK4:
    """Fixed-step ETDRK4 integrator.

    Parameters
    ----------
    linear : array of the diagonal of ``L`` (same shape as the state, or
        broadcastable to it). Real or complex.
    nonlinear : ``N(u, t) -> array`` with the state's shape.
    dt : time step ``h``.
    contour_points : points ``M`` on the contour of radius ``radius`` around
        each ``h L`` (Kassam & Trefethen use 32 and radius 1).
    """

    def __init__(self, linear: np.ndarray, nonlinear: Callable[[np.ndarray, float], np.ndarray],
                 dt: float, *, contour_points: int = 32, radius: float = 1.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        L = np.asarray(linear)
        self.dt = float(dt)
        self.nonlinear = nonlinear
        h = self.dt
        hL = h * L
        self.E = np.exp(hL)
        self.E2 = np.exp(hL / 2)
        # Full circle of contour points so complex L (dispersion) is exact too;
        # for real L the imaginary parts cancel and are dropped.
        r = radius * np.exp(2j * np.pi * (np.arange(1, contour_points + 1) - 0.5) / contour_points)
        LR = hL[..., None] + r
        eLR = np.exp(LR)
        mean = lambda z: np.mean(z, axis=-1)  # noqa: E731
        Q = h * mean((np.exp(LR / 2) - 1) / LR)
        f1 = h * mean((-4 - LR + eLR * (4 - 3 * LR + LR ** 2)) / LR ** 3)
        f2 = h * mean((2 + LR + eLR * (-2 + LR)) / LR ** 3)
        f3 = h * mean((-4 - 3 * LR - LR ** 2 + eLR * (4 - LR)) / LR ** 3)
        if not np.iscomplexobj(L):
            Q, f1, f2, f3 = Q.real, f1.real, f2.real, f3.real
        self.Q, self.f1, self.f2, self.f3 = Q, f1, f2, f3

    def step(self, u: np.ndarray, t: float) -> np.ndarray:
        h, N = self.dt, self.nonlinear
        Nu = N(u, t)
        a = self.E2 * u + self.Q * Nu
        Na = N(a, t + h / 2)
        b = self.E2 * u + self.Q * Na
        Nb = N(b, t + h / 2)
        c = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = N(c, t + h)
        return self.E * u + Nu * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3

    def solve(self, u0: np.ndarray, t_end: float, *, t0: float = 0.0, save_every: int = 1,
              check_every: int = 10, blowup_limit: float = 1e37,
              raise_on_blowup: bool = False) -> ETDRK4Result:
        """Integrate from ``t0`` to ``t_end`` (rounded to a whole number of
        steps; the actual final time is ``result.t[-1]``).

        Every ``check_every`` steps the state is checked with
        :func:`check_finite`; on blow-up the run stops and the result keeps
        everything saved up to the last good state (or the error is raised
        when ``raise_on_blowup``)."""
        n_steps = int(round((t_end - t0) / self.dt))
        u = np.array(u0, copy=True)
        ts, us = [t0], [u.copy()]
        t = t0
        blew: Optional[SimulationBlowUp] = None
        for i in range(1, n_steps + 1):
            u = self.step(u, t)
            t = t0 + i * self.dt
            if check_every and (i % check_every == 0 or i == n_steps):
                try:
                    check_finite(u, limit=blowup_limit, step=i, t=t)
                except SimulationBlowUp as exc:
                    if raise_on_blowup:
                        raise
                    blew = exc
                    break
            if i % save_every == 0 or i == n_steps:
                ts.append(t)
                us.append(u.copy())
        return ETDRK4Result(np.array(ts), np.array(us), i if n_steps else 0, blew,
                            {"dt": self.dt, "scheme": "ETDRK4 (Cox-Matthews, Kassam-Trefethen contour)"})
