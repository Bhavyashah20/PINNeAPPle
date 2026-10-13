"""One-dimensional compressible flow of a calorically perfect gas: isentropic ratios, normal shocks, Pitot pressure
and the modified-Newtonian stagnation pressure coefficient.

Formulas from NACA Report 1135, "Equations, tables, and charts for compressible flow" (Ames Research Staff, 1953),
eqs. 44-45 (isentropic), 93-99 (normal shock) and 100 (Rayleigh Pitot formula); Anderson, *Modern Compressible
Flow*, 3rd ed. (2003), ch. 3; modified Newtonian theory: Lees, "Hypersonic flow", Proc. 5th Int. Aero. Conf. (1955),
and Anderson, *Hypersonic and High-Temperature Gas Dynamics*, 2nd ed. (2006), §3.2.
"""
from __future__ import annotations

import numpy as np

__all__ = ["isentropic_ratios", "normal_shock", "pitot_pressure_ratio", "cp_max_newtonian"]


def isentropic_ratios(M, gamma: float = 1.4) -> dict[str, np.ndarray]:
    """Stagnation-to-static ratios at Mach ``M`` (NACA 1135 eqs. 43-45, 80): ``T0_T = 1 + (gamma - 1) / 2 M^2``,
    ``p0_p = T0_T^(gamma / (gamma - 1))``, ``rho0_rho = T0_T^(1 / (gamma - 1))`` and the area ratio
    ``A_Astar = (1 / M) (2 / (gamma + 1) T0_T)^((gamma + 1) / (2 (gamma - 1)))`` (``inf`` at M = 0)."""
    M = np.asarray(M, dtype=np.float64)
    g = gamma
    t = 1.0 + 0.5 * (g - 1.0) * M * M
    with np.errstate(divide="ignore"):
        a = np.where(M > 0, (2.0 / (g + 1.0) * t) ** ((g + 1.0) / (2.0 * (g - 1.0))) / np.where(M > 0, M, 1.0),
                     np.inf)
    return {"T0_T": t, "p0_p": t ** (g / (g - 1.0)), "rho0_rho": t ** (1.0 / (g - 1.0)), "A_Astar": a}


def normal_shock(M, gamma: float = 1.4) -> dict[str, np.ndarray]:
    """Jump across a normal shock at upstream Mach ``M >= 1`` (NACA 1135 eqs. 93-99):

    - ``p2_p1 = 1 + 2 gamma / (gamma + 1) (M^2 - 1)``
    - ``rho2_rho1 = (gamma + 1) M^2 / ((gamma - 1) M^2 + 2)`` (= u1 / u2)
    - ``T2_T1 = p2_p1 / rho2_rho1``
    - ``M2 = sqrt((1 + (gamma - 1) / 2 M^2) / (gamma M^2 - (gamma - 1) / 2))``
    - ``p02_p01`` = total-pressure recovery ``rho2_rho1^(gamma / (gamma - 1)) p2_p1^(-1 / (gamma - 1))``.
    """
    M = np.asarray(M, dtype=np.float64)
    if np.any(M < 1.0):
        raise ValueError("a normal shock needs a supersonic upstream Mach number (M >= 1)")
    g = gamma
    m2 = M * M
    p = 1.0 + 2.0 * g / (g + 1.0) * (m2 - 1.0)
    r = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)
    M2 = np.sqrt((1.0 + 0.5 * (g - 1.0) * m2) / (g * m2 - 0.5 * (g - 1.0)))
    p0 = r ** (g / (g - 1.0)) * p ** (-1.0 / (g - 1.0))
    return {"p2_p1": p, "rho2_rho1": r, "T2_T1": p / r, "M2": M2, "p02_p01": p0}


def pitot_pressure_ratio(M, gamma: float = 1.4):
    """Pitot (stagnation) pressure over free-stream static pressure, ``p02 / p1``.

    Supersonic (``M >= 1``): the Rayleigh Pitot formula (NACA 1135 eq. 100), a normal shock followed by isentropic
    compression to rest,
    ``p02 / p1 = ((gamma + 1)^2 M^2 / (4 gamma M^2 - 2 (gamma - 1)))^(gamma / (gamma - 1))
    (1 - gamma + 2 gamma M^2) / (gamma + 1)``.
    Subsonic: no shock, the isentropic ``p0 / p``. Returns a float for a scalar ``M``.
    """
    Ma = np.asarray(M, dtype=np.float64)
    g = gamma
    m2 = Ma * Ma
    with np.errstate(invalid="ignore", divide="ignore"):
        sup = (((g + 1.0) ** 2 * m2 / (4.0 * g * m2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
               * (1.0 - g + 2.0 * g * m2) / (g + 1.0))
    out = np.where(Ma >= 1.0, sup, isentropic_ratios(Ma, g)["p0_p"])
    return float(out) if out.ndim == 0 else out


def cp_max_newtonian(M, gamma: float = 1.4):
    """Stagnation-point pressure coefficient of modified Newtonian theory (Lees 1955),
    ``Cp_max = (p02 / p1 - 1) / (gamma M^2 / 2)``, with ``p02`` the Pitot pressure behind the normal shock
    (:func:`pitot_pressure_ratio`); the surface law is then ``Cp = Cp_max sin^2(theta)``. Tends to
    ``((gamma + 1)^2 / (4 gamma))^(gamma / (gamma - 1)) 4 / (gamma + 1)`` (1.839 for gamma = 1.4) as M -> infinity
    (1.657 at M = 2)."""
    Ma = np.asarray(M, dtype=np.float64)
    out = 2.0 / (gamma * Ma * Ma) * (np.asarray(pitot_pressure_ratio(Ma, gamma)) - 1.0)
    return float(out) if out.ndim == 0 else out
