"""Classical N-body integrators for the Sun-planets(-Moon) problem, with conservation diagnostics.

* ``leapfrog``   kick-drift-kick, 2nd order, symplectic
* ``yoshida4``   Yoshida (1990) 4th-order symplectic composition of leapfrog
* ``rk4``        classical Runge-Kutta 4 (not symplectic: energy drifts secularly; shown for contrast)
* ``dop853``     SciPy's adaptive 8th-order Dormand-Prince, rtol 1e-12 (reference)

Units: km, s, km^3/s^2. Initial conditions come from the JPL ephemeris (:func:`initial_state`); the result can be
compared back against the ephemeris to show what Newtonian point masses miss (relativity, asteroids, figure effects).
"""
from __future__ import annotations

import numpy as np

from .ephemeris import GM, default_ephemeris

SOLAR_SYSTEM = ["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]


def initial_state(jd_tdb: float, bodies=SOLAR_SYSTEM, eph=None):
    eph = eph or default_ephemeris()
    r = np.zeros((len(bodies), 3))
    v = np.zeros((len(bodies), 3))
    for k, b in enumerate(bodies):
        p, vv = eph.barycentric(b, np.array([jd_tdb]))
        r[k], v[k] = p[0], vv[0]
    mu = np.array([GM[b] for b in bodies])
    return r, v, mu


def accelerations(r, mu):
    d = r[None, :, :] - r[:, None, :]                 # d[i, j] = r_j - r_i
    dist3 = np.sum(d**2, -1) ** 1.5
    np.fill_diagonal(dist3, np.inf)
    return np.sum(mu[None, :, None] * d / dist3[..., None], axis=1)


def energy(r, v, mu):
    """Total energy per unit G (km^2/s^2 * km^3/s^2 / ... consistent units: sum mu_i v_i^2/2 - sum mu_i mu_j/r_ij)."""
    kin = 0.5 * np.sum(mu * np.sum(v**2, -1))
    d = np.linalg.norm(r[None] - r[:, None], axis=-1)
    iu = np.triu_indices(len(mu), 1)
    pot = -np.sum(mu[iu[0]] * mu[iu[1]] / d[iu])
    return kin + pot


def angular_momentum(r, v, mu):
    return np.sum(mu[:, None] * np.cross(r, v), axis=0)


_Y1 = 1.0 / (2 - 2 ** (1 / 3))
_Y0 = -(2 ** (1 / 3)) * _Y1


def _leapfrog_step(r, v, mu, h):
    v = v + 0.5 * h * accelerations(r, mu)
    r = r + h * v
    v = v + 0.5 * h * accelerations(r, mu)
    return r, v


def _rk4_step(r, v, mu, h):
    def f(r_, v_):
        return v_, accelerations(r_, mu)
    k1r, k1v = f(r, v)
    k2r, k2v = f(r + h / 2 * k1r, v + h / 2 * k1v)
    k3r, k3v = f(r + h / 2 * k2r, v + h / 2 * k2v)
    k4r, k4v = f(r + h * k3r, v + h * k3v)
    return r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


def integrate(r0, v0, mu, t_end_s: float, dt_s: float, method: str = "yoshida4", n_out: int = 200):
    """Integrate from t=0 to ``t_end_s``; returns dict with times (s), r (n_out, N, 3), v, and the relative
    energy error and angular-momentum drift histories."""
    t_out = np.linspace(0.0, t_end_s, n_out)
    if method == "dop853":
        from scipy.integrate import solve_ivp
        N = len(mu)

        def rhs(_t, y):
            r = y[:3 * N].reshape(N, 3)
            v = y[3 * N:].reshape(N, 3)
            return np.concatenate([v.ravel(), accelerations(r, mu).ravel()])
        sol = solve_ivp(rhs, (0, t_end_s), np.concatenate([r0.ravel(), v0.ravel()]), method="DOP853",
                        t_eval=t_out, rtol=1e-12, atol=1e-6)
        R = sol.y[:3 * N].T.reshape(-1, N, 3)
        V = sol.y[3 * N:].T.reshape(-1, N, 3)
        nfev = sol.nfev
    else:
        step = {"leapfrog": _leapfrog_step, "rk4": _rk4_step, "yoshida4": None}[method]
        R, V = [r0.copy()], [v0.copy()]
        r, v = r0.copy(), v0.copy()
        t = 0.0
        nfev = 0
        for k in range(1, n_out):
            while t < t_out[k] - 1e-9:
                h = min(dt_s, t_out[k] - t)
                if method == "yoshida4":
                    for c in (_Y1, _Y0, _Y1):
                        r, v = _leapfrog_step(r, v, mu, c * h)
                    nfev += 6
                else:
                    r, v = step(r, v, mu, h)
                    nfev += 2 if method == "leapfrog" else 4
                t += h
            R.append(r.copy())
            V.append(v.copy())
        R, V = np.array(R), np.array(V)
    E0 = energy(r0, v0, mu)
    L0 = angular_momentum(r0, v0, mu)
    dE = np.array([(energy(R[k], V[k], mu) - E0) / abs(E0) for k in range(len(t_out))])
    dL = np.array([np.linalg.norm(angular_momentum(R[k], V[k], mu) - L0) / np.linalg.norm(L0)
                   for k in range(len(t_out))])
    return {"t_s": t_out, "r": R, "v": V, "rel_energy_error": dE, "rel_angmom_error": dL, "method": method,
            "n_force_evals": int(nfev)}
