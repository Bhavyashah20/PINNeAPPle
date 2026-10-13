"""Two-body orbits: osculating elements from a state vector, Kepler's equation, propagation, and the
heliocentric orbits of the planets from the JPL ephemeris.

    el = elements_from_state(r_km, v_km_s, GM_SUN)          # a, e, i, Omega, omega, M, period ...
    r, v = state_from_elements(el, GM_SUN, dt_s=86400 * 30)
"""
from __future__ import annotations

import numpy as np

from . import frames
from .ephemeris import AU_KM, GM, GM_SUN, default_ephemeris


def solve_kepler(M, e, tol=1e-14, maxiter=50):
    """Eccentric anomaly E from mean anomaly M (rad) for 0 <= e < 1 (Newton, Danby starter)."""
    M = np.mod(np.asarray(M, dtype=float), 2 * np.pi)
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        dE = f / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def elements_from_state(r, v, mu=GM_SUN) -> dict:
    """Osculating Keplerian elements (angles in degrees) of a bound orbit from position (km) / velocity (km/s)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n)
    evec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(evec)
    energy = vn**2 / 2 - mu / rn
    a = -mu / (2 * energy)
    inc = np.arccos(h[2] / hn)
    Om = np.arctan2(n[1], n[0]) if nn > 1e-12 else 0.0
    if nn > 1e-12 and e > 1e-12:
        w = np.arccos(np.clip(np.dot(n, evec) / (nn * e), -1, 1))
        if evec[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0.0
    nu = np.arccos(np.clip(np.dot(evec, r) / (e * rn), -1, 1)) if e > 1e-12 else 0.0
    if np.dot(r, v) < 0:
        nu = 2 * np.pi - nu
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    return {"a_km": a, "a_au": a / AU_KM, "e": e, "i_deg": np.degrees(inc), "Omega_deg": np.degrees(Om) % 360,
            "omega_deg": np.degrees(w) % 360, "M_deg": np.degrees(M) % 360, "nu_deg": np.degrees(nu) % 360,
            "period_days": period / 86400, "period_years": period / 86400 / 365.25,
            "perihelion_au": a * (1 - e) / AU_KM, "aphelion_au": a * (1 + e) / AU_KM,
            "specific_energy_km2_s2": energy, "specific_angular_momentum_km2_s": hn}


def state_from_elements(el: dict, mu=GM_SUN, dt_s: float | np.ndarray = 0.0):
    """Position/velocity (km, km/s) after ``dt_s`` seconds of pure Keplerian motion."""
    a, e = el["a_km"], el["e"]
    i, Om, w = np.radians([el["i_deg"], el["Omega_deg"], el["omega_deg"]])
    n = np.sqrt(mu / a**3)
    M = np.radians(el["M_deg"]) + n * np.asarray(dt_s, dtype=float)
    E = solve_kepler(M, e)
    cosE, sinE = np.cos(E), np.sin(E)
    x, y = a * (cosE - e), a * np.sqrt(1 - e**2) * sinE
    r = a * (1 - e * cosE)
    vx, vy = -np.sqrt(mu * a) / r * sinE, np.sqrt(mu * a * (1 - e**2)) / r * cosE
    R = (frames.rot_z(-Om) @ frames.rot_x(-i) @ frames.rot_z(-w))
    P = np.stack([x, y, np.zeros_like(x)], -1)
    V = np.stack([vx, vy, np.zeros_like(vx)], -1)
    return P @ R.T, V @ R.T


def heliocentric_state(body: str, jd_tdb, frame: str = "ecliptic", eph=None):
    """Heliocentric position (km) and velocity (km/s) of a body, J2000 ecliptic (default) or ICRF axes."""
    eph = eph or default_ephemeris()
    p, v = eph.barycentric(body, jd_tdb)
    sp, sv = eph.barycentric("sun", jd_tdb)
    p, v = p - sp, v - sv
    if frame == "ecliptic":
        E = frames.ecliptic_matrix(None)
        p, v = p @ E.T, v @ E.T
    return p, v


def planet_elements(body: str, jd_tdb: float, eph=None) -> dict:
    """Osculating heliocentric elements (J2000 ecliptic) of a planet; Moon: geocentric elements."""
    eph = eph or default_ephemeris()
    if body == "moon":
        p, v = eph.relative("moon", "earth", np.array([jd_tdb]))
        E = frames.ecliptic_matrix(None)
        el = elements_from_state(E @ p[0], E @ v[0], GM["earth"] + GM["moon"])
        el["center"] = "earth"
        return el
    p, v = heliocentric_state(body, np.array([jd_tdb]), eph=eph)
    el = elements_from_state(p[0], v[0], GM_SUN + GM.get(body, 0.0))
    el["center"] = "sun"
    return el


def orbit_polyline(body: str, jd_tdb: float, n: int = 256, eph=None):
    """One revolution of the osculating orbit, heliocentric J2000 ecliptic, in AU (n, 3)."""
    eph = eph or default_ephemeris()
    el = planet_elements(body, jd_tdb, eph)
    mu = GM["earth"] + GM["moon"] if body == "moon" else GM_SUN + GM.get(body, 0.0)
    T = el["period_days"] * 86400
    p, _ = state_from_elements(el, mu, np.linspace(0, T, n))
    return p / AU_KM


def light_time_and_doppler(range_km, range_rate_km_s, rest_wavelength_nm: float = 656.281):
    """One-way light time and the relativistic Doppler shift of a spectral line (H-alpha by default)."""
    beta = np.asarray(range_rate_km_s) / 299792.458
    factor = np.sqrt((1 + beta) / (1 - beta))
    return {"light_time_s": np.asarray(range_km) / 299792.458, "beta": beta, "doppler_factor": factor,
            "observed_nm": rest_wavelength_nm * factor, "shift_nm": rest_wavelength_nm * (factor - 1)}
