"""Apparent places of the Sun, Moon and planets (and catalogue stars) for an observer on Earth.

The reduction follows the classical chain (Explanatory Supplement 3rd ed. ch. 7; IERS Conventions 2010):
light-time iteration on the barycentric ephemeris, gravitational deflection by the Sun, relativistic annual +
diurnal aberration, frame bias / precession / nutation to the true equator and equinox of date, then the local
horizon with the apparent sidereal time, and optionally atmospheric refraction.

    obs = Observer(-23.55, -46.63, 760)               # Sao Paulo
    p = apparent("mars", Time.now(), obs)
    p.ra, p.dec, p.alt, p.az, p.distance_km
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import frames
from .ephemeris import AU_KM, C_KM_S, GM_SUN, RADIUS_KM, default_ephemeris
from .timescales import DAY_S, Time, gast_rad


@dataclass
class Observer:
    """A place on Earth: geodetic latitude/longitude (deg, east positive), height (m), and the weather used by the
    refraction model."""
    lat: float
    lon: float
    elevation_m: float = 0.0
    pressure_mbar: float | None = None      # default: standard atmosphere at the elevation
    temperature_c: float = 10.0
    name: str = ""

    def __post_init__(self):
        if self.pressure_mbar is None:
            self.pressure_mbar = 1013.25 * (1 - 2.25577e-5 * self.elevation_m) ** 5.25588

    @property
    def itrs(self):
        return frames.geodetic_to_itrs(self.lat, self.lon, self.elevation_m)

    def gcrs(self, t: Time):
        """Position (km) and velocity (km/s) of the observer relative to the geocentre, GCRS axes; (m, 3)."""
        r = self.itrs
        th = np.atleast_1d(gast_rad(t))
        R = frames.rot_z(-th)                                         # terrestrial (no polar motion) -> TOD
        r_tod = R @ r
        v_tod = np.cross(np.array([0, 0, frames.OMEGA_EARTH]), r_tod)
        NPBt = np.swapaxes(frames.npb_matrix(np.atleast_1d(t.tt)), -1, -2)
        return np.einsum("mij,mj->mi", NPBt, r_tod), np.einsum("mij,mj->mi", NPBt, v_tod)


@dataclass
class Apparent:
    body: str
    ra: np.ndarray                # deg, true equator & equinox of date
    dec: np.ndarray
    ra_astrometric: np.ndarray    # deg, GCRS (ICRF axes), light-time corrected, no aberration/deflection
    dec_astrometric: np.ndarray
    distance_km: np.ndarray
    light_time_s: np.ndarray
    range_rate_km_s: np.ndarray
    alt: np.ndarray | None = None
    az: np.ndarray | None = None
    alt_apparent: np.ndarray | None = None      # with refraction
    hour_angle: np.ndarray | None = None        # deg
    unit_gcrs: np.ndarray | None = field(default=None, repr=False)   # apparent direction, GCRS axes
    unit_date: np.ndarray | None = field(default=None, repr=False)   # apparent direction, true of date

    @property
    def angular_radius_deg(self):
        r = RADIUS_KM.get(self.body)
        return None if r is None else np.degrees(np.arcsin(np.clip(r / self.distance_km, 0, 1)))


def _deflect(p, e_vec, q_vec, E_km):
    """Light deflection by the Sun (IERS 2010 eq. 7.69 form). p: unit observer->target, e: unit Sun->observer,
    q: unit Sun->target, E: Sun-observer distance (km)."""
    g = 2 * GM_SUN / (C_KM_S**2 * E_km)
    pq = np.sum(p * q_vec, -1, keepdims=True)
    ep = np.sum(e_vec * p, -1, keepdims=True)
    qe = np.sum(q_vec * e_vec, -1, keepdims=True)
    return p + g[..., None] * (pq * e_vec - ep * q_vec) / (1 + qe)


def _aberrate(p, v_over_c):
    """Relativistic aberration of unit vector ``p`` for observer velocity ``v/c`` (barycentric)."""
    v2 = np.sum(v_over_c**2, -1, keepdims=True)
    binv = np.sqrt(1 - v2)
    pv = np.sum(p * v_over_c, -1, keepdims=True)
    out = (binv * p + (1 + pv / (1 + binv)) * v_over_c) / (1 + pv)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def refraction_deg(alt_true_deg, pressure_mbar=1010.0, temperature_c=10.0):
    """Atmospheric refraction (deg) for a true altitude, Saemundsson (1986) with the usual P/T scaling.
    Zero below -1 deg, where the formula is not meaningful."""
    h = np.asarray(alt_true_deg, dtype=float)
    R = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0
    R *= (pressure_mbar / 1010.0) * (283.0 / (273.0 + temperature_c))
    return np.where(h > -1.0, R, 0.0)


def airmass(alt_apparent_deg):
    """Relative air mass (Kasten & Young 1989), finite down to the horizon; ``inf`` below it."""
    h = np.asarray(alt_apparent_deg, dtype=float)
    X = 1.0 / (np.sin(np.radians(h)) + 0.50572 * (np.maximum(h, -0.5) + 6.07995) ** -1.6364)
    return np.where(h > -0.5, X, np.inf)


def extinction_mag(alt_apparent_deg, k_v=0.25):
    """Atmospheric extinction (mag) at V for an extinction coefficient ``k_v`` (mag/airmass; 0.2-0.3 typical)."""
    return k_v * airmass(alt_apparent_deg)


def _horizon(unit_date, t: Time, observer: Observer):
    ra, dec, _ = frames.radec(unit_date)
    last = np.degrees(np.atleast_1d(gast_rad(t))) + observer.lon
    ha = np.mod(last - ra, 360.0)
    alt, az = frames.equatorial_to_horizontal(np.radians(ha), np.radians(dec), np.radians(observer.lat))
    return alt, az, ha


def apparent(body: str, t: Time, observer: Observer | None = None, eph=None, *, deflection: bool = True,
             refraction: bool = True) -> Apparent:
    """Apparent place of a solar-system ``body`` at ``t`` (scalar or array), geocentric if ``observer`` is None."""
    eph = eph or default_ephemeris()
    jd = np.atleast_1d(t.tdb)
    tt = Time(np.atleast_1d(t.utc), np.atleast_1d(t.dut1))
    e_pos, e_vel = eph.barycentric("earth", jd)
    if observer is not None:
        o_pos, o_vel = observer.gcrs(tt)
        e_pos, e_vel = e_pos + o_pos, e_vel + o_vel
    # light-time iteration
    tau = np.zeros(len(jd))
    for _ in range(3):
        b_pos, b_vel = eph.barycentric(body, jd - tau / DAY_S)
        d = b_pos - e_pos
        tau = np.linalg.norm(d, axis=-1) / C_KM_S
    dist = np.linalg.norm(d, axis=-1)
    p = d / dist[:, None]
    rr = np.sum(p * (b_vel - e_vel), -1)
    ra_a, dec_a, _ = frames.radec(p)
    if deflection and body != "sun":
        s_pos, _ = eph.barycentric("sun", jd - tau / DAY_S)
        e = e_pos - s_pos
        q = b_pos - s_pos
        E = np.linalg.norm(e, axis=-1)
        p = _deflect(p, e / E[:, None], q / np.linalg.norm(q, axis=-1, keepdims=True), E)
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
    p = _aberrate(p, e_vel / C_KM_S)
    NPB = frames.npb_matrix(np.atleast_1d(tt.tt))
    u_date = np.einsum("mij,mj->mi", NPB, p)
    ra, dec, _ = frames.radec(u_date)
    out = Apparent(body, ra, dec, ra_a, dec_a, dist, tau, rr, unit_gcrs=p, unit_date=u_date)
    if observer is not None:
        alt, az, ha = _horizon(u_date, tt, observer)
        out.alt, out.az, out.hour_angle = alt, az, ha
        out.alt_apparent = alt + (refraction_deg(alt, observer.pressure_mbar, observer.temperature_c)
                                  if refraction else 0.0)
    return out


def star_apparent(ra_j2000, dec_j2000, t: Time, observer: Observer | None = None, *, pm_ra_mas=0.0,
                  pm_dec_mas=0.0, parallax_mas=0.0, eph=None):
    """Apparent places of catalogue stars (ICRS J2000 epoch 2000.0) at a single instant ``t``.

    Applies proper motion (linear in the tangent plane, pm_ra is mu_alpha* = mu_alpha cos dec), annual parallax
    and aberration, precession and nutation. Returns (ra, dec[, alt, az, alt_apparent]) arrays in degrees."""
    eph = eph or default_ephemeris()
    years = float((np.ravel(t.tt)[0] - 2451545.0) / 365.25)
    ra0, dec0 = np.radians(ra_j2000), np.radians(dec_j2000)
    u = frames.from_radec(ra_j2000, dec_j2000)
    east = np.stack([-np.sin(ra0), np.cos(ra0), np.zeros_like(ra0)], -1)
    north = np.stack([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)], -1)
    mas = np.pi / (180 * 3600e3)
    u = u + (np.asarray(pm_ra_mas)[..., None] * east + np.asarray(pm_dec_mas)[..., None] * north) * mas * years
    e_pos, e_vel = eph.barycentric("earth", np.atleast_1d(t.tdb))
    tt = Time(np.atleast_1d(t.utc), np.atleast_1d(t.dut1))
    if observer is not None:
        o_pos, o_vel = observer.gcrs(tt)
        e_pos, e_vel = e_pos + o_pos, e_vel + o_vel
    u = u - np.asarray(parallax_mas)[..., None] * mas * (e_pos[0] / AU_KM)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    u = _aberrate(u, np.broadcast_to(e_vel[0] / C_KM_S, u.shape))
    u_date = u @ frames.npb_matrix(float(np.ravel(tt.tt)[0])).T
    ra, dec, _ = frames.radec(u_date)
    if observer is None:
        return ra, dec
    last = np.degrees(float(np.ravel(gast_rad(tt))[0])) + observer.lon
    ha = np.mod(last - ra, 360.0)
    alt, az = frames.equatorial_to_horizontal(np.radians(ha), np.radians(dec), np.radians(observer.lat))
    return ra, dec, alt, az, alt + refraction_deg(alt, observer.pressure_mbar, observer.temperature_c)
