"""Earth satellites from public two-line elements (CelesTrak GP data): SGP4 propagation (the ``sgp4`` package,
MIT, Vallado et al. 2006 reference implementation), TEME -> terrestrial frame, topocentric look angles,
Earth-shadow test, standard-magnitude brightness model and visible-pass prediction.

    sats = parse_tle(text)                      # or fetch_tles("stations")
    passes = find_passes(sats[0], Observer(-23.55, -46.63), Time.now(), days=2)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

from . import frames
from .astrometry import Observer, apparent
from .ephemeris import RADIUS_KM, default_ephemeris
from .timescales import DAY_S, Time, datetime_from_jd, gast_rad, julian_centuries

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
R_EARTH = 6378.137

# Standard magnitude at 1000 km range and 90 deg phase angle (half illuminated); approximate values from
# visual observations (Heavens-Above / McCants conventions). Unknown objects get DEFAULT_STD_MAG.
STD_MAG = {25544: -1.3, 20580: 2.2, 48274: -0.4}
DEFAULT_STD_MAG = 4.0


@dataclass
class Satellite:
    name: str
    line1: str
    line2: str

    def __post_init__(self):
        from sgp4.api import Satrec
        self.satrec = Satrec.twoline2rv(self.line1, self.line2)
        self.norad = int(self.line1[2:7])
        self.epoch_jd = self.satrec.jdsatepoch + self.satrec.jdsatepochF

    @property
    def std_mag(self):
        return STD_MAG.get(self.norad, DEFAULT_STD_MAG)

    def teme(self, jd_utc):
        jd = np.atleast_1d(np.asarray(jd_utc, dtype=float))
        whole = np.floor(jd - 0.5) + 0.5
        e, r, v = self.satrec.sgp4_array(whole, jd - whole)
        return np.asarray(r), np.asarray(v), np.asarray(e)

    def itrs(self, t: Time):
        """Position (km) and velocity (km/s) in the terrestrial frame (TEME -> PEF with GMST 1982; polar motion,
        < 15 m here, ignored)."""
        r, v, err = self.teme(t.utc)
        th = gmst82(t)
        R = frames.rot_z(th)
        rp = np.einsum("mij,mj->mi", R, r)
        vp = np.einsum("mij,mj->mi", R, v) - np.cross(np.array([0, 0, frames.OMEGA_EARTH]), rp)
        return rp, vp, err


def gmst82(t: Time):
    """GMST (rad), IAU 1982 expression used by the SGP4 TEME frame."""
    T = julian_centuries(t.ut1)
    s = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
    return np.mod(np.radians(s / 240.0), 2 * np.pi)


def parse_tle(text: str) -> list[Satellite]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            out.append(Satellite(f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append(Satellite(lines[i].lstrip("0 ").strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def fetch_tles(group: str = "stations", cache_dir: str | None = None, max_age_hours: float = 6.0,
               timeout: float = 15.0) -> tuple[list[Satellite], dict]:
    """Current TLEs of a CelesTrak group, cached on disk for ``max_age_hours`` (CelesTrak asks clients not to
    poll more than once per update cycle). Returns (satellites, info) where info has ``source``, ``fetched_utc``
    and ``error`` when the network is unavailable and no cache exists."""
    cache_dir = cache_dir or os.path.expanduser("~/.cache/pinneapple/tle")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{group}.tle")
    meta = os.path.join(cache_dir, f"{group}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_hours * 3600:
        return parse_tle(open(path).read()), json.load(open(meta)) if os.path.exists(meta) else {}
    url = CELESTRAK_GP.format(group=group)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle-observatory/1.0"})
        text = urllib.request.urlopen(req, timeout=timeout).read().decode()
        sats = parse_tle(text)
        if not sats:
            raise ValueError("empty TLE response")
        open(path, "w").write(text)
        info = {"source": url, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        json.dump(info, open(meta, "w"))
        return sats, info
    except Exception as exc:  # noqa: BLE001  network errors of every kind fall back to the cache
        if os.path.exists(path):
            info = json.load(open(meta)) if os.path.exists(meta) else {}
            return parse_tle(open(path).read()), {**info, "stale": True, "error": str(exc)}
        return [], {"source": url, "error": str(exc)}


def _enu(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return e, n, u


def sun_itrs(t: Time, eph=None):
    """Geocentric Sun position (km) in the terrestrial frame."""
    s = apparent("sun", t, None, eph or default_ephemeris(), deflection=False)
    v = s.unit_date * s.distance_km[:, None]
    return np.einsum("mij,mj->mi", frames.rot_z(gast_rad(t)), v)


def look(sat: Satellite, observer: Observer, t: Time, eph=None) -> dict:
    """Topocentric altitude/azimuth/range, sub-satellite point, illumination and magnitude, vectorised over t."""
    r, v, err = sat.itrs(t)
    o = observer.itrs
    rho = r - o
    rng = np.linalg.norm(rho, axis=-1)
    e, n, u = _enu(observer.lat, observer.lon)
    alt = np.degrees(np.arcsin(rho @ u / rng))
    az = np.degrees(np.arctan2(rho @ e, rho @ n)) % 360
    rate = np.sum(rho * v, -1) / rng
    sun = sun_itrs(t, eph)
    sunlit = _sunlit_fraction(r, sun)
    sun_alt = np.degrees(np.arcsin(np.sum((sun - o) * u, -1) / np.linalg.norm(sun - o, axis=-1)))
    to_sun = sun - r
    phase = np.arccos(np.clip(np.sum(to_sun * (o - r), -1) / (np.linalg.norm(to_sun, axis=-1) * rng), -1, 1))
    lat, lon, h = frames.itrs_to_geodetic(r)
    return {"alt": alt, "az": az, "range_km": rng, "range_rate_km_s": rate, "lat": lat, "lon": lon, "height_km": h,
            "sunlit": sunlit, "sun_alt": sun_alt, "phase_angle_deg": np.degrees(phase),
            "magnitude": magnitude(sat.std_mag, rng, phase, sunlit), "error": err}


def _sunlit_fraction(r, sun):
    """1 in sunlight, 0 in the umbra, linear across the penumbra (conical shadow, apparent disc overlap)."""
    d_sun = sun - r
    rs = np.linalg.norm(d_sun, axis=-1)
    re = np.linalg.norm(r, axis=-1)
    a_sun = np.arcsin(RADIUS_KM["sun"] / rs)
    a_earth = np.arcsin(np.clip(R_EARTH / re, -1, 1))
    theta = np.arccos(np.clip(np.sum(-r * d_sun, -1) / (re * rs), -1, 1))
    return np.clip((theta - (a_earth - a_sun)) / (2 * a_sun), 0.0, 1.0)


def magnitude(std_mag, range_km, phase_rad, sunlit=1.0):
    """Diffuse-sphere brightness: m = m_std + 5 log10(d / 1000 km) - 2.5 log10(pi F(phase)), with
    F = (sin p + (pi - p) cos p) / pi (equals 1/pi at 90 deg, where m = m_std)."""
    p = np.asarray(phase_rad, dtype=float)
    F = (np.sin(p) + (np.pi - p) * np.cos(p)) / np.pi
    with np.errstate(divide="ignore"):
        m = std_mag + 5 * np.log10(np.asarray(range_km) / 1000.0) - 2.5 * np.log10(np.maximum(np.pi * F, 1e-6))
        m = m - 2.5 * np.log10(np.maximum(np.asarray(sunlit, dtype=float), 1e-6))
    return np.where(np.asarray(sunlit) > 0, m, np.nan)


def find_passes(sat: Satellite, observer: Observer, t0: Time, days: float = 1.0, min_alt: float = 10.0,
                step_s: float = 20.0, eph=None) -> list[dict]:
    """Passes above the horizon in [t0, t0 + days): rise / culmination / set (UTC), max altitude, and
    visibility (satellite sunlit, observer's Sun below -6 deg, altitude >= ``min_alt``) with the brightest
    magnitude while visible. Times are refined to ~0.1 s."""
    eph = eph or default_ephemeris()
    jd0 = float(np.ravel(t0.utc)[0])
    jd = np.arange(jd0, jd0 + days, step_s / DAY_S)
    lk = look(sat, observer, Time(jd), eph)
    alt = lk["alt"]

    def alt_at(x):
        return look(sat, observer, Time(np.atleast_1d(x)), eph)["alt"]

    def refine(a, b, rising):
        for _ in range(30):
            m = 0.5 * (a + b)
            if (alt_at(m)[0] > 0) == rising:
                b = m
            else:
                a = m
        return 0.5 * (a + b)

    above = alt > 0
    edges = np.diff(above.astype(int))
    starts = list(np.nonzero(edges == 1)[0])
    ends = list(np.nonzero(edges == -1)[0])
    if above[0]:
        starts = [None] + starts
    out = []
    for s in starts:
        e = next((x for x in ends if s is None or x > s), None)
        i0 = 0 if s is None else s + 1
        i1 = len(jd) - 1 if e is None else e
        rise = None if s is None else refine(jd[s], jd[s + 1], True)
        set_ = None if e is None else refine(jd[e], jd[e + 1], False)
        k = i0 + int(np.argmax(alt[i0:i1 + 1]))
        # golden-section on the culmination
        a, b = jd[max(k - 1, 0)], jd[min(k + 1, len(jd) - 1)]
        g = (np.sqrt(5) - 1) / 2
        for _ in range(25):
            c, d = b - g * (b - a), a + g * (b - a)
            if alt_at(c)[0] > alt_at(d)[0]:
                b = d
            else:
                a = c
        culm = 0.5 * (a + b)
        cl = look(sat, observer, Time(np.array([culm])), eph)
        seg = slice(i0, i1 + 1)
        vis = (lk["sunlit"][seg] > 0.5) & (lk["sun_alt"][seg] < -6) & (alt[seg] >= min_alt)
        if cl["alt"][0] < min_alt:
            continue
        mags = np.where(vis, lk["magnitude"][seg], np.nan)
        out.append({
            "satellite": sat.name, "norad": sat.norad,
            "rise_utc": None if rise is None else _iso(rise), "rise_jd": rise,
            "rise_az": None if rise is None else float(look(sat, observer, Time(np.array([rise])), eph)["az"][0]),
            "culmination_utc": _iso(culm), "culmination_jd": culm, "max_alt": float(cl["alt"][0]),
            "culmination_az": float(cl["az"][0]),
            "set_utc": None if set_ is None else _iso(set_), "set_jd": set_,
            "set_az": None if set_ is None else float(look(sat, observer, Time(np.array([set_])), eph)["az"][0]),
            "visible": bool(vis.any()),
            "brightest_mag": None if not vis.any() else float(np.nanmin(mags)),
            "visible_from_utc": _iso(jd[seg][vis][0]) if vis.any() else None,
            "visible_to_utc": _iso(jd[seg][vis][-1]) if vis.any() else None,
            "tle_age_days": float(culm - sat.epoch_jd),
        })
    return out


def _iso(jd):
    return datetime_from_jd(jd).isoformat(timespec="seconds").replace("+00:00", "Z")
