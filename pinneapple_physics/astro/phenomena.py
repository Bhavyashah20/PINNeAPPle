"""Almanac phenomena: rise / transit / set, twilight, Moon phase, planet magnitudes and elongations,
oppositions and conjunctions, equinoxes and solstices, lunar and solar eclipses, transits of Mercury and Venus.

Every event is found the same way: sample a smooth function of time on a grid finer than the phenomenon,
bracket the sign changes (or the minima), then refine by bisection / golden-section to < 1 s.
"""
from __future__ import annotations

import numpy as np

from . import frames
from .astrometry import Observer, apparent
from .ephemeris import AU_KM, PLANETS, RADIUS_KM, default_ephemeris
from .timescales import DAY_S, Time, datetime_from_jd

STD_REFRACTION_DEG = 34.0 / 60.0          # refraction at the horizon adopted by almanacs (USNO)
SECOND = 1.0 / DAY_S


# ----------------------------------------------------------------------------------------------- root finding
def find_roots(f, jd0: float, jd1: float, step: float, tol: float = 0.5 * SECOND):
    """All sign changes of ``f(jd_array)`` in [jd0, jd1], each refined to ``tol`` days.
    Returns a list of (jd, direction) with direction +1 for a rising crossing."""
    grid = np.arange(jd0, jd1 + step, step)
    v = f(grid)
    out = []
    for i in np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]:
        # Illinois (modified regula falsi): superlinear, but keeps the bracket like bisection
        a, b, fa, fb = grid[i], grid[i + 1], v[i], v[i + 1]
        side = 0
        for _ in range(60):
            m = (a * fb - b * fa) / (fb - fa)
            if not a < m < b:
                m = 0.5 * (a + b)
            fm = f(np.array([m]))[0]
            if np.sign(fm) == np.sign(fa):
                a, fa = m, fm
                if side == -1:
                    fb *= 0.5
                side = -1
            else:
                b, fb = m, fm
                if side == 1:
                    fa *= 0.5
                side = 1
            if b - a < tol or fm == 0 or abs(fm) < 1e-10:
                break
        out.append((m, 1 if v[i + 1] > v[i] else -1))
    return out


def find_minima(f, jd0: float, jd1: float, step: float, tol: float = 1.0 * SECOND):
    """Local minima of ``f`` (golden-section refinement). Returns a list of (jd, value)."""
    grid = np.arange(jd0, jd1 + step, step)
    v = f(grid)
    out = []
    g = (np.sqrt(5) - 1) / 2
    for i in range(1, len(v) - 1):
        if v[i] <= v[i - 1] and v[i] < v[i + 1]:
            a, b = grid[i - 1], grid[i + 1]
            c, d = b - g * (b - a), a + g * (b - a)
            fc, fd = f(np.array([c]))[0], f(np.array([d]))[0]
            while b - a > tol:
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - g * (b - a)
                    fc = f(np.array([c]))[0]
                else:
                    a, c, fc = c, d, fd
                    d = a + g * (b - a)
                    fd = f(np.array([d]))[0]
            m = 0.5 * (a + b)
            out.append((m, float(f(np.array([m]))[0])))
    return out


def _iso(jd):
    return datetime_from_jd(jd).isoformat(timespec="seconds").replace("+00:00", "Z")


# ----------------------------------------------------------------------------------------- rise, set, transit
def horizon_altitude(body: str, distance_km=None) -> float:
    """Geometric altitude of the body's centre at rise/set: -(34' refraction + semi-diameter)."""
    if body in ("sun", "moon") and distance_km is not None:
        sd = np.degrees(np.arcsin(RADIUS_KM[body] / distance_km))
    elif body == "sun":
        sd = 16.0 / 60.0
    else:
        sd = 0.0
    return -(STD_REFRACTION_DEG + sd)


def rise_transit_set(body: str, observer: Observer, jd_start: float, days: float = 1.0, eph=None):
    """Rise, upper transit and set times (UTC Julian dates) of a solar-system body in [jd_start, +days)."""
    eph = eph or default_ephemeris()

    def alt_minus_h0(jd):
        p = apparent(body, Time(jd), observer, eph, refraction=False)
        h0 = np.array([horizon_altitude(body, d) for d in p.distance_km])
        return p.alt - h0

    def ha(jd):
        p = apparent(body, Time(jd), observer, eph, refraction=False)
        return (p.hour_angle + 180.0) % 360.0 - 180.0

    step = 1.0 / 48 if body == "moon" else 1.0 / 24
    ev = find_roots(alt_minus_h0, jd_start, jd_start + days, step)
    rises = [jd for jd, d in ev if d > 0]
    sets = [jd for jd, d in ev if d < 0]
    transits = [jd for jd, d in find_roots(ha, jd_start, jd_start + days, step) if d > 0]
    # drop the +-180 deg wrap (lower culmination) which also flips sign
    transits = [jd for jd in transits if abs(ha(np.array([jd]))[0]) < 1.0]
    return {"rise": rises, "transit": transits, "set": sets}


TWILIGHT = {"civil": -6.0, "nautical": -12.0, "astronomical": -18.0}


def twilight_events(observer: Observer, jd_start: float, days: float = 1.0, eph=None):
    """Sunrise/sunset and the start/end of civil, nautical and astronomical twilight."""
    eph = eph or default_ephemeris()
    cache = {}

    def sun_alt(jd):
        key = (jd[0], len(jd))
        if key not in cache:
            cache[key] = apparent("sun", Time(jd), observer, eph, refraction=False).alt
        return cache[key]

    out = {}
    for name, h in [("sun", -50.0 / 60.0), *TWILIGHT.items()]:
        ev = find_roots(lambda jd, h=h: sun_alt(jd) - h, jd_start, jd_start + days, 1.0 / 48)
        out[name] = {"rising": [jd for jd, d in ev if d > 0], "setting": [jd for jd, d in ev if d < 0]}
    return out


def sky_state(sun_alt_deg: float) -> str:
    if sun_alt_deg > -50.0 / 60.0:
        return "day"
    if sun_alt_deg > -6:
        return "civil twilight"
    if sun_alt_deg > -12:
        return "nautical twilight"
    if sun_alt_deg > -18:
        return "astronomical twilight"
    return "night"


# ---------------------------------------------------------------------------------------- phases, magnitudes
def phase_angle(body: str, t: Time, eph=None):
    """Sun-body-Earth angle (deg) and the Sun-body distance (km), geocentric."""
    eph = eph or default_ephemeris()
    p = apparent(body, t, None, eph)
    s = apparent("sun", t, None, eph)
    psi = np.radians(frames.angular_separation(p.ra, p.dec, s.ra, s.dec))     # elongation
    R, D = s.distance_km, p.distance_km
    r = np.sqrt(R**2 + D**2 - 2 * R * D * np.cos(psi))
    i = np.arctan2(R * np.sin(psi), D - R * np.cos(psi))
    return np.degrees(i), r, np.degrees(psi)


def illuminated_fraction(body: str, t: Time, eph=None):
    i, _, _ = phase_angle(body, t, eph)
    return (1 + np.cos(np.radians(i))) / 2


def moon_phase(t: Time, eph=None) -> dict:
    """Illuminated fraction, phase angle, elongation, age-like phase longitude (0 new, 90 first quarter, 180 full)
    and the position angle of the bright limb (deg, from north through east)."""
    eph = eph or default_ephemeris()
    m = apparent("moon", t, None, eph)
    s = apparent("sun", t, None, eph)
    E = frames.ecliptic_matrix(float(np.ravel(t.tt)[0]))
    lm = np.degrees(np.arctan2(*(E @ m.unit_gcrs[0])[[1, 0]]))
    ls = np.degrees(np.arctan2(*(E @ s.unit_gcrs[0])[[1, 0]]))
    plon = (lm - ls) % 360.0
    i, _, psi = phase_angle("moon", t, eph)
    k = float((1 + np.cos(np.radians(i[0]))) / 2)
    chi = frames.position_angle(m.ra[0], m.dec[0], s.ra[0], s.dec[0])
    names = ["New Moon", "Waxing crescent", "First quarter", "Waxing gibbous", "Full Moon", "Waning gibbous",
             "Last quarter", "Waning crescent"]
    return {"illuminated_fraction": k, "phase_angle_deg": float(i[0]), "elongation_deg": float(psi[0]),
            "phase_longitude_deg": float(plon), "age_days": float(plon / 360.0 * 29.530589),
            "name": names[int(((plon + 22.5) % 360) // 45)], "bright_limb_pa_deg": float(chi),
            "distance_km": float(m.distance_km[0])}


def magnitude(body: str, t: Time, eph=None):
    """Visual magnitude of the Sun, Moon and planets (Meeus ch. 41 / Explanatory Supplement 1992 formulas;
    Saturn without the ring term, so up to ~0.8 mag too faint when the rings are open)."""
    eph = eph or default_ephemeris()
    if body == "sun":
        d = apparent("sun", t, None, eph).distance_km / AU_KM
        return -26.74 + 5 * np.log10(d)
    i, r, _ = phase_angle(body, t, eph)
    D = apparent(body, t, None, eph).distance_km
    if body == "moon":
        # Allen (1976): full Moon -12.73 at mean distance; 5 log10 of the Sun and Earth distances in AU
        return -12.73 + 0.026 * np.abs(i) + 4e-9 * i**4 + 5 * np.log10((r / AU_KM) * (D / 384400.0))
    rd = 5 * np.log10((r / AU_KM) * (D / AU_KM))
    f = {
        "mercury": -0.42 + 0.0380 * i - 0.000273 * i**2 + 0.000002 * i**3,
        "venus": -4.40 + 0.0009 * i + 0.000239 * i**2 - 0.00000065 * i**3,
        "mars": -1.52 + 0.016 * i,
        "jupiter": -9.40 + 0.005 * i,
        "saturn": -8.88 + 0 * i,
        "uranus": -7.19 + 0 * i,
        "neptune": -6.87 + 0 * i,
        "pluto": -1.00 + 0 * i,
    }[body]
    return f + rd


# ------------------------------------------------------------------------ elongation, opposition, conjunction
def _ecl_lon(body, jd, eph):
    t = Time(jd)
    p = apparent(body, t, None, eph)
    E = frames.ecliptic_matrix(t.tt)
    v = np.einsum("mij,mj->mi", E, p.unit_gcrs)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def _lon_diff(body, jd, eph, target):
    d = _ecl_lon(body, jd, eph) - _ecl_lon("sun", jd, eph) - target
    return (d + 180.0) % 360.0 - 180.0


def planet_events(body: str, jd0: float, jd1: float, eph=None):
    """Oppositions and conjunctions (geocentric ecliptic longitude of date equal to the Sun's or opposite), and
    for Mercury / Venus the greatest elongations; with elongation and magnitude at each event."""
    eph = eph or default_ephemeris()
    out = []
    step = 2.0 if body in ("mercury", "venus") else 5.0

    def f(jd):                              # zero at conjunction and at opposition
        return np.sin(np.radians(_lon_diff(body, jd, eph, 0.0)))

    for jd, _d in find_roots(f, jd0, jd1, step):
        opposite = abs(_lon_diff(body, np.array([jd]), eph, 0.0)[0]) > 90
        if body in ("mercury", "venus"):
            if opposite:
                continue
            kind = "superior conjunction" if apparent(body, Time(jd), None, eph).distance_km[0] > \
                apparent("sun", Time(jd), None, eph).distance_km[0] else "inferior conjunction"
        else:
            kind = "opposition" if opposite else "conjunction"
        out.append({"event": kind, "jd": jd})
    if body in ("mercury", "venus"):
        def neg_elong(jd):
            return -phase_angle(body, Time(jd), eph)[2]
        for jd, _v in find_minima(neg_elong, jd0, jd1, step):
            side = "east" if _lon_diff(body, np.array([jd]), eph, 0.0)[0] > 0 else "west"
            out.append({"event": f"greatest elongation {side}", "jd": jd})
    for e in out:
        t = Time(e["jd"])
        e["utc"] = _iso(e["jd"])
        e["elongation_deg"] = float(phase_angle(body, t, eph)[2][0])
        e["magnitude"] = float(np.ravel(magnitude(body, t, eph))[0])
        e["distance_au"] = float(apparent(body, t, None, eph).distance_km[0] / AU_KM)
    return sorted(out, key=lambda e: e["jd"])


def lunar_phase_events(jd0: float, jd1: float, eph=None):
    """New Moon, first quarter, full Moon, last quarter between two dates."""
    eph = eph or default_ephemeris()
    out = []
    for target, name in [(0, "New Moon"), (90, "First quarter"), (180, "Full Moon"), (270, "Last quarter")]:
        def f(jd, target=target):          # continuous: zero at the target and at target + 180
            return np.sin(np.radians(_ecl_lon("moon", jd, eph) - _ecl_lon("sun", jd, eph) - target))
        for jd, direction in find_roots(f, jd0, jd1, 1.0):
            if direction > 0:
                out.append({"event": name, "jd": jd, "utc": _iso(jd)})
    return sorted(out, key=lambda e: e["jd"])


def seasons(jd0: float, jd1: float, eph=None):
    """Equinoxes and solstices: apparent geocentric longitude of the Sun = 0, 90, 180, 270 deg."""
    eph = eph or default_ephemeris()
    names = {0: "March equinox", 90: "June solstice", 180: "September equinox", 270: "December solstice"}
    out = []
    for target, name in names.items():
        def f(jd, target=target):
            return np.sin(np.radians(_ecl_lon("sun", jd, eph) - target))
        for jd, direction in find_roots(f, jd0, jd1, 2.0):
            if direction > 0:
                out.append({"event": name, "jd": jd, "utc": _iso(jd)})
    return sorted(out, key=lambda e: e["jd"])


# ------------------------------------------------------------------------------------------------- eclipses
def _sd(body, dist):
    return np.degrees(np.arcsin(RADIUS_KM[body] / dist))


def _parallax(dist):
    return np.degrees(np.arcsin(6378.137 / dist))


def _moon_ecl_lat(jd, eph):
    t = Time(np.atleast_1d(jd))
    m = apparent("moon", t, None, eph)
    v = np.einsum("mij,mj->mi", frames.ecliptic_matrix(t.tt), m.unit_gcrs)
    return float(np.degrees(np.arcsin(v[0, 2])))


ECLIPSE_LAT_LIMIT = 1.6      # deg: |Moon ecliptic latitude| at syzygy beyond which no eclipse is possible


def lunar_eclipses(jd0: float, jd1: float, eph=None):
    """Lunar eclipses: shadow geometry at each full Moon with the Danjon 2% enlargement of the Earth's shadow.
    Returns greatest-eclipse time, type and umbral / penumbral magnitudes."""
    eph = eph or default_ephemeris()
    out = []

    def geom(jd):
        t = Time(jd)
        m, s = apparent("moon", t, None, eph), apparent("sun", t, None, eph)
        sep = frames.angular_separation(m.ra, m.dec, (s.ra + 180) % 360, -s.dec)
        pm, ps, ss, sm = _parallax(m.distance_km), _parallax(s.distance_km), _sd("sun", s.distance_km), \
            _sd("moon", m.distance_km)
        umbra = 1.02 * (0.998340 * pm + ps - ss)
        pen = 1.02 * (0.998340 * pm + ps + ss)
        return sep, umbra, pen, sm

    for full in [e for e in lunar_phase_events(jd0, jd1, eph) if e["event"] == "Full Moon"]:
        if abs(_moon_ecl_lat(full["jd"], eph)) > ECLIPSE_LAT_LIMIT:
            continue
        f = lambda jd: geom(jd)[0]  # noqa: E731
        mins = find_minima(f, full["jd"] - 0.25, full["jd"] + 0.25, 1.0 / 48)
        if not mins:
            continue
        jd, _ = min(mins, key=lambda x: x[1])
        sep, umbra, pen, sm = (float(np.ravel(x)[0]) for x in geom(np.array([jd])))
        if sep > pen + sm:
            continue
        mag_u = (umbra + sm - sep) / (2 * sm)
        mag_p = (pen + sm - sep) / (2 * sm)
        kind = "total" if mag_u >= 1 else "partial" if mag_u > 0 else "penumbral"
        out.append({"event": f"{kind} lunar eclipse", "jd": jd, "utc": _iso(jd), "umbral_magnitude": mag_u,
                    "penumbral_magnitude": mag_p, "gamma_deg": sep})
    return out


def solar_eclipses(jd0: float, jd1: float, observer: Observer | None = None, eph=None):
    """Solar eclipses anywhere on Earth (geocentric criterion), with an approximate type, and the local
    circumstances (maximum obscuration and its time) for ``observer`` if given."""
    eph = eph or default_ephemeris()
    out = []

    def sep_geo(jd):
        t = Time(jd)
        m, s = apparent("moon", t, None, eph), apparent("sun", t, None, eph)
        return frames.angular_separation(m.ra, m.dec, s.ra, s.dec)

    for new in [e for e in lunar_phase_events(jd0, jd1, eph) if e["event"] == "New Moon"]:
        if abs(_moon_ecl_lat(new["jd"], eph)) > ECLIPSE_LAT_LIMIT:
            continue
        mins = find_minima(sep_geo, new["jd"] - 0.25, new["jd"] + 0.25, 1.0 / 48)
        if not mins:
            continue
        jd, sep = min(mins, key=lambda x: x[1])
        t = Time(jd)
        m, s = apparent("moon", t, None, eph), apparent("sun", t, None, eph)
        pm, ps = _parallax(m.distance_km[0]), _parallax(s.distance_km[0])
        sm, ss = _sd("moon", m.distance_km[0]), _sd("sun", s.distance_km[0])
        if sep > 0.998340 * pm - ps + sm + ss:
            continue
        gamma = sep / (0.998340 * pm - ps)          # ~ distance of the shadow axis from the geocentre, Earth radii
        if gamma < 0.9972:
            # Moon apparent size at the sub-shadow point is enlarged by ~ (1 + sin(parallax)) * cos(z) factor
            sm_top = sm * (1 + np.sin(np.radians(pm)) * np.sqrt(max(0.0, 1 - gamma**2)))
            kind = "total" if sm_top > ss * 1.0005 else "annular" if sm < ss * 0.9995 else "hybrid"
        else:
            kind = "partial"
        ev = {"event": f"{kind} solar eclipse", "jd": jd, "utc": _iso(jd), "gamma": float(gamma)}
        if observer is not None:
            def sep_top(jd_):
                tt = Time(jd_)
                mm, s2 = apparent("moon", tt, observer, eph), apparent("sun", tt, observer, eph)
                return frames.angular_separation(mm.ra, mm.dec, s2.ra, s2.dec)
            lm = find_minima(sep_top, jd - 0.2, jd + 0.2, 1.0 / 96)
            if lm:
                jl, sl = min(lm, key=lambda x: x[1])
                tt = Time(jl)
                mm, s2 = apparent("moon", tt, observer, eph), apparent("sun", tt, observer, eph)
                rm, rs = _sd("moon", mm.distance_km[0]), _sd("sun", s2.distance_km[0])
                if sl < rm + rs:
                    ev["local"] = {"max_utc": _iso(jl), "jd": jl, "magnitude": float((rm + rs - sl) / (2 * rs)),
                                   "obscuration": float(_overlap_fraction(rs, rm, sl)),
                                   "sun_altitude_deg": float(s2.alt[0]), "visible": bool(s2.alt[0] > -0.8)}
        out.append(ev)
    return out


def _overlap_fraction(rs, rm, d):
    """Fraction of the solar disc (radius rs) covered by the lunar disc (radius rm) at centre distance d."""
    if d >= rs + rm:
        return 0.0
    if d <= abs(rm - rs):
        return min(1.0, (rm / rs) ** 2)
    a = rs**2 * np.arccos((d**2 + rs**2 - rm**2) / (2 * d * rs))
    b = rm**2 * np.arccos((d**2 + rm**2 - rs**2) / (2 * d * rm))
    c = 0.5 * np.sqrt((-d + rs + rm) * (d + rs - rm) * (d - rs + rm) * (d + rs + rm))
    return (a + b - c) / (np.pi * rs**2)


def planetary_transits(jd0: float, jd1: float, eph=None):
    """Transits of Mercury and Venus across the Sun (geocentric): inferior conjunctions with the planet's
    minimum separation from the Sun's centre below the solar radius plus the planet's."""
    eph = eph or default_ephemeris()
    out = []
    for body in ("mercury", "venus"):
        for e in planet_events(body, jd0, jd1, eph):
            if e["event"] != "inferior conjunction":
                continue

            def sep(jd, body=body):
                t = Time(jd)
                p, s = apparent(body, t, None, eph), apparent("sun", t, None, eph)
                return frames.angular_separation(p.ra, p.dec, s.ra, s.dec)
            mins = find_minima(sep, e["jd"] - 0.5, e["jd"] + 0.5, 1.0 / 48)
            if not mins:
                continue
            jd, d = min(mins, key=lambda x: x[1])
            s = apparent("sun", Time(jd), None, eph)
            if d < _sd("sun", s.distance_km[0]):
                out.append({"event": f"transit of {body.capitalize()}", "jd": jd, "utc": _iso(jd),
                            "min_separation_arcsec": d * 3600})
    return sorted(out, key=lambda e: e["jd"])


__all__ = ["find_roots", "find_minima", "rise_transit_set", "twilight_events", "sky_state", "moon_phase",
           "magnitude", "illuminated_fraction", "phase_angle", "planet_events", "lunar_phase_events", "seasons",
           "lunar_eclipses", "solar_eclipses", "planetary_transits", "PLANETS"]
