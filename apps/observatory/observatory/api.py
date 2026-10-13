"""PINNeAPPle Observatory: web API + single-page 3-D UI.

The sky and the solar system as they are now from the visitor's location, computed by pinneapple_physics.astro
(JPL DE421, IAU 2006 precession / nutation, apparent places, refraction), with satellites from CelesTrak TLEs,
astrophysics analyses, and the PINNeAPPle engine (a physics-informed orbit surrogate checked against the classical
integrator) shown with its trust checks and the validation of every method against independent references.

Run:  uvicorn observatory.api:app --port 8094   (from apps/observatory)

Environment (all optional):
  OBS_USER / OBS_PASSWORD   HTTP Basic login on everything except /health
  OBS_MAX_HEAVY             concurrent heavy computations per worker (default 2)
  OBS_TLE_GROUPS            CelesTrak groups to load (default "stations,visual")
  OBS_TLE_FILE              a local TLE file used instead of CelesTrak (offline demos, tests)
  OBS_CACHE_DIR             TLE cache directory (default ~/.cache/pinneapple/tle)
  OBS_PRECOMPUTE            "0" to skip the background warm-up (validation table, default surrogate)
"""
from __future__ import annotations

import io
import math
import os
import sys
import threading
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_physics.astro import frames
from pinneapple_physics.astro.astrometry import Observer, airmass, apparent, extinction_mag
from pinneapple_physics.astro.ephemeris import AU_KM, PLANETS, RADIUS_KM, default_ephemeris
from pinneapple_physics.astro.phenomena import (
    lunar_eclipses,
    lunar_phase_events,
    magnitude,
    moon_phase,
    phase_angle,
    planet_events,
    planetary_transits,
    rise_transit_set,
    seasons,
    sky_state,
    solar_eclipses,
    twilight_events,
)
from pinneapple_physics.astro.timescales import Time, gast_rad

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
SYSTEM = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

app = FastAPI(title="PINNeAPPle Observatory", version=VERSION,
              description="The sky and the solar system, now, from where you stand: JPL ephemerides, satellites, "
                          "astrophysics analyses and a physics-informed orbit surrogate with its trust checks.")
app.add_middleware(GZipMiddleware, minimum_size=1500)
install(app, prefix="OBS")
_HEAVY = BusyLimiter("OBS_MAX_HEAVY")
EPH = default_ephemeris()


def _time(t: str | None) -> Time:
    if not t or t == "now":
        return Time.now()
    try:
        tt = Time.from_iso(t)
    except ValueError as exc:
        raise HTTPException(400, f"bad time {t!r}: use ISO 8601, e.g. 2026-10-13T21:00:00Z") from exc
    lo, hi = EPH.jd_range
    if not lo + 2 < float(tt.tdb) < hi - 2:
        raise HTTPException(400, "time outside the DE421 span (1900-2050)")
    return tt


def _obs(lat: float, lon: float, elev: float) -> Observer:
    if not (-90 <= lat <= 90 and -180 <= lon <= 360):
        raise HTTPException(400, "latitude must be in [-90, 90] and longitude in [-180, 360]")
    return Observer(lat, ((lon + 180) % 360) - 180, elev)


def _iso(jd):
    return None if jd is None else Time(jd).iso()


def _r(x, n=5):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "ephemeris": os.path.basename(EPH.path)}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------------------------------------------- catalogue
@lru_cache(maxsize=1)
def _catalog_payload():
    from pinneapple_physics.astro.stars import load_bright_stars, load_constellations
    c = load_bright_stars()
    return {
        "source": "Yale Bright Star Catalogue 5th rev. ed. (Hoffleit & Warren 1991); constellation figures "
                  "d3-celestial (BSD-3, O. Frohn)",
        "hr": c.hr.tolist(), "ra": np.round(c.ra, 4).tolist(), "dec": np.round(c.dec, 4).tolist(),
        "pmra": np.nan_to_num(c.pmra).tolist(), "pmdec": np.nan_to_num(c.pmdec).tolist(),
        "v": c.v.tolist(), "bv": [None if np.isnan(x) else round(float(x), 3) for x in c.bv],
        "name": [c.label(i) if (c.name[i] or c.v[i] < 2.6) else "" for i in range(len(c))],
        "proper": [bool(c.name[i]) for i in range(len(c))],
        "constellations": load_constellations(),
    }


@app.get("/api/catalog")
def catalog():
    return JSONResponse(_catalog_payload(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/milkyway.jpg")
def milkyway():
    import pinneapple_physics.astro as astro
    return FileResponse(os.path.join(os.path.dirname(astro.__file__), "data", "milkyway.jpg"),
                        headers={"Cache-Control": "public, max-age=86400"})


# ------------------------------------------------------------------------------------------------------- sky
def _body_now(b: str, t: Time, obs: Observer) -> dict:
    p = apparent(b, t, obs)
    g = apparent(b, t, None)
    d = float(p.distance_km[0])
    out = {"name": b, "ra": _r(p.ra[0]), "dec": _r(p.dec[0]), "alt": _r(p.alt[0]), "az": _r(p.az[0]),
           "alt_apparent": _r(p.alt_apparent[0]), "hour_angle": _r(p.hour_angle[0]),
           "ra_j2000": _r(p.ra_astrometric[0]), "dec_j2000": _r(p.dec_astrometric[0]),
           "distance_km": _r(d, 1), "distance_au": _r(d / AU_KM, 7), "light_time_s": _r(p.light_time_s[0], 3),
           "range_rate_km_s": _r(p.range_rate_km_s[0], 4),
           "angular_diameter_arcsec": _r(2 * np.degrees(np.arcsin(RADIUS_KM[b] / d)) * 3600, 2),
           "magnitude": _r(np.ravel(magnitude(b, t, EPH))[0], 2)}
    if p.alt_apparent[0] > -0.5:
        X = float(airmass(p.alt_apparent[0]))
        out.update(airmass=_r(X, 3), extinction_mag=_r(float(extinction_mag(p.alt_apparent[0])), 3))
    if b != "sun":
        i, r, psi = phase_angle(b, t, EPH)
        out.update(phase_angle=_r(i[0], 3), illuminated=_r((1 + np.cos(np.radians(i[0]))) / 2, 4),
                   elongation=_r(psi[0], 3), sun_distance_au=_r(r[0] / AU_KM, 6))
    out["geocentric"] = {"ra": _r(g.ra[0]), "dec": _r(g.dec[0]), "distance_km": _r(g.distance_km[0], 1)}
    return out


@lru_cache(maxsize=64)
def _almanac(lat, lon, elev, day_jd):
    obs = Observer(lat, lon, elev)
    rts = {b: {k: [_iso(x) for x in v] for k, v in rise_transit_set(b, obs, day_jd, 1.0, EPH).items()}
           for b in BODIES}
    tw = twilight_events(obs, day_jd, 1.0, EPH)
    tw = {k: {kk: [_iso(x) for x in vv] for kk, vv in v.items()} for k, v in tw.items()}
    return {"window_start": _iso(day_jd), "rise_transit_set": rts, "twilight": tw}


@app.get("/api/sky")
def sky(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = Query(0.0, ge=-500, le=9000),
        t: str | None = None, span_h: float = Query(6.0, gt=0, le=48), step_min: float = Query(5.0, ge=0.5, le=60)):
    """Everything the sky view needs: per-body samples (topocentric apparent alt/az) over +-span_h/2 for smooth
    animation, the state now, the GCRS -> true-of-date matrix and sidereal time for the stars, the Moon phase,
    rise/transit/set and twilight for the observer's day."""
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    jd0 = float(tt.utc)
    n = int(round(span_h * 60 / step_min)) + 1
    jds = jd0 + (np.arange(n) - (n - 1) / 2) * step_min / 1440.0
    T = Time(jds)
    samples = {}
    for b in BODIES:
        p = apparent(b, T, obs)
        samples[b] = {"alt": np.round(p.alt_apparent, 4).tolist(), "az": np.round(p.az, 4).tolist(),
                      "ra": np.round(p.ra, 5).tolist(), "dec": np.round(p.dec, 5).tolist(),
                      "dist_km": np.round(p.distance_km, 1).tolist()}
    now = {b: _body_now(b, tt, obs) for b in BODIES}
    # local "day": from local noon before t to the next, using the longitude for the time zone
    local_noon = math.floor(jd0 + lon / 360.0 - 0.0) + 0.0 - lon / 360.0
    if local_noon > jd0:
        local_noon -= 1
    alm = _almanac(round(obs.lat, 3), round(obs.lon, 3), round(obs.elevation_m), round(local_noon, 6))
    return {
        "t": tt.iso(), "jd_utc": jd0, "jd_tt": float(tt.tt), "dut1": float(tt.dut1),
        "observer": {"lat": obs.lat, "lon": obs.lon, "elevation_m": obs.elevation_m,
                     "pressure_mbar": round(obs.pressure_mbar, 1), "temperature_c": obs.temperature_c},
        "gast_deg": float(np.degrees(gast_rad(tt))), "sidereal_rate_deg_per_day": 360.98564736629,
        "lst_hours": float((np.degrees(gast_rad(tt)) + obs.lon) % 360 / 15),
        "npb": frames.npb_matrix(float(tt.tt)).tolist(),
        "sky_state": sky_state(now["sun"]["alt"]), "sun_alt": now["sun"]["alt"],
        "samples": {"jd": jds.tolist(), "bodies": samples}, "now": now,
        "moon": moon_phase(tt, EPH), "almanac": alm,
        "method": "pinneapple_physics.astro: JPL DE421, light time, solar deflection, aberration, IAU 2006/1980 "
                  "precession-nutation, Saemundsson refraction",
    }


# ---------------------------------------------------------------------------------------------- solar system
@lru_cache(maxsize=32)
def _system(day_jd: float, days: int):
    from pinneapple_physics.astro.orbits import heliocentric_state, orbit_polyline, planet_elements
    jds = day_jd + np.arange(-days, days + 1, 1.0)
    out = {"jd": jds.tolist(), "planets": {}, "orbits": {}, "elements": {}}
    for b in SYSTEM:
        p, _ = heliocentric_state(b, jds, eph=EPH)
        out["planets"][b] = np.round(p / AU_KM, 6).tolist()
        out["orbits"][b] = np.round(orbit_polyline(b, day_jd, 360, EPH), 5).tolist()
        out["elements"][b] = {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in planet_elements(b, day_jd, EPH).items()}
    # Moon, geocentric, every 3 h over the window (it moves 13 deg/day)
    jm = day_jd + np.arange(-days, days + 0.125, 0.125)
    pm, _ = EPH.relative("moon", "earth", jm)
    E = frames.ecliptic_matrix(None)
    out["moon"] = {"jd": jm.tolist(), "pos_km": np.round(pm @ E.T, 1).tolist()}
    out["elements"]["moon"] = {k: (round(v, 6) if isinstance(v, float) else v)
                               for k, v in planet_elements("moon", day_jd, EPH).items()}
    return out


@app.get("/api/system")
def system(t: str | None = None, days: int = Query(60, ge=5, le=400)):
    """Heliocentric J2000-ecliptic positions (AU) of the planets once a day over +-days, osculating orbits and
    elements, the geocentric Moon, and the Earth's orientation (GAST, obliquity) at t."""
    tt = _time(t)
    day = math.floor(float(tt.tdb) - 0.5) + 0.5
    d = _system(day, days)
    return {**d, "t": tt.iso(), "jd_tdb": float(tt.tdb), "jd_utc": float(tt.utc), "gast_deg": float(np.degrees(gast_rad(tt))),
            "obliquity_deg": 23.4392911, "npb": frames.npb_matrix(float(tt.tt)).tolist(),
            "ecliptic_j2000": frames.ecliptic_matrix(None).tolist()}


# ------------------------------------------------------------------------------------------------ satellites
_TLE = {"t": 0.0, "sats": [], "info": {}}
_TLE_LOCK = threading.Lock()


def _satellites():
    from pinneapple_physics.astro.satellites import fetch_tles, parse_tle
    with _TLE_LOCK:
        age = time.time() - _TLE["t"]
        if (_TLE["sats"] and age < 3600) or (not _TLE["sats"] and _TLE["t"] and age < 600):
            return _TLE["sats"], _TLE["info"]            # failures are retried every 10 min, not per request
        path = os.environ.get("OBS_TLE_FILE")
        if path:
            sats, info = parse_tle(open(path).read()), {"source": os.path.basename(path), "offline_file": True}
        else:
            sats, info, seen = [], {"groups": {}}, set()
            for g in (os.environ.get("OBS_TLE_GROUPS") or "stations,visual").split(","):
                s, i = fetch_tles(g.strip(), os.environ.get("OBS_CACHE_DIR"))
                info["groups"][g] = i
                for x in s:
                    if x.norad not in seen:
                        seen.add(x.norad)
                        sats.append(x)
        _TLE.update(t=time.time(), sats=sats, info=info)
        return sats, info


@app.get("/api/satellites")
def satellites(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0,
               t: str | None = None, minutes: float = Query(20.0, ge=1, le=120), step_s: float = Query(10.0, ge=2),
               limit: int = Query(60, ge=1, le=300)):
    """Look angles of the loaded satellites over +-minutes/2 around t (for animation), with sunlit flags and
    magnitudes. Satellites with stale TLEs (> 30 days from t) are dropped."""
    from pinneapple_physics.astro.satellites import look
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    sats, info = _satellites()
    jd0 = float(tt.utc)
    n = int(minutes * 60 / step_s) + 1
    jds = jd0 + (np.arange(n) - (n - 1) / 2) * step_s / 86400.0
    T = Time(jds)
    out = []
    for s in sats[:limit * 3]:
        if abs(jd0 - s.epoch_jd) > 30:
            continue
        lk = look(s, obs, T, EPH)
        if np.any(lk["error"] != 0):
            continue
        out.append({"name": s.name, "norad": s.norad, "tle_age_days": round(jd0 - s.epoch_jd, 2),
                    "alt": np.round(lk["alt"], 3).tolist(), "az": np.round(lk["az"], 3).tolist(),
                    "range_km": np.round(lk["range_km"], 1).tolist(),
                    "sunlit": np.round(lk["sunlit"], 2).tolist(),
                    "mag": [None if not np.isfinite(m) else round(float(m), 2) for m in lk["magnitude"]],
                    "lat": np.round(lk["lat"], 3).tolist(), "lon": np.round(lk["lon"], 3).tolist(),
                    "height_km": np.round(lk["height_km"], 1).tolist()})
        if len(out) >= limit:
            break
    return {"t": tt.iso(), "jd": jds.tolist(), "satellites": out, "source": info,
            "count_loaded": len(sats),
            "note": None if sats else "No TLEs available (CelesTrak unreachable and no cache)."}


@app.get("/api/passes")
def passes(lat: float = Query(-23.5505), lon: float = Query(-46.6333), elev: float = 0.0, t: str | None = None,
           days: float = Query(3.0, gt=0, le=10), norad: str = Query("25544,48274,20580"),
           visible_only: bool = False):
    """Next passes (rise / culmination / set, max altitude, visibility, brightest magnitude) of selected
    satellites; defaults: ISS, Tiangong (CSS), Hubble."""
    from pinneapple_physics.astro.satellites import find_passes
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    sats, info = _satellites()
    want = {int(x) for x in norad.split(",") if x.strip().isdigit()}
    out = []
    with _HEAVY:
        for s in sats:
            if s.norad in want and abs(float(tt.utc) - s.epoch_jd) <= 30:
                out += find_passes(s, obs, tt, days, eph=EPH)
    out.sort(key=lambda p: p["culmination_jd"])
    if visible_only:
        out = [p for p in out if p["visible"]]
    return {"t": tt.iso(), "passes": out, "source": info,
            "method": "SGP4 (sgp4 package, Vallado 2006) -> TEME -> ITRS (GMST82); conical Earth shadow; "
                      "diffuse-sphere magnitude from the standard magnitude at 1000 km, 90 deg phase"}


# -------------------------------------------------------------------------------------------------- analyses
def _target_vec(name: str, t: Time, obs: Observer | None):
    """(ra, dec, alt, az) of a body name or 'HR <n>' star at t."""
    from pinneapple_physics.astro.astrometry import star_apparent
    from pinneapple_physics.astro.stars import load_bright_stars
    key = name.strip().lower()
    if key in BODIES:
        p = apparent(key, t, obs)
        return (float(p.ra[0]), float(p.dec[0]), None if obs is None else float(p.alt_apparent[0]),
                None if obs is None else float(p.az[0]))
    if key.startswith("hr"):
        c = load_bright_stars()
        i = c.index_of(int(key[2:].strip()))
        r = star_apparent(c.ra[i:i + 1], c.dec[i:i + 1], t, obs, pm_ra_mas=c.pmra[i:i + 1],
                          pm_dec_mas=c.pmdec[i:i + 1], parallax_mas=np.nan_to_num(c.plx[i:i + 1]))
        return (float(r[0][0]), float(r[1][0]), None if obs is None else float(r[4][0]),
                None if obs is None else float(r[3][0]))
    raise HTTPException(404, f"unknown target {name!r} (use a body name or 'HR <number>')")


@app.get("/api/analysis/separation")
def separation(a: str, b: str, t: str | None = None, lat: float = -23.5505, lon: float = -46.6333):
    tt = _time(t)
    obs = _obs(lat, lon, 0)
    A, B = _target_vec(a, tt, obs), _target_vec(b, tt, obs)
    sep = float(frames.angular_separation(A[0], A[1], B[0], B[1]))
    pa = float(frames.position_angle(A[0], A[1], B[0], B[1]))
    return {"a": a, "b": b, "separation_deg": sep, "separation_dms": _dms(sep), "position_angle_deg": pa,
            "a_radec": A[:2], "b_radec": B[:2],
            "method": "Vincenty great-circle formula on apparent places of date (topocentric)"}


def _dms(d):
    s = abs(d) * 3600
    return f"{int(s // 3600)}° {int(s % 3600 // 60):02d}′ {s % 60:04.1f}″"


@app.get("/api/analysis/target")
def target(name: str = "jupiter", lat: float = -23.5505, lon: float = -46.6333, elev: float = 0.0,
           t: str | None = None, min_alt: float = 30.0):
    """Altitude / airmass curve over the next 24 h, the dark-sky window (Sun < -18 deg) and the best observing
    window tonight: when the target is highest while the sky is dark, with the Moon's separation and altitude."""
    tt = _time(t)
    obs = _obs(lat, lon, elev)
    jd0 = float(tt.utc)
    jds = jd0 + np.arange(0, 24 * 60 + 1, 10) / 1440.0
    T = Time(jds)
    key = name.strip().lower()
    if key in BODIES:
        p = apparent(key, T, obs)
        alt, ra, dec = p.alt_apparent, p.ra, p.dec
    else:
        res = [_target_vec(name, Time(j), obs) for j in jds]
        ra, dec, alt = (np.array([r[k] for r in res]) for k in range(3))
    sun = apparent("sun", T, obs).alt
    moon = apparent("moon", T, obs)
    msep = frames.angular_separation(ra, dec, moon.ra, moon.dec)
    X = airmass(alt)
    dark = sun < -18
    ok = dark & (alt >= min_alt)
    win = None
    if ok.any():
        idx = np.nonzero(ok)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        best = max(runs, key=lambda r: len(r))
        k = best[np.argmax(alt[best])]
        win = {"start": _iso(jds[best[0]]), "end": _iso(jds[best[-1]]), "best": _iso(jds[k]),
               "max_alt": float(alt[k]), "airmass": float(X[k]), "moon_separation": float(msep[k]),
               "moon_alt": float(moon.alt_apparent[k]), "hours": len(best) / 6.0}
    return {"name": name, "t": tt.iso(), "jd": jds.tolist(), "times": [Time(j).iso() for j in jds[::6]],
            "alt": np.round(alt, 3).tolist(), "airmass": [None if not np.isfinite(x) or x > 40 else round(float(x), 3)
                                                        for x in X],
            "sun_alt": np.round(sun, 3).tolist(), "moon_alt": np.round(moon.alt_apparent, 3).tolist(),
            "moon_separation": np.round(msep, 3).tolist(), "dark": dark.tolist(), "best_window": win,
            "min_alt": min_alt}


@app.get("/api/analysis/orbit")
def orbit(body: str = "mars", t: str | None = None, lat: float = -23.5505, lon: float = -46.6333,
          line_nm: float = Query(656.281, gt=50, lt=5000)):
    """Osculating Keplerian elements, distance, light time and the Doppler shift of a spectral line."""
    from pinneapple_physics.astro.orbits import light_time_and_doppler, planet_elements
    b = body.lower()
    if b not in BODIES or b == "sun":
        raise HTTPException(404, "body must be the Moon or a planet")
    tt = _time(t)
    el = planet_elements(b, float(tt.tdb), EPH)
    p = apparent(b, tt, _obs(lat, lon, 0))
    dop = light_time_and_doppler(p.distance_km[0], p.range_rate_km_s[0], line_nm)
    return {"body": b, "t": tt.iso(), "elements": {k: (float(v) if not isinstance(v, str) else v)
                                                   for k, v in el.items()},
            "distance_km": float(p.distance_km[0]), "distance_au": float(p.distance_km[0] / AU_KM),
            "light_time_s": float(dop["light_time_s"]), "range_rate_km_s": float(p.range_rate_km_s[0]),
            "doppler": {"rest_nm": line_nm, "observed_nm": float(dop["observed_nm"]),
                        "shift_pm": float(dop["shift_nm"] * 1000), "beta": float(dop["beta"])},
            "method": "elements from the DE421 heliocentric state (J2000 ecliptic); relativistic Doppler from the "
                      "topocentric range rate"}


@app.get("/api/analysis/star")
def star(hr: int, t: str | None = None, lat: float = -23.5505, lon: float = -46.6333):
    from pinneapple_physics.astro.stars import load_bright_stars, star_properties
    c = load_bright_stars()
    try:
        i = c.index_of(hr)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    tt = _time(t)
    ra, dec, alt, az = _target_vec(f"HR {hr}", tt, _obs(lat, lon, 0))
    props = star_properties(c, i)
    return {**props, "ra_date": ra, "dec_date": dec, "alt": alt, "az": az,
            "method": "distance from the BSC5 parallax; Teff from B-V (Ballesteros 2012); BC_V from a black-body "
                      "V-band integral zero-pointed on the Sun; L and R from Stefan-Boltzmann"}


@lru_cache(maxsize=1)
def _hr_points():
    from pinneapple_physics.astro.stars import hr_diagram, load_bright_stars
    d = hr_diagram(load_bright_stars())
    return {"hr": d["hr"].tolist(), "bv": np.round(d["bv"], 3).tolist(), "abs_mag": np.round(d["abs_mag"], 3).tolist()}


@app.get("/api/analysis/hr")
def hr_diagram():
    return JSONResponse({**_hr_points(), "note": "stars with a BSC5 trigonometric parallax >= 5 mas"},
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/analysis/blackbody")
def blackbody(T: float = Query(5772.0, ge=500, le=60000)):
    from pinneapple_physics.astro.stars import blackbody_spectrum
    s = blackbody_spectrum(T)
    return {"T": T, "wavelength_nm": np.round(s["wavelength_nm"], 2).tolist(),
            "radiance": s["radiance"].tolist(), "peak_nm": s["peak_nm"], "flux_w_m2": s["flux_w_m2"], "rgb": s["rgb"]}


@lru_cache(maxsize=16)
def _events(day_jd: float, days: int, lat: float, lon: float):
    obs = Observer(lat, lon, 0.0)
    jd1 = day_jd + days
    ev = []
    for e in lunar_phase_events(day_jd, jd1, EPH):
        ev.append({**e, "kind": "moon"})
    for e in seasons(day_jd, jd1, EPH):
        ev.append({**e, "kind": "season"})
    for b in PLANETS:
        for e in planet_events(b, day_jd, jd1, EPH):
            ev.append({**e, "kind": "planet", "body": b, "event": f"{b.capitalize()} {e['event']}"})
    for e in lunar_eclipses(day_jd, jd1, EPH):
        ev.append({**e, "kind": "eclipse"})
    for e in solar_eclipses(day_jd, jd1, obs, EPH):
        ev.append({**e, "kind": "eclipse"})
    for e in planetary_transits(day_jd, jd1, EPH):
        ev.append({**e, "kind": "transit"})
    for e in ev:
        e.pop("jd", None)
        if "local" in e:
            e["local"].pop("jd", None)
    return sorted(ev, key=lambda e: e["utc"])


@app.get("/api/analysis/events")
def events(t: str | None = None, days: int = Query(365, ge=7, le=800), lat: float = -23.5505, lon: float = -46.6333):
    """Moon phases, equinoxes / solstices, oppositions, conjunctions and greatest elongations, lunar and solar
    eclipses (with local circumstances), transits of Mercury and Venus."""
    tt = _time(t)
    with _HEAVY:
        ev = _events(math.floor(float(tt.utc) - 0.5) + 0.5, days, round(lat, 1), round(lon, 1))
    return {"t": tt.iso(), "days": days, "events": ev,
            "method": "root finding on apparent geocentric ecliptic longitudes; eclipse geometry with the Danjon "
                      "shadow enlargement; local solar-eclipse circumstances from topocentric separations"}


# ----------------------------------------------------------------------------------------- lensing / black hole
_SKY = {}


@app.get("/api/analysis/lensing.png")
def lensing_png(ra: float = Query(266.4, ge=0, le=360), dec: float = Query(-29.0, ge=-90, le=90),
                fov: float = Query(50.0, ge=5, le=120), r_cam: float = Query(40.0, ge=6, le=500),
                width: int = Query(480, ge=64, le=960)):
    from PIL import Image

    from pinneapple_physics.astro.lensing import catalog_sky, lensed_sky
    height = int(width * 9 / 16) // 2 * 2
    with _HEAVY:
        if "sky" not in _SKY:
            _SKY["sky"] = catalog_sky()
        L = lensed_sky(ra, dec, fov, width, height, r_cam, sky=_SKY["sky"])
    img = (np.clip(L["image"] * 2.4, 0, 1) ** 0.62 * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Shadow-Radius-Deg": f"{L['shadow_radius_deg']:.4f}", "Cache-Control": "max-age=3600"})


@app.get("/api/analysis/lensing")
def lensing_info(elongation: float = Query(0.2666, gt=0.2, le=180)):
    from pinneapple_physics.astro.lensing import eht_comparison, sun_deflection_arcsec
    return {"eht": eht_comparison(), "sun_deflection_arcsec": float(sun_deflection_arcsec(elongation)),
            "elongation_deg": elongation,
            "method": "Schwarzschild null geodesics (RK4 on u'' = 3u^2 - u) from pinneapple_physics.blackhole; shadow "
                      "diameter 2 sqrt(27) GM/(c^2 D); weak-field deflection 4GM/(c^2 b)"}


# ---------------------------------------------------------------------------------------------------- engine
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _surrogate(body: str, day_jd: float, days: int):
    from pinneapple_physics.astro.surrogate import train_orbit_pinn
    key = ("pinn", body, day_jd, days)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    rep = train_orbit_pinn(body, day_jd, float(days), iters=800, eph=EPH)
    with _CACHE_LOCK:
        _CACHE[key] = rep
    return rep


@app.get("/api/engine/surrogate")
def engine_surrogate(body: str = "mars", t: str | None = None, days: int = Query(365, ge=30, le=730)):
    """Train (or fetch from cache) the physics-informed orbit surrogate and its trust report."""
    if body not in ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"):
        raise HTTPException(404, "body must be a planet")
    tt = _time(t)
    day = math.floor(float(tt.tdb) - 0.5) + 0.5
    with _HEAVY:
        return _surrogate(body, day, days)


@app.get("/api/engine/integrators")
def engine_integrators(years: float = Query(2.0, gt=0, le=20), dt_hours: float = Query(6.0, ge=0.5, le=48)):
    """Leapfrog, Yoshida-4 and RK4 on the Sun-planets-Moon problem from DE421 initial conditions: energy and
    angular-momentum conservation, and the position difference to DE421."""
    from pinneapple_physics.astro.nbody import SOLAR_SYSTEM, initial_state, integrate
    key = ("int", years, dt_hours)
    if key in _CACHE:
        return _CACHE[key]
    jd0 = 2451545.0
    r0, v0, mu = initial_state(jd0)
    out = {"years": years, "dt_hours": dt_hours, "methods": {}}
    with _HEAVY:
        for m in ("leapfrog", "yoshida4", "rk4"):
            t0 = time.time()
            res = integrate(r0, v0, mu, years * 365.25 * 86400, dt_hours * 3600, m, n_out=60)
            jd = jd0 + res["t_s"] / 86400
            errs = {}
            for b in ("earth", "mars", "jupiter"):
                k = SOLAR_SYSTEM.index(b)
                p, _ = EPH.barycentric(b, jd)
                errs[b] = np.linalg.norm(res["r"][:, k] - p, axis=1).round(2).tolist()
            out["methods"][m] = {"rel_energy_error": res["rel_energy_error"].tolist(),
                                 "rel_angmom_error": res["rel_angmom_error"].tolist(), "error_vs_de421_km": errs,
                                 "force_evals": res["n_force_evals"], "seconds": time.time() - t0}
        out["t_days"] = (res["t_s"] / 86400).tolist()
    _CACHE[key] = out
    return out


_VALIDATION = {"rows": None, "running": False, "seconds": None}


def _run_validation():
    from pinneapple_physics.astro import validation
    _VALIDATION["running"] = True
    t0 = time.time()
    try:
        _VALIDATION["rows"] = validation.run_all()
    except Exception as exc:  # noqa: BLE001
        _VALIDATION["error"] = str(exc)
    _VALIDATION["running"] = False
    _VALIDATION["seconds"] = time.time() - t0


@app.get("/api/validation")
def validation_table():
    """Accuracy of every method against independent references (computed once at start-up)."""
    if _VALIDATION["rows"] is None and not _VALIDATION["running"]:
        threading.Thread(target=_run_validation, daemon=True).start()
    return {"rows": _VALIDATION["rows"], "running": _VALIDATION["running"], "seconds": _VALIDATION["seconds"],
            "error": _VALIDATION.get("error")}


@app.on_event("startup")
def _warm():
    if os.environ.get("OBS_PRECOMPUTE", "1") == "0":
        return

    def go():
        _run_validation()
        try:
            tt = Time.now()
            _surrogate("mars", math.floor(float(tt.tdb) - 0.5) + 0.5, 365)
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=go, daemon=True).start()
