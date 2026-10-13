"""Time scales for positional astronomy: UTC, TAI, TT, TDB, UT1 and sidereal time.

Written from the published definitions (IERS Conventions 2010, Explanatory Supplement 3rd ed., Meeus
"Astronomical Algorithms" 2nd ed.). All functions are vectorised over NumPy arrays of Julian dates.

    t = Time.from_datetime(datetime(2026, 10, 13, 21, 0, tzinfo=timezone.utc))
    t.tt, t.tdb, t.ut1          # Julian dates
    gast_hours(t)               # Greenwich apparent sidereal time
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

J2000 = 2451545.0
DAY_S = 86400.0
TT_MINUS_TAI = 32.184

# TAI - UTC (s) from 1972 on: (Julian date of 0h UTC when it takes effect, value). Source: IERS Bulletin C.
_LEAP = np.array([
    (2441317.5, 10), (2441499.5, 11), (2441683.5, 12), (2442048.5, 13), (2442413.5, 14), (2442778.5, 15),
    (2443144.5, 16), (2443509.5, 17), (2443874.5, 18), (2444239.5, 19), (2444786.5, 20), (2445151.5, 21),
    (2445516.5, 22), (2446247.5, 23), (2447161.5, 24), (2447892.5, 25), (2448257.5, 26), (2448804.5, 27),
    (2449169.5, 28), (2449534.5, 29), (2450083.5, 30), (2450630.5, 31), (2451179.5, 32), (2453736.5, 33),
    (2454832.5, 34), (2456109.5, 35), (2457204.5, 36), (2457754.5, 37),
])


def tai_minus_utc(jd_utc):
    """TAI-UTC in seconds. Before 1972 returns ``nan`` (UTC then was rubber seconds; use :func:`delta_t`)."""
    jd = np.asarray(jd_utc, dtype=float)
    i = np.searchsorted(_LEAP[:, 0], jd, side="right") - 1
    out = np.where(i >= 0, _LEAP[np.clip(i, 0, None), 1], np.nan)
    return out if out.ndim else float(out)


def delta_t_model(year):
    """TT - UT1 (s) from the Espenak & Meeus (2006) polynomials, 1900-2150. Used before 1972 and as a fallback."""
    y = np.asarray(year, dtype=float)
    out = np.empty_like(y)
    conds = [y < 1920, y < 1941, y < 1961, y < 1986, y < 2005, y < 2050, np.ones_like(y, bool)]
    t = [y - 1900, y - 1920, y - 1950, y - 1975, y - 2000, y - 2000, (y - 1820) / 100]
    vals = [
        -2.79 + 1.494119 * t[0] - 0.0598939 * t[0]**2 + 0.0061966 * t[0]**3 - 0.000197 * t[0]**4,
        21.20 + 0.84493 * t[1] - 0.076100 * t[1]**2 + 0.0020936 * t[1]**3,
        29.07 + 0.407 * t[2] - t[2]**2 / 233 + t[2]**3 / 2547,
        45.45 + 1.067 * t[3] - t[3]**2 / 260 - t[3]**3 / 718,
        63.86 + 0.3345 * t[4] - 0.060374 * t[4]**2 + 0.0017275 * t[4]**3 + 0.000651814 * t[4]**4
        + 0.00002373599 * t[4]**5,
        62.92 + 0.32217 * t[5] + 0.005589 * t[5]**2,
        -20 + 32 * t[6]**2 - 0.5628 * (2150 - y),
    ]
    done = np.zeros_like(y, bool)
    for c, v in zip(conds, vals, strict=True):
        m = c & ~done
        out[m] = v[m]
        done |= m
    return out if out.ndim else float(out)


class _Dut1Table:
    """UT1-UTC from an IERS ``finals2000A.all`` file, if one is available (env ``PINNEAPPLE_IERS_FINALS`` or the
    copy shipped by the ``skyfield-data`` package). Past the end of the table the last value is held (UT1-UTC drifts
    by a few tenths of a second per year and never exceeds 0.9 s); before it, and without a table, 0 is used."""

    def __init__(self):
        self.mjd = self.dut1 = None
        path = os.environ.get("PINNEAPPLE_IERS_FINALS")
        if not path:
            try:
                import skyfield_data  # noqa: F401  (MIT; bundles IERS finals2000A.all)
                path = os.path.join(os.path.dirname(skyfield_data.__file__), "data", "finals2000A.all")
            except ImportError:
                path = None
        if path and os.path.exists(path):
            mjd, d = [], []
            with open(path) as fh:
                for line in fh:
                    s = line[58:68].strip()
                    if len(line) > 68 and s:
                        try:
                            mjd.append(float(line[7:15]))
                            d.append(float(s))
                        except ValueError:
                            continue
            if mjd:
                self.mjd, self.dut1 = np.array(mjd), np.array(d)

    def __call__(self, jd_utc):
        jd = np.asarray(jd_utc, dtype=float)
        if self.mjd is None:
            return np.zeros_like(jd)
        mjd = jd - 2400000.5
        # leap-second steps: interpolate on (UT1 - TAI) which is smooth, then convert back
        tai = tai_minus_utc(self.mjd + 2400000.5)
        smooth = np.interp(np.minimum(mjd, self.mjd[-1]), self.mjd, self.dut1 - tai)
        out = smooth + np.nan_to_num(tai_minus_utc(np.minimum(jd, self.mjd[-1] + 2400000.5)))
        return np.where(mjd >= self.mjd[0], out, 0.0)


_DUT1 = None


def dut1(jd_utc):
    global _DUT1
    if _DUT1 is None:
        _DUT1 = _Dut1Table()
    return _DUT1(jd_utc)


def tdb_minus_tt(jd_tt):
    """TDB - TT (s), the 2-term periodic approximation (error < 30 us, Explanatory Supplement eq. 10.2)."""
    g = np.radians(357.53 + 0.98560028 * (np.asarray(jd_tt) - J2000))
    return 0.001657 * np.sin(g) + 0.000014 * np.sin(2 * g)


def jd_from_datetime(dt) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return 2440587.5 + dt.timestamp() / DAY_S


def datetime_from_jd(jd) -> datetime:
    return datetime.fromtimestamp((float(jd) - 2440587.5) * DAY_S, tz=timezone.utc)


@dataclass
class Time:
    """An instant (or array of instants) on the UTC scale with the derived scales as Julian dates.

    ``dut1`` (UT1-UTC, s) defaults to the IERS table when available, else 0 (bounded by 0.9 s).
    Before 1972 the input is taken as UT1 and TT = UT1 + Delta-T (Espenak & Meeus model)."""
    utc: np.ndarray
    dut1: np.ndarray | None = None

    def __post_init__(self):
        self.utc = np.asarray(self.utc, dtype=float)
        tai_utc = tai_minus_utc(self.utc)
        year = 2000.0 + (self.utc - J2000) / 365.25
        modern = np.isfinite(tai_utc)
        if self.dut1 is None:
            self.dut1 = np.where(modern, dut1(self.utc), 0.0)
        self.dut1 = np.asarray(self.dut1, dtype=float)
        tt_utc = np.where(modern, np.nan_to_num(tai_utc) + TT_MINUS_TAI, delta_t_model(year))
        self.tt = self.utc + tt_utc / DAY_S
        self.ut1 = self.utc + np.where(modern, self.dut1, 0.0) / DAY_S
        self.tdb = self.tt + tdb_minus_tt(self.tt) / DAY_S

    @classmethod
    def from_datetime(cls, dt, dut1=None) -> Time:
        return cls(jd_from_datetime(dt), dut1)

    @classmethod
    def from_iso(cls, s: str) -> Time:
        return cls.from_datetime(datetime.fromisoformat(s.replace("Z", "+00:00")))

    @classmethod
    def now(cls) -> Time:
        return cls.from_datetime(datetime.now(timezone.utc))

    @classmethod
    def from_tt(cls, jd_tt) -> Time:
        """Inverse of the UTC->TT conversion (iterated once; exact away from a leap-second instant)."""
        jd_tt = np.asarray(jd_tt, dtype=float)
        guess = cls(jd_tt - 69.184 / DAY_S)
        return cls(guess.utc + (jd_tt - guess.tt))

    @property
    def delta_t(self):
        """TT - UT1 in seconds."""
        return (self.tt - self.ut1) * DAY_S

    def offset(self, days) -> Time:
        return Time(self.utc + np.asarray(days, dtype=float))

    def iso(self) -> str:
        return datetime_from_jd(np.ravel(self.utc)[0]).isoformat().replace("+00:00", "Z")

    def __len__(self):
        return int(self.utc.size)


def julian_centuries(jd):
    return (np.asarray(jd, dtype=float) - J2000) / 36525.0


def era(jd_ut1):
    """Earth rotation angle (rad), IAU 2000."""
    du = np.asarray(jd_ut1, dtype=float) - J2000
    # 1.00273781191135448 * du split as du + 0.0027... * du keeps the fraction of a turn precise
    return 2 * np.pi * np.mod(0.7790572732640 + 0.00273781191135448 * du + np.mod(du, 1.0), 1.0)


def gmst_rad(t: Time):
    """Greenwich mean sidereal time (rad), IAU 2006 (ERA-based, Capitaine et al. 2005)."""
    T = julian_centuries(t.tt)
    poly = (0.014506 + 4612.156534 * T + 1.3915817 * T**2 - 0.00000044 * T**3
            - 0.000029956 * T**4 - 0.0000000368 * T**5)
    return np.mod(era(t.ut1) + np.radians(poly / 3600.0), 2 * np.pi)


def gast_rad(t: Time):
    """Greenwich apparent sidereal time (rad): GMST + equation of the equinoxes (Delta-psi cos eps)."""
    from .frames import mean_obliquity, nutation
    dpsi, deps = nutation(t.tt)
    eps = mean_obliquity(t.tt) + deps
    return np.mod(gmst_rad(t) + dpsi * np.cos(eps), 2 * np.pi)


def gast_hours(t: Time):
    return np.degrees(gast_rad(t)) / 15.0
