"""Solar-system ephemeris on top of a JPL DE kernel (public domain), read by :mod:`.spk`.

The kernel is found, in order, at ``$PINNEAPPLE_DE_KERNEL``, in the ``skyfield-data`` package (MIT, ships JPL DE421,
1900-2050), or ``~/.cache/pinneapple/de421.bsp``.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .spk import SPK
from .timescales import DAY_S, J2000

# NAIF ids. Jupiter..Pluto are system barycentres (DE421 carries no planet-centre segments for them; the offset
# is < 0.1" seen from Earth).
BODIES = {"sun": 10, "mercury": 199, "venus": 299, "earth": 399, "moon": 301, "mars": 499, "jupiter": 5,
          "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9, "emb": 3}
PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]

AU_KM = 149597870.7
C_KM_S = 299792.458
GM_SUN = 1.32712440041e11            # km^3/s^2 (DE430/DE440 value)
GM = {"sun": GM_SUN, "mercury": 22031.78, "venus": 324858.59, "earth": 398600.435, "moon": 4902.800,
      "mars": 42828.375, "jupiter": 126712764.1, "saturn": 37940585.2, "uranus": 5794556.4,
      "neptune": 6836527.1, "pluto": 975.5}
# mean radii (km), IAU WGCCRE 2015
RADIUS_KM = {"sun": 695700.0, "mercury": 2439.4, "venus": 6051.8, "earth": 6371.0, "moon": 1737.4,
             "mars": 3389.5, "jupiter": 69911.0, "saturn": 58232.0, "uranus": 25362.0, "neptune": 24622.0,
             "pluto": 1188.3}


def find_kernel() -> str:
    env = os.environ.get("PINNEAPPLE_DE_KERNEL")
    if env and os.path.exists(env):
        return env
    try:
        import skyfield_data
        p = os.path.join(os.path.dirname(skyfield_data.__file__), "data", "de421.bsp")
        if os.path.exists(p):
            return p
    except ImportError:
        pass
    p = os.path.expanduser("~/.cache/pinneapple/de421.bsp")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        "No JPL DE kernel found. `pip install skyfield-data` (bundles DE421), or set PINNEAPPLE_DE_KERNEL to a "
        ".bsp file from https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/ .")


class Ephemeris:
    def __init__(self, path: str | None = None):
        self.path = path or find_kernel()
        self.kernel = SPK(self.path)
        a, b = self.kernel.coverage
        self.jd_range = (J2000 + a / DAY_S, J2000 + b / DAY_S)

    def _et(self, jd_tdb):
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        if np.any(jd < self.jd_range[0]) or np.any(jd > self.jd_range[1]):
            raise ValueError(f"date outside the ephemeris span (JD {self.jd_range[0]:.0f}-{self.jd_range[1]:.0f})")
        return (jd - J2000) * DAY_S

    def barycentric(self, body: str, jd_tdb):
        """ICRF position (km) and velocity (km/s) relative to the solar-system barycentre, (m, 3) each."""
        return self.kernel.state(BODIES[body], 0, self._et(jd_tdb))

    def relative(self, body: str, center: str, jd_tdb):
        return self.kernel.state(BODIES[body], BODIES[center], self._et(jd_tdb))


@lru_cache(maxsize=4)
def default_ephemeris(path: str | None = None) -> Ephemeris:
    return Ephemeris(path)
