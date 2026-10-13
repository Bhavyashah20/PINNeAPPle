"""Reference-frame rotations: frame bias, precession (IAU 2006), nutation (IAU 1980, terms above 5 mas),
geodetic observer positions (WGS84), horizon (alt-az) and ecliptic coordinates.

Truncating the nutation series to its 18 largest terms leaves < 0.02" (validated against IAU 2000A in
``tests/test_astro_accuracy.py``), well inside the arcsecond budget of the observatory.
"""
from __future__ import annotations

import numpy as np

from .timescales import julian_centuries

ARCSEC = np.pi / (180 * 3600)

# (D, M, M', F, Omega,  dpsi0, dpsi1,  deps0, deps1) in units of 0.0001"; Meeus table 22.A (IAU 1980 theory)
_NUT = np.array([
    (0, 0, 0, 0, 1, -171996, -174.2, 92025, 8.9),
    (-2, 0, 0, 2, 2, -13187, -1.6, 5736, -3.1),
    (0, 0, 0, 2, 2, -2274, -0.2, 977, -0.5),
    (0, 0, 0, 0, 2, 2062, 0.2, -895, 0.5),
    (0, 1, 0, 0, 0, 1426, -3.4, 54, -0.1),
    (0, 0, 1, 0, 0, 712, 0.1, -7, 0),
    (-2, 1, 0, 2, 2, -517, 1.2, 224, -0.6),
    (0, 0, 0, 2, 1, -386, -0.4, 200, 0),
    (0, 0, 1, 2, 2, -301, 0, 129, -0.1),
    (-2, -1, 0, 2, 2, 217, -0.5, -95, 0.3),
    (-2, 0, 1, 0, 0, -158, 0, 0, 0),
    (-2, 0, 0, 2, 1, 129, 0.1, -70, 0),
    (0, 0, -1, 2, 2, 123, 0, -53, 0),
    (2, 0, 0, 0, 0, 63, 0, 0, 0),
    (0, 0, 1, 0, 1, 63, 0.1, -33, 0),
    (2, 0, -1, 2, 2, -59, 0, 26, 0),
    (0, 0, -1, 0, 1, -58, -0.1, 32, 0),
    (0, 0, 1, 2, 1, -51, 0, 27, 0),
])


def fundamental_args(T):
    """Delaunay arguments D, M, M', F, Omega (rad) for Julian centuries TT since J2000 (Meeus ch. 22)."""
    D = 297.85036 + 445267.111480 * T - 0.0019142 * T**2 + T**3 / 189474
    M = 357.52772 + 35999.050340 * T - 0.0001603 * T**2 - T**3 / 300000
    Mp = 134.96298 + 477198.867398 * T + 0.0086972 * T**2 + T**3 / 56250
    F = 93.27191 + 483202.017538 * T - 0.0036825 * T**2 + T**3 / 327270
    Om = 125.04452 - 1934.136261 * T + 0.0020708 * T**2 + T**3 / 450000
    return np.radians(np.stack([D, M, Mp, F, Om], axis=-1))


def nutation(jd_tt):
    """Nutation in longitude and obliquity (rad)."""
    T = julian_centuries(jd_tt)
    args = fundamental_args(T)                                    # (..., 5)
    arg = np.tensordot(args, _NUT[:, :5].T, axes=([-1], [0]))     # (..., n)
    Tn = np.asarray(T)[..., None]
    dpsi = ((_NUT[:, 5] + _NUT[:, 6] * Tn) * np.sin(arg)).sum(-1) * 1e-4 * ARCSEC
    deps = ((_NUT[:, 7] + _NUT[:, 8] * Tn) * np.cos(arg)).sum(-1) * 1e-4 * ARCSEC
    return dpsi, deps


def mean_obliquity(jd_tt):
    T = julian_centuries(jd_tt)
    # IAU 2006 (Hilton et al. 2006)
    return (84381.406 - 46.836769 * T - 0.0001831 * T**2 + 0.00200340 * T**3 - 0.000000576 * T**4
            - 0.0000000434 * T**5) * ARCSEC


def rot_x(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, s], -1), np.stack([z, -s, c], -1)], -2)


def rot_y(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, -s], -1), np.stack([z, o, z], -1), np.stack([s, z, c], -1)], -2)


def rot_z(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, s, z], -1), np.stack([-s, c, z], -1), np.stack([z, z, o], -1)], -2)


# ICRS -> mean J2000 frame bias (IERS Conventions 2010, eq. 5.21 small-angle form)
_ETA0, _XI0, _DA0 = -0.0068192 * ARCSEC, -0.0166170 * ARCSEC, -0.01460 * ARCSEC
FRAME_BIAS = rot_x(-_ETA0) @ rot_y(_XI0) @ rot_z(_DA0)


def precession_matrix(jd_tt):
    """Mean J2000 -> mean equator/equinox of date: IAU 2006 (P03) angles, IERS Conventions 2010 eq. 5.40."""
    T = julian_centuries(jd_tt)
    zeta = (2.650545 + 2306.083227 * T + 0.2988499 * T**2 + 0.01801828 * T**3 - 0.000005971 * T**4
            - 0.0000003173 * T**5) * ARCSEC
    z = (-2.650545 + 2306.077181 * T + 1.0927348 * T**2 + 0.01826837 * T**3 - 0.000028596 * T**4
         - 0.0000002904 * T**5) * ARCSEC
    theta = (2004.191903 * T - 0.4294934 * T**2 - 0.04182264 * T**3 - 0.000007089 * T**4
             - 0.0000001274 * T**5) * ARCSEC
    return rot_z(-z) @ rot_y(theta) @ rot_z(-zeta)


def nutation_matrix(jd_tt):
    """Mean of date -> true equator/equinox of date."""
    dpsi, deps = nutation(jd_tt)
    eps0 = mean_obliquity(jd_tt)
    return rot_x(-(eps0 + deps)) @ rot_z(-dpsi) @ rot_x(eps0)


def npb_matrix(jd_tt):
    """GCRS -> true equator and equinox of date (N P B)."""
    return nutation_matrix(jd_tt) @ precession_matrix(jd_tt) @ FRAME_BIAS


def ecliptic_matrix(jd_tt=None):
    """GCRS -> ecliptic. ``None`` gives the J2000 mean ecliptic (IAU 2006 obliquity 84381.406")."""
    if jd_tt is None:
        return rot_x(84381.406 * ARCSEC) @ FRAME_BIAS
    dpsi, deps = nutation(jd_tt)
    return rot_x(mean_obliquity(jd_tt) + deps) @ npb_matrix(jd_tt)


# WGS84
WGS84_A = 6378.137
WGS84_F = 1 / 298.257223563
OMEGA_EARTH = 7.2921150e-5          # rad/s


def geodetic_to_itrs(lat_deg, lon_deg, height_m=0.0):
    """Observer position (km) in the terrestrial frame."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = np.asarray(height_m) / 1000.0
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon), (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + h) * np.sin(lat)], -1)


def itrs_to_geodetic(r):
    """Inverse of :func:`geodetic_to_itrs` (Bowring iteration); returns lat, lon (deg), height (km)."""
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    e2 = WGS84_F * (2 - WGS84_F)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def radec(v):
    """Unit or full vectors (..., 3) -> RA (deg, 0..360), Dec (deg), norm."""
    v = np.asarray(v, dtype=float)
    d = np.linalg.norm(v, axis=-1)
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[..., 2] / d, -1, 1)))
    return ra, dec, d


def from_radec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def equatorial_to_horizontal(ha_rad, dec_rad, lat_rad):
    """Hour angle, declination, latitude (rad) -> altitude, azimuth (deg; azimuth from north through east)."""
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    az = np.arctan2(-np.cos(dec_rad) * np.sin(ha_rad),
                    np.sin(dec_rad) * np.cos(lat_rad) - np.cos(dec_rad) * np.sin(lat_rad) * np.cos(ha_rad))
    return np.degrees(alt), np.degrees(az) % 360.0


def horizontal_to_equatorial(alt_deg, az_deg, lat_deg):
    """Altitude/azimuth -> hour angle (deg) and declination (deg)."""
    alt, az, lat = np.radians(alt_deg), np.radians(az_deg), np.radians(lat_deg)
    sin_dec = np.sin(lat) * np.sin(alt) + np.cos(lat) * np.cos(alt) * np.cos(az)
    dec = np.arcsin(np.clip(sin_dec, -1, 1))
    ha = np.arctan2(-np.sin(az) * np.cos(alt), np.sin(alt) * np.cos(lat) - np.cos(alt) * np.sin(lat) * np.cos(az))
    return np.degrees(ha) % 360.0, np.degrees(dec)


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation (deg), Vincenty form (stable at all angles)."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    num = np.hypot(np.cos(dec2) * np.sin(dra),
                   np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.degrees(np.arctan2(num, den))


def position_angle(ra1, dec1, ra2, dec2):
    """Position angle (deg, north through east) of point 2 seen from point 1."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    return np.degrees(np.arctan2(np.sin(dra) * np.cos(dec2),
                                 np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))) % 360
