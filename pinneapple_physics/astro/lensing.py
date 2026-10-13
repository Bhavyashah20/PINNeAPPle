"""Gravitational lensing demos on the real sky: a Schwarzschild black hole placed in front of the stars you are
looking at, rendered with the geodesic ray tracer of :mod:`pinneapple_physics.blackhole.raytrace`, plus the closed
forms (weak-field deflection, Einstein radius, shadow diameter) checked against the Event Horizon Telescope.
"""
from __future__ import annotations

import math

import numpy as np

from . import frames
from .stars import blackbody_rgb, load_bright_stars, teff_from_bv

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98840987e30
PC = 3.0856775814913673e16
MUAS = 1e-6 / 3600 * math.pi / 180

# published EHT ring diameters (Event Horizon Telescope Collaboration 2019 ApJL 875 L1; 2022 ApJL 930 L12) and the
# mass / distance used to predict them (Gravity Collab. 2019 for Sgr A*; EHT 2019 VI for M87*)
EHT = {
    "M87*": {"mass_msun": 6.5e9, "distance_pc": 16.8e6, "ring_uas": 42.0, "ring_err_uas": 3.0},
    "Sgr A*": {"mass_msun": 4.0e6, "distance_pc": 8.15e3, "ring_uas": 51.8, "ring_err_uas": 2.3},
}


def deflection_angle_rad(mass_kg, impact_parameter_m):
    """Weak-field light deflection 4GM / (c^2 b)."""
    return 4 * G * mass_kg / (C**2 * np.asarray(impact_parameter_m, dtype=float))


def einstein_radius_rad(mass_kg, d_lens_m, d_source_m):
    d_ls = d_source_m - d_lens_m
    return np.sqrt(4 * G * mass_kg / C**2 * d_ls / (d_lens_m * d_source_m))


def shadow_diameter_rad(mass_kg, distance_m):
    """Angular diameter of a Schwarzschild shadow, 2 sqrt(27) GM / (c^2 D)."""
    return 2 * math.sqrt(27) * G * mass_kg / (C**2 * distance_m)


def eht_comparison() -> list[dict]:
    rows = []
    for name, d in EHT.items():
        pred = shadow_diameter_rad(d["mass_msun"] * M_SUN, d["distance_pc"] * PC) / MUAS
        rows.append({"object": name, "predicted_uas": pred, "observed_uas": d["ring_uas"],
                     "observed_err_uas": d["ring_err_uas"],
                     "sigma": abs(pred - d["ring_uas"]) / d["ring_err_uas"]})
    return rows


def sun_deflection_arcsec(elongation_deg):
    """Deflection of starlight by the Sun at an angular distance from its centre (Eddington 1919: 1.75" at the
    limb), using b = 1 AU sin(elongation)."""
    b = 1.495978707e11 * np.sin(np.radians(elongation_deg))
    return np.degrees(deflection_angle_rad(M_SUN, b)) * 3600


def catalog_sky(width: int = 4096, height: int = 2048, mag_limit: float = 6.5, milky_way: float = 0.12) -> np.ndarray:
    """Equirectangular RGB image (x = RA / 360 * width, Dec up) of the bright-star catalogue over the Milky Way
    glow: stars drawn as Gaussian splats with black-body colours and flux ~ 10^(-0.4 V)."""
    import os

    from PIL import Image
    cat = load_bright_stars()
    ok = cat.v <= mag_limit
    ra, dec, v = cat.ra[ok], cat.dec[ok], cat.v[ok]
    bv = np.nan_to_num(cat.bv[ok], nan=0.6)
    mw = Image.open(os.path.join(os.path.dirname(__file__), "data", "milkyway.jpg")).convert("L").resize((width, height))
    img = (np.asarray(mw, np.float32) / 255.0 * milky_way)[..., None] * np.array([0.78, 0.82, 1.0], np.float32)
    x = ra / 360 * width
    y = (90 - dec) / 180 * (height - 1)
    flux = 10 ** (-0.4 * (v - 1.5))
    col = blackbody_rgb(teff_from_bv(bv))
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            xi = np.floor(x).astype(int) + dx
            yi = np.floor(y).astype(int) + dy
            w = np.exp(-((xi + 0.5 - x) ** 2 + (yi + 0.5 - y) ** 2) / 1.2)
            np.add.at(img, (np.clip(yi, 0, height - 1), xi % width), (flux * w)[:, None] * col)
    return np.clip(img, 0, 1)


def _sample(sky, ra, dec):
    """Bilinear lookup in an equirectangular image."""
    H, W = sky.shape[:2]
    x = (ra / 360.0 * W - 0.5) % W
    y = np.clip((90.0 - dec) / 180.0 * (H - 1), 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1 = (x0 + 1) % W
    return ((1 - fx) * (1 - fy) * sky[y0, x0] + fx * (1 - fy) * sky[y0, x1] + (1 - fx) * fy * sky[y0 + 1, x0]
            + fx * fy * sky[y0 + 1, x1])


_TRACES: dict = {}


def _trace(W, H, r_cam, fov_deg):
    """Geodesics depend only on the camera, not on where it points: trace once per (size, r_cam, fov)."""
    from ..blackhole.raytrace import Camera
    key = (W, H, round(r_cam, 3), round(fov_deg, 3))
    if key not in _TRACES:
        if len(_TRACES) > 8:
            _TRACES.pop(next(iter(_TRACES)))
        cam = Camera(width=W, height=H, r_cam=r_cam, inclination_deg=90.0, fov_deg=fov_deg)
        grid = {"r_faces": np.geomspace(2.05, 2.1, 3), "theta_faces": np.linspace(0, math.pi, 3)}
        _TRACES[key] = (cam, cam.trace(grid, r_max=2.1))
    return _TRACES[key]


def lensed_sky(ra_deg: float, dec_deg: float, fov_deg: float = 30.0, width: int = 480, height: int = 270,
               r_cam: float = 30.0, sky: np.ndarray | None = None, supersample: int = 2) -> dict:
    """Render the real sky around (ra, dec) as seen through a Schwarzschild black hole at that position.

    The camera sits at ``r_cam`` GM/c^2 from the hole; the geodesics come from ``Camera.trace`` of the black-hole
    module (pure NumPy RK4 on u'' = 3u^2 - u). Returns the image, the shadow mask and the shadow's angular radius."""
    W, H = width * supersample, height * supersample
    cam, paths = _trace(W, H, float(r_cam), float(fov_deg))
    _, fwd, right, up = cam.basis()
    target = frames.from_radec(ra_deg, dec_deg)
    north = np.array([0.0, 0.0, 1.0]) - target * target[2]
    north = north / np.linalg.norm(north) if np.linalg.norm(north) > 1e-9 else np.array([1.0, 0, 0])
    east = np.cross(north, target)
    # seen from inside the celestial sphere east is to the left, so camera right -> -east
    B = np.stack([fwd, right, up])
    S = np.stack([target, -east, north])
    M = S.T @ B
    d = paths.escape_dir @ M.T
    d[paths.captured] = target                       # placeholder; captured rays are painted black below
    ra, dec, _ = frames.radec(d)
    sky = catalog_sky() if sky is None else sky
    img = _sample(sky, ra, dec).reshape(H, W, 3)
    cap = paths.captured.reshape(H, W)
    img[cap] = 0.0
    if supersample > 1:
        img = img.reshape(height, supersample, width, supersample, 3).mean((1, 3))
        cap = cap.reshape(height, supersample, width, supersample).mean((1, 3)) > 0.5
    alpha = math.degrees(math.asin(min(1.0, math.sqrt(27) * math.sqrt(1 - 2 / r_cam) / r_cam)))
    return {"image": img, "captured": cap, "shadow_radius_deg": alpha, "fov_deg": fov_deg}


# ------------------------------------------------------------------------------------------- accretion disc view
_DISC_CACHE: dict = {}
B_CRIT = 3 * math.sqrt(3.0)                    # critical impact parameter of the Schwarzschild photon sphere (GM/c^2)


def trace_thin_disc(width: int = 640, height: int = 360, inclination_deg: float = 80.0, r_cam: float = 60.0,
                    fov_deg: float = 32.0, max_crossings: int = 3, dphi: float = 0.004) -> dict:
    """Exact backward ray tracing for an infinitely thin equatorial disc around a Schwarzschild black hole
    (G = M = c = 1).

    Every pixel's photon moves in a plane through the hole; in that plane the orbit obeys the Binet equation of
    a null geodesic, u'' + u = 3 u^2 (u = 1/r, ' = d/dphi), integrated with RK4 from the static camera at
    ``r_cam``. The plane meets the equator at phi = phi0 + k pi, so the radii where the ray pierces the disc
    are read off exactly (primary image, the secondary image wrapped around the hole, the photon ring).
    Returns the crossing radii (from the camera side), each photon's angular momentum about the disc axis,
    the capture mask and the asymptotic escape direction (hole frame: disc normal = +z)."""
    i = math.radians(inclination_deg)
    C = np.array([math.sin(i), 0.0, math.cos(i)])
    fwd = -C
    z = np.array([0.0, 0.0, 1.0])
    up = z - fwd * (z @ fwd)
    up = up / np.linalg.norm(up)
    right = np.cross(fwd, up)
    t = math.tan(math.radians(fov_deg) / 2)
    X, Y = np.meshgrid(((np.arange(width) + 0.5) / width * 2 - 1) * t,
                       (1 - (np.arange(height) + 0.5) / height * 2) * t * height / width)
    d = fwd + X[..., None] * right + Y[..., None] * up
    d = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
    n = len(d)
    cosa = d @ C
    sina = np.sqrt(np.clip(1 - cosa ** 2, 1e-18, None))
    e2 = (d - cosa[:, None] * C) / sina[:, None]
    u0 = 1.0 / r_cam
    b = r_cam * sina / math.sqrt(1 - 2 * u0)                 # impact parameter measured by a static observer
    w = np.sign(-cosa) * np.sqrt(np.clip(1 / b ** 2 - u0 ** 2 * (1 - 2 * u0), 0, None))
    phi0 = np.mod(np.arctan2(-C[2], e2[:, 2]), math.pi)       # z(phi) = r (cos phi C_z + sin phi e2_z) = 0
    lz = b * np.cross(np.broadcast_to(C, (n, 3)), e2)[:, 2]    # traced ray; the real photon has -lz
    u = np.full(n, u0)
    alive = np.ones(n, bool)
    captured = np.zeros(n, bool)
    phi_end = np.zeros(n)
    hit_r = np.full((n, max_crossings), np.nan)
    k = np.zeros(n, int)
    nxt = phi0.copy()

    def f(x):
        return 3 * x * x - x

    phi = 0.0
    for _ in range(int(6 * math.pi / dphi)):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        ua, wa = u[idx], w[idx]
        k1u, k1w = wa, f(ua)
        k2u, k2w = wa + 0.5 * dphi * k1w, f(ua + 0.5 * dphi * k1u)
        k3u, k3w = wa + 0.5 * dphi * k2w, f(ua + 0.5 * dphi * k2u)
        k4u, k4w = wa + dphi * k3w, f(ua + dphi * k3u)
        un = ua + dphi / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        wn = wa + dphi / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        cr = nxt[idx] <= phi + dphi
        if cr.any():
            jj = idx[cr]
            fr = (nxt[jj] - phi) / dphi
            uc = ua[cr] + fr * (un[cr] - ua[cr])
            ok = k[jj] < max_crossings
            hit_r[jj[ok], k[jj[ok]]] = 1 / np.maximum(uc[ok], 1e-9)
            k[jj] += 1
            nxt[jj] += math.pi
        u[idx], w[idx] = un, wn
        phi += dphi
        cap = un > 0.5
        esc = (un < 0.5 * u0) & (wn < 0)
        captured[idx[cap]] = True
        phi_end[idx[esc]] = phi + un[esc] / -wn[esc]           # remaining turn to u = 0 (u ~ linear there)
        alive[idx[cap | esc]] = False
    escaped = ~captured & ~alive
    pe = phi_end[:, None]
    esc_dir = np.cos(pe) * C + np.sin(pe) * e2
    esc_dir[~escaped] = 0.0
    return {"shape": (height, width), "hit_r": hit_r, "lz_photon": -lz, "b": b, "captured": captured,
            "escape_dir": esc_dir, "basis": (fwd, right, up), "r_cam": r_cam, "fov_deg": fov_deg,
            "inclination_deg": inclination_deg}


def disc_image(tr: dict, r_in: float = 6.0, r_out: float = 20.0, doppler: bool = True, t_peak: float = 3900.0,
               exposure: float = 2.4, gamma: float = 0.6, opacity: float = 0.92) -> tuple[np.ndarray, np.ndarray]:
    """Colour and intensity of the disc for traced rays.

    Emission: the Novikov-Thorne-like flux F ~ r^-3 (1 - sqrt(r_in / r)) of a thin disc with zero torque at the
    ISCO, a local blackbody at T ~ F^1/4. Gas on Keplerian circular orbits (Omega = r^-3/2, exact in
    Schwarzschild coordinates); the photon's frequency ratio is g = sqrt(1 - 3/r) / (1 - Omega L_z), times the
    static camera's blueshift 1/sqrt(1 - 2/r_cam). Observed bolometric intensity ~ g^4 F, colour temperature g T.
    Without Doppler, only the gravitational part sqrt(1 - 2/r) is kept. Crossings are composited front to back
    (each one ``opacity`` opaque). Returns (rgb (H, W, 3) in [0, 1], coverage (H, W))."""
    from ..blackhole.raytrace import blackbody_rgb
    H, W = tr["shape"]
    r = tr["hit_r"]
    valid = np.isfinite(r) & (r >= r_in) & (r <= 2 * r_out)
    rr = np.where(valid, r, r_in * 2)
    flux = rr ** -3 * np.clip(1 - np.sqrt(r_in / rr), 0, None) * np.exp(-((rr / r_out) ** 4))
    temp = (flux / (r_in * 1.36) ** -3 / 0.1) ** 0.25          # ~1 near the flux maximum (r ~ 1.36 r_in)
    cam = 1 / math.sqrt(1 - 2 / tr["r_cam"])
    if doppler:
        g = np.sqrt(np.clip(1 - 3 / rr, 1e-6, None)) / (1 - rr ** -1.5 * tr["lz_photon"][:, None]) * cam
    else:
        g = np.sqrt(1 - 2 / rr) * cam
    inten = np.where(valid, g ** 4 * flux, 0.0)
    s = np.percentile(inten[valid], 99.5) if valid.any() else 1.0
    lum = 1 - np.exp(-exposure * (inten / s) ** gamma)
    col = blackbody_rgb(t_peak * np.where(valid, g * temp, 1.0).reshape(-1)).reshape(r.shape + (3,))
    rgb = np.zeros((len(r), 3))
    trans = np.ones(len(r))
    for kk in range(r.shape[1]):
        a = opacity * valid[:, kk] * np.clip(lum[:, kk] * 4, 0, 1)        # faint outer disc is translucent
        rgb += trans[:, None] * lum[:, kk, None] * col[:, kk]
        trans *= 1 - a
    return np.clip(rgb.reshape(H, W, 3), 0, 1), (1 - trans).reshape(H, W)


def accretion_view(ra_deg: float, dec_deg: float, width: int = 640, height: int = 360, inclination_deg: float = 86.0,
                   r_cam: float = 60.0, fov_deg: float = 42.0, doppler: bool = True, sky: np.ndarray | None = None):
    """A Schwarzschild black hole with a thin accretion disc placed in front of the real sky around (ra, dec):
    exact null geodesics (``trace_thin_disc``), Doppler beaming and gravitational redshift (``disc_image``), the
    lensed background from the bright-star catalogue and the Milky Way, then a light bloom.

    The trace depends only on the camera, so it is cached; each new sky direction only re-samples the background
    along the stored escape directions."""
    from ..blackhole.raytrace import bloom
    key = (width, height, round(inclination_deg, 2), round(r_cam, 2), round(fov_deg, 2))
    if key not in _DISC_CACHE:
        if len(_DISC_CACHE) > 4:
            _DISC_CACHE.pop(next(iter(_DISC_CACHE)))
        _DISC_CACHE[key] = trace_thin_disc(width, height, inclination_deg, r_cam, fov_deg)
    tr = _DISC_CACHE[key]
    disc, cover = disc_image(tr, doppler=doppler)
    fwd, right, up = tr["basis"]
    target = frames.from_radec(ra_deg, dec_deg)
    north = np.array([0.0, 0.0, 1.0]) - target * target[2]
    north = north / np.linalg.norm(north) if np.linalg.norm(north) > 1e-9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(north, target)
    M = np.stack([target, -east, north]).T @ np.stack([fwd, right, up])          # hole frame -> ICRS
    d = tr["escape_dir"] @ M.T
    esc = np.linalg.norm(d, axis=1) > 0
    d[~esc] = target
    ra, dec, _ = frames.radec(d)
    sky = catalog_sky(2048, 1024, milky_way=0.10) if sky is None else sky
    layer = _sample(sky, ra, dec).reshape(height, width, 3) * 0.7
    layer[~esc.reshape(height, width)] = 0.0
    img = disc + layer * (1 - cover[..., None])
    img = bloom(img, 0.7, max(1.5, width / 160))
    alpha = math.degrees(math.asin(min(1.0, B_CRIT * math.sqrt(1 - 2 / r_cam) / r_cam)))
    return {"image": img, "shadow_radius_deg": alpha, "fov_deg": fov_deg, "inclination_deg": inclination_deg}
