"""Build the planet textures used by the 3-D views (apps/observatory/observatory/static/textures).

* Earth day / night: NASA Blue Marble and NASA "Earth at Night" imagery (public domain), from the copies in the
  three-globe npm package (MIT): pass its example/img directory. Resized to 2048 x 1024.
* Every other body: *procedural* equirectangular maps (multi-octave value noise, banding, and for the Moon its main
  maria and rayed craters placed at their selenographic coordinates). They are stylised, not imagery; drop NASA/USGS
  maps with the same file names into the directory to replace them (see README).

    python tools/make_textures.py [THREE_GLOBE_IMG_DIR]
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

OUT = os.path.join(os.path.dirname(__file__), "..", "observatory", "static", "textures")
W, H = 1024, 512
rng = np.random.default_rng(7)


def noise(octaves=6, base=4, persistence=0.55, wrap=True):
    """Multi-octave value noise on the equirectangular grid, periodic in longitude, in [0, 1]."""
    out = np.zeros((H, W))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.normal(size=(max(2, n // 2), n))
        g = np.concatenate([g, g[:, :1]], 1) if wrap else g
        z = zoom(g, (H / g.shape[0], (W + W / n) / g.shape[1]), order=3)[:H, :W]
        out += amp * z
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-12)


LAT = np.linspace(90, -90, H)[:, None] * np.ones((1, W))
LON = np.linspace(-180, 180, W, endpoint=False)[None, :] * np.ones((H, 1))


def save(name, rgb, quality=88):
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype("uint8")).save(os.path.join(OUT, name), quality=quality)


def ramp(x, stops):
    """Piecewise-linear colour ramp; stops = [(pos, (r, g, b)), ...] with colours 0..255."""
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], float) / 255
    return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1)


def disc(lat0, lon0, radius_deg):
    """Angular distance (deg) of every pixel from (lat0, lon0) in units of radius."""
    la, lo, a, b = map(np.radians, (LAT, LON, lat0, lon0))
    c = np.sin(la) * np.sin(a) + np.cos(la) * np.cos(a) * np.cos(lo - b)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) / radius_deg


def moon():
    base = 0.62 + 0.25 * (noise(7, 6) - 0.5)
    # main maria (selenographic lat, lon, angular radius deg; approximate)
    maria = [(33, -16, 17), (28, 17, 11), (8.5, 31, 13), (17, 59, 8.5), (-8, 51, 13), (-15, 35, 6),
             (-21, -17, 11), (18, -57, 26), (5, -45, 18), (-24, -39, 6), (56, 1, 8), (13, 4, 4), (-10, -23, 7),
             (-2, -40, 9), (-19, 22, 5)]
    mare = np.zeros((H, W))
    for la, lo, r in maria:
        d = disc(la, lo, r)
        mare = np.maximum(mare, np.clip(1.6 - 0.8 * d, 0, 1) * (0.7 + 0.6 * noise(4, 8)))
    mare = gaussian_filter(np.clip(mare, 0, 1), 5)
    alb = base * (1 - 0.48 * mare)
    # craters: many small bowls with bright rims
    for _ in range(900):
        la, lo = np.degrees(np.arcsin(rng.uniform(-1, 1))), rng.uniform(-180, 180)
        r = rng.pareto(2.2) * 0.6 + 0.4
        if r > 6:
            continue
        d = disc(la, lo, r)
        alb += 0.05 * np.exp(-((d - 1) / 0.18) ** 2) - 0.035 * (d < 0.9)
    for la, lo, r, k in [(-43.3, -11.2, 1.4, 0.35), (9.6, -20.1, 1.5, 0.2), (8.1, -38, 0.9, 0.18)]:  # Tycho...
        d = disc(la, lo, r)
        alb += k * np.exp(-d**2) + 0.12 * k * np.clip(1 - d / 25, 0, 1) * noise(2, 32)
    save("moon.jpg", np.clip(alb, 0, 1)[..., None] * np.array([1.0, 0.98, 0.95]))


def banded(name, stops, bands=18, turb=0.12, seed_off=0, storm=None):
    n = noise(6, 8)
    y = (LAT / 90 + 1) / 2
    w = y + turb * (n - 0.5) + 0.02 * np.sin(np.radians(LON) * 3 + LAT / 7)
    x = 0.5 + 0.5 * np.sin(w * bands * np.pi + 2 * noise(3, 2))
    rgb = ramp(x, stops)
    if storm:
        la, lo, r, col = storm
        d = disc(la, lo, r)
        rgb = rgb * (1 - np.clip(1.3 - d, 0, 1)[..., None]) + np.clip(1.3 - d, 0, 1)[..., None] * np.array(col) / 255
    save(name, gaussian_filter(rgb, (1.2, 1.2, 0)))


def rocky(name, stops, oct_=7, polar=None):
    n = noise(oct_, 5)
    rgb = ramp(n, stops)
    if polar:
        cap = np.clip((np.abs(LAT) - polar) / 4, 0, 1)[..., None]
        rgb = rgb * (1 - cap) + cap * 0.95
    save(name, rgb)


def sun():
    n = noise(7, 16, 0.6)
    save("sun.jpg", ramp(n, [(0, (255, 120, 20)), (0.5, (255, 190, 60)), (1, (255, 245, 200))]))


def rings():
    r = np.linspace(0, 1, 1024)
    dens = 0.55 + 0.45 * noise(5, 64)[0]
    # C ring, B ring, Cassini division, A ring, Encke gap (radial fraction of the texture: 1.24 .. 2.27 R_Saturn)
    rad = 1.24 + r * (2.27 - 1.24)
    prof = np.where(rad < 1.53, 0.25, np.where(rad < 1.95, 0.95, np.where(rad < 2.03, 0.08, 0.7)))
    prof = prof * np.where(np.abs(rad - 2.214) < 0.006, 0.1, 1.0)
    a = np.clip(prof * dens, 0, 1)
    col = ramp(r, [(0, (150, 130, 110)), (0.4, (230, 210, 175)), (1, (200, 185, 160))])
    img = np.concatenate([col, a[:, None]], 1)[None].repeat(16, 0)
    Image.fromarray((img * 255).astype("uint8"), "RGBA").save(os.path.join(OUT, "saturn_rings.png"))


def earth(src):
    for a, b in [("earth-blue-marble.jpg", "earth_day.jpg"), ("earth-night.jpg", "earth_night.jpg")]:
        im = Image.open(os.path.join(src, a)).convert("RGB").resize((2048, 1024), Image.LANCZOS)
        im.save(os.path.join(OUT, b), quality=85)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1:
        earth(sys.argv[1])
    moon()
    sun()
    rings()
    rocky("mercury.jpg", [(0, (90, 85, 80)), (0.5, (150, 142, 132)), (1, (205, 198, 188))])
    rocky("mars.jpg", [(0, (95, 45, 25)), (0.45, (170, 85, 45)), (0.7, (200, 120, 70)), (1, (225, 170, 120))],
          polar=78)
    banded("venus.jpg", [(0, (205, 175, 120)), (0.5, (230, 205, 150)), (1, (245, 230, 190))], bands=5, turb=0.4)
    banded("jupiter.jpg", [(0, (150, 105, 75)), (0.35, (200, 165, 130)), (0.65, (235, 220, 195)),
                           (1, (250, 245, 235))], bands=22, turb=0.07, storm=(-22, 40, 6, (190, 95, 60)))
    banded("saturn.jpg", [(0, (190, 160, 110)), (0.5, (220, 200, 150)), (1, (240, 225, 185))], bands=16, turb=0.06)
    banded("uranus.jpg", [(0, (150, 205, 210)), (1, (185, 230, 232))], bands=4, turb=0.05)
    banded("neptune.jpg", [(0, (50, 80, 175)), (0.6, (75, 115, 210)), (1, (120, 160, 230))], bands=6, turb=0.15,
           storm=(-20, -60, 7, (30, 45, 120)))
    print("textures in", os.path.abspath(OUT))
