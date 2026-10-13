"""Three rotor-driven aircraft from published figures: an ATR 72-600-class regional twin turboprop, an H145-class
light twin helicopter and an F450-frame-class quadcopter drone. Lifting surfaces and rotor / propeller blades are
lofted through NACA sections with twist (constant geometric pitch for the propellers, linear twist for the main
rotor); liveries, glazing and cabin windows are face materials or thin window inserts on the closed skins."""
from __future__ import annotations

import math

import numpy as np

from .. import kit
from ..core import FRAME_VEHICLE, Body, GeometrySpec, Param, Source, register


# ============================================================================================ shared helpers
def _n(base: float, res: float, lo: int) -> int:
    return max(lo, int(round(base * res)))


def _loft_ij(rings):
    """kit.loft plus, per face, the ring index ``i`` (band between rings i and i+1) and the point index ``j``
    (between points j and j+1); caps get i = j = -1."""
    R = np.asarray(rings, float)
    nr, n, _ = R.shape
    V, F = kit.loft(R)
    m = (nr - 1) * n
    k = np.arange(len(F))
    kk = np.where(k < m, k, k - m)
    i = np.where(k < 2 * m, kk // n, -1)
    j = np.where(k < 2 * m, kk % n, -1)
    return V, F, i, j


def _skin(xs, hw, zb, zt, e, th):
    """Fuselage-like skin: super-ellipse rings (half-width hw, between zb and zt, exponent e) sampled at the
    parameter angles ``th`` (radians, 0 = starboard side, pi/2 = top). Returns V, F and per face the mid station x
    and the mid parameter angle (deg, -180..180; NaN on the caps)."""
    th = np.asarray(th, float)
    c, s = np.cos(th), np.sin(th)
    rings = []
    for k, x in enumerate(xs):
        ee = float(e[k]) if np.ndim(e) else float(e)
        hh = 0.5 * (zt[k] - zb[k])
        zc = 0.5 * (zt[k] + zb[k])
        y = hw[k] * np.sign(c) * np.abs(c) ** (2 / ee)
        z = zc + hh * np.sign(s) * np.abs(s) ** (2 / ee)
        rings.append(np.stack([np.full_like(th, x), y, z], 1))
    V, F, i, j = _loft_ij(rings)
    xs = np.asarray(xs, float)
    len(th)
    th2 = np.r_[th, th[0] + 2 * math.pi]
    xm = np.where(i >= 0, 0.5 * (xs[np.maximum(i, 0)] + xs[np.minimum(np.maximum(i, 0) + 1, len(xs) - 1)]), np.nan)
    tm = 0.5 * (th2[np.maximum(j, 0)] + th2[np.maximum(j, 0) + 1])
    tm = np.degrees(np.arctan2(np.sin(tm), np.cos(tm)))
    tm = np.where(j >= 0, tm, np.nan)
    return V, F, xm, tm


def _angles(n: int, extra_deg=()):
    """Ring parameter angles: ``n`` uniform ones plus the extra (deg) ones, mirrored about the vertical plane."""
    base = np.linspace(-180, 180, n, endpoint=False)
    ex = np.asarray(list(extra_deg), float)
    allx = np.r_[base, ex, 180 - ex]
    allx = (allx + 180) % 360 - 180
    allx = np.unique(np.round(allx, 6))
    # drop angles closer than 1.2 deg to a neighbour (keep the extra ones)
    keep = [allx[0]]
    for a in allx[1:]:
        if a - keep[-1] > 1.2:
            keep.append(a)
        elif np.any(np.isclose(a, np.r_[ex, 180 - ex])):
            keep[-1] = a
    keep = np.array(keep)
    if keep[-1] - keep[0] > 358.8:
        keep = keep[:-1]
    return np.radians(keep)


def _flap(sec: np.ndarray, delta: float, hinge: float = 0.75, blend: float = 0.08) -> np.ndarray:
    """Deflect the aft part of a unit section trailing edge down by ``delta`` deg about a hinge at ``hinge`` x chord
    (a smooth bend over +-blend, a morphing plain flap)."""
    if abs(delta) < 1e-9:
        return sec
    out = sec.copy()
    w = np.clip((sec[:, 0] - (hinge - blend)) / (2 * blend), 0, 1)
    w = w * w * (3 - 2 * w)
    a = np.radians(delta) * w
    u, v = sec[:, 0] - hinge, sec[:, 1]
    out[:, 0] = hinge + u * np.cos(a) + v * np.sin(a)
    out[:, 1] = -u * np.sin(a) + v * np.cos(a)
    return out


def _lifting_surface(stations, n_sec, frame=None):
    """Loft through stations (le, chord, twist, t, m, p, flap); returns V, F, station index per face."""
    rings = []
    for s in stations:
        sec = kit.naca4(s.get("m", 0.0), s.get("p", 0.4), s["t"], n_sec)
        sec = _flap(sec, s.get("flap", 0.0))
        rings.append(kit.section_ring(sec, s["chord"], s.get("twist", 0.0), s["le"], frame, 0.25))
    return _loft_ij(rings)


def _blade(rs, chord, beta, tc, camber, sweep, n_sec, ax, sp, sense=1, round_root=0.0):
    """One rotor / propeller blade along the span direction ``sp`` from the rotation centre (origin): at each radius
    a NACA section of chord c(r), blade angle beta(r) (deg, from the rotation plane, leading edge toward the
    direction of motion, suction side toward the thrust direction ``ax``), LE swept back by sweep(r). ``sense`` = +1
    rotates right-handed about ``ax``. ``round_root``: radius below which the section blends to a circle (shank)."""
    ax, sp = np.asarray(ax, float), np.asarray(sp, float)
    e1 = sense * np.cross(sp, ax)            # chordwise, leading -> trailing edge (opposite to the motion)
    e2 = ax
    circ = kit.round_section(n_sec)
    rings = []
    for r in rs:
        c = chord(r)
        sec = kit.naca4(camber(r), 0.4, tc(r), n_sec)
        pv = 0.35
        if round_root > 0:
            w = float(np.clip((round_root * 1.6 - r) / (0.6 * round_root), 0, 1))
            if w > 0:
                sec = kit.blend_sections(sec, circ, w)
                pv = 0.35 + 0.15 * w
        le = r * sp - pv * c * e1 + sweep(r) * e1
        rings.append(kit.section_ring(sec, c, beta(r), le, frame=(e1, e2), pivot=pv))
    return _loft_ij(rings)


def _wheel(center, radius, width, rim_frac=0.62, n=28, nm=10, axis="y"):
    """Tyre + hub as one closed solid of revolution; returns V, F and per-face materials (tyre / wheel_metal)."""
    hw = width / 2
    t = np.linspace(0, math.pi, nm)
    # tyre shoulder: half super-ellipse in (axial, radial) from the hub disc to the tread and back
    xa = -hw * np.cos(t)
    rr = radius * rim_frac + (radius - radius * rim_frac) * np.sin(t) ** 0.35
    x = np.r_[-hw * 0.82, xa, hw * 0.82]
    r = np.r_[radius * rim_frac * 0.98, rr, radius * rim_frac * 0.98]
    x[1], x[-2] = -hw, hw
    V, F = kit.revolve(np.r_[x[0], x[1:-1], x[-1]], r, n, axis=axis, center=center)
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    c = V[F].mean(1) - np.asarray(center, float)
    rad = np.linalg.norm(np.delete(c, ax, axis=1), axis=1)
    mats = np.where(rad < radius * rim_frac * 1.01, "wheel_metal", "tyre")
    return V, F, mats


def _revolve_loop(profile_ra, n, center, axis="y"):
    """Torus-like closed solid: a closed (radius, axial) profile revolved about an axis through ``center``."""
    P = np.asarray(profile_ra, float)
    m = len(P)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr, aa = P[:, 0][None, :], P[:, 1][None, :]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    if axis == "y":
        X, Y, Z = rr * c, np.broadcast_to(aa, (n, m)), rr * s
    elif axis == "z":
        X, Y, Z = rr * c, rr * s, np.broadcast_to(aa, (n, m))
    else:
        X, Y, Z = np.broadcast_to(aa, (n, m)), rr * c, rr * s
    V = np.stack([X, Y, Z], -1).reshape(-1, 3) + np.asarray(center, float)
    i = np.arange(n)[:, None]
    j = np.arange(m)[None, :]
    a = i * m + j
    b = ((i + 1) % n) * m + j
    cc = ((i + 1) % n) * m + (j + 1) % m
    d = i * m + (j + 1) % m
    F = np.concatenate([np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([a, cc, d], -1).reshape(-1, 3)])
    from ..meshops import orient_outward
    return orient_outward(V, F)


def _window_inserts(centres, half_x, half_z, y_out, depth, n=12, side=1):
    """Rounded-rectangle window panes (prisms across y) at (x, z) centres: from y_out - depth to y_out (signed by
    side). Returns one merged mesh (one closed shell per pane)."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    ex = 4.0
    px = half_x * np.sign(c) * np.abs(c) ** (2 / ex)
    pz = half_z * np.sign(s) * np.abs(s) ** (2 / ex)
    out = []
    for (x, z), yo in zip(centres, y_out, strict=False):
        poly = np.stack([x + px, z + pz], 1)
        y0, y1 = (yo - depth, yo) if side > 0 else (-yo, -yo + depth)
        out.append(kit.extrude(poly, y0, y1, plane="xz"))
    return kit.merge(out)


def _se_half_width(hw, zb, zt, e, z):
    """Half-width of a super-ellipse section (hw, zb..zt, exponent e) at height z."""
    hh = 0.5 * (zt - zb)
    zc = 0.5 * (zt + zb)
    q = np.clip(np.abs((z - zc) / hh), 0, 1)
    return hw * (1 - q ** e) ** (1 / e)


# ============================================================================================ ATR 72-600
ATR_FACTS = ("ATR (2014). ATR 72-600 Aircraft Characteristics for Airport Planning (ATR 72 AC/AP), section 2 "
             "general dimensions; and ATR, 'ATR 72-600' product data sheet. https://www.atr-aircraft.com/"
             "our-aircraft/atr-72-600/")
HS568F = ("Collins Aerospace (Hamilton Sundstrand) 568F six-blade composite propeller, diameter 3.93 m, as fitted "
          "to the ATR 72-600 (EASA Type-Certificate Data Sheet P.009 for 568F propellers; ATR 72 TCDS EASA.A.084)")

ATR_MATERIALS = {
    "atr_white": {"color": [0.95, 0.96, 0.97], "metallic": 0.0, "roughness": 0.25, "clearcoat": 1.0},
    "wing_grey": {"color": [0.80, 0.82, 0.84], "metallic": 0.0, "roughness": 0.4},
    "livery_navy": {"color": [0.05, 0.14, 0.38], "metallic": 0.0, "roughness": 0.3, "clearcoat": 0.8},
    "livery_teal": {"color": [0.0, 0.56, 0.62], "metallic": 0.0, "roughness": 0.3, "clearcoat": 0.8},
    "prop_black": {"color": [0.07, 0.07, 0.08], "metallic": 0.0, "roughness": 0.45},
    "prop_tip": {"color": [0.85, 0.86, 0.86], "metallic": 0.6, "roughness": 0.3},
    "tyre": {"color": [0.04, 0.04, 0.045], "metallic": 0.0, "roughness": 0.85},
    "deice": {"color": [0.12, 0.12, 0.13], "metallic": 0.0, "roughness": 0.6},
}


def _atr_fuselage(res):
    """Fuselage outer mould line: nose 0-4.6 m, constant section 4.6-17.5 m, upswept tail cone to 27.17 m."""
    from scipy.interpolate import PchipInterpolator
    HW, ZB, ZT = 1.385, 1.00, 3.62
    xk = np.array([0.0, 0.08, 0.35, 0.8, 1.4, 2.0, 2.6, 3.2, 3.9, 4.6])
    ztk = np.array([2.06, 2.30, 2.55, 2.80, 3.02, 3.22, 3.40, 3.53, 3.60, ZT])
    zbk = np.array([1.94, 1.74, 1.52, 1.32, 1.17, 1.08, 1.03, 1.01, 1.0, ZB])
    hwk = np.array([0.02, 0.36, 0.66, 0.92, 1.12, 1.25, 1.33, 1.37, 1.384, HW])
    x_n = np.unique(np.r_[4.6 * np.linspace(0, 1, _n(16, res, 12)) ** 1.35, 1.75, 2.42, 2.50, 3.25, 3.32])
    x_c = np.linspace(4.6, 17.5, _n(4, res, 3))[1:]
    x_t = 17.5 + 9.67 * np.linspace(0, 1, _n(20, res, 14))[1:]
    xs = np.r_[x_n, x_c, x_t]
    hw, zb, zt, e = [], [], [], []
    for x in xs:
        if x <= 4.6:
            hw.append(float(PchipInterpolator(xk, hwk)(x)))
            zt.append(float(PchipInterpolator(xk, ztk)(x)))
            zb.append(float(PchipInterpolator(xk, zbk)(x)))
            e.append(2.0 + 0.2 * x / 4.6)
        elif x <= 17.5:
            hw.append(HW)
            zt.append(ZT)
            zb.append(ZB)
            e.append(2.2)
        else:
            s = min((x - 17.5) / 9.67, 1.0)
            hw.append(0.10 + (HW - 0.10) * (1 - s ** 1.8) ** 0.9)
            zt.append(ZT - 0.42 * s ** 2.2)
            zb.append(ZB + 2.02 * s ** 1.45)
            e.append(2.2 - 0.2 * s)
    hw, zb, zt, e = map(np.asarray, (hw, zb, zt, e))
    th = _angles(_n(32, res, 28), extra_deg=(-24, -16, -13, -9.5, 38, 50, 64, 86))
    V, F, xm, tm = _skin(xs, hw, zb, zt, e, th)
    a = np.where(np.abs(tm) > 90, np.sign(tm) * (180 - np.abs(tm)), tm)   # angle mirrored to the starboard side
    mats = np.full(len(F), "atr_white", dtype=object)
    # cheat-lines: navy band and a thin teal line, nose to tail cone
    mats[(xm > 0.6) & (a > -24) & (a < -16)] = "livery_navy"
    mats[(xm > 1.0) & (a > -13) & (a < -9.5)] = "livery_teal"
    # cockpit glazing: two front panes either side of the centre post, then the side windows
    front = (xm > 1.75) & (xm < 2.42) & (a > 50) & (a < 86)
    side = (xm > 2.50) & (xm < 3.25) & (a > 38) & (a < 64)
    mats[front | side] = "glass"
    mats[np.isnan(xm)] = "atr_white"
    sec = dict(xs=xs, hw=hw, zb=zb, zt=zt, e=e)
    return V, F, mats, sec


def _atr(p, res):
    bodies = []
    nsec = _n(18, res, 14)
    V, F, mats, fus = _atr_fuselage(res)
    bodies.append(Body("fuselage", V, F, mats, "fuselage"))
    # ---------------------------------------------------------------- cabin windows (inserts) both sides
    win_x = 5.95 + 0.762 * np.arange(15)
    zc_w = 2.42
    yo = [float(_se_half_width(1.385, 1.0, 3.62, 2.2, zc_w)) + 0.010 for _ in win_x]
    cen = [(x, zc_w) for x in win_x]
    nw = _n(10, res, 8)
    for sd, nm in ((1, "windows_starboard"), (-1, "windows_port")):
        Vw, Fw = _window_inserts(cen, 0.125, 0.17, yo, 0.06, nw, sd)
        bodies.append(Body(nm, Vw, Fw, "glass", "fuselage", aero=False))
    # passenger door outline (aft, port) and the forward cargo door (port): thin dark frames as inserts
    # ---------------------------------------------------------------- wing (one loft tip to tip)
    b2 = p["wing_span"] / 2
    lam = 0.55
    y_brk = 4.6
    c0 = (p["wing_area"] / 2) / (y_brk + (b2 - y_brk) * (1 + lam) / 2)
    ct = lam * c0
    x_le0, z_w = 11.0, 3.80
    sweep_le = math.radians(3.1)
    dih = math.radians(2.0)
    y_flap = 0.70 * b2
    ny = _n(5, res, 4)
    ys = np.unique(np.r_[0.0, 1.0, 2.6, y_brk, np.linspace(y_brk, b2 - 0.35, ny + 2)[1:], y_flap, y_flap + 0.2, y_flap + 0.4, y_flap + 0.6,
                         b2 - 0.12, b2])
    st = []
    for yy in np.r_[-ys[::-1], ys[1:]]:
        ay = abs(yy)
        if ay <= y_brk:
            c, le = c0, x_le0
            z = z_w
        else:
            f = (ay - y_brk) / (b2 - y_brk)
            c = c0 + (ct - c0) * f
            le = x_le0 + (ay - y_brk) * math.tan(sweep_le)
            z = z_w + (ay - y_brk) * math.tan(dih)
        t = 0.18 - 0.05 * min(ay / b2, 1.0)
        if ay > b2 - 0.13:          # rounded tip
            k = (ay - (b2 - 0.13)) / 0.13
            le += 0.08 * c * k
            c *= 1 - 0.25 * k
            t *= 1 - 0.55 * k
        tw = 1.0 - 3.0 * min(ay / b2, 1.0)
        wf = float(np.clip(1 - (ay - y_flap) / 0.6, 0, 1))      # flap side edge blended over 0.6 m
        fl = p["flap_deg"] * wf * wf * (3 - 2 * wf)
        st.append({"le": (le, yy, z), "chord": c, "twist": tw, "t": t, "m": 0.03, "p": 0.4, "flap": fl})
    Vwg, Fwg, iw, jw = _lifting_surface(st, nsec, frame=((1.0, 0, 0), (0, 0, 1.0)))
    # de-icing boots: dark leading-edge strip on the outer wing
    cen_w = Vwg[Fwg].mean(1)
    2 * nsec - 2
    jj = np.where(jw >= 0, jw, 0)
    le_band = (jw >= 0) & (np.abs(jj - (nsec - 1)) <= 2) & (np.abs(cen_w[:, 1]) > 1.6)
    wm = np.where(le_band, "deice", "wing_grey").astype(object)
    bodies.append(Body("wing", Vwg, Fwg, wm, "wing"))
    # wing-to-fuselage fairing on the fuselage top
    xf = np.linspace(10.2, 15.6, _n(12, res, 8))
    s = (xf - xf[0]) / (xf[-1] - xf[0])
    hwf = 0.95 * np.sqrt(np.clip(np.sin(math.pi * s ** 0.7), 0, 1)) + 0.02
    ztf = 3.45 + 0.55 * np.sin(math.pi * np.clip(s ** 0.7, 0, 1)) ** 0.6
    ztf[0] = ztf[-1] = 3.5
    zbf = np.full_like(xf, 3.10)
    bodies.append(Body("wing_fairing", *kit.body_of_sections(xf, hwf, zbf, ztf, _n(28, res, 20), 2.4), "atr_white",
                       "fuselage"))
    # ---------------------------------------------------------------- nacelles, spinners, intakes, propellers
    yn, zp, xp = 4.05, 3.40, 8.45
    xs_n = np.array([8.75, 9.1, 9.6, 10.3, 11.2, 12.4, 13.4, 14.3, 15.0, 15.6, 16.0])
    hw_n = np.array([0.38, 0.41, 0.45, 0.47, 0.47, 0.46, 0.43, 0.37, 0.29, 0.17, 0.04])
    zb_n = np.array([3.02, 2.90, 2.80, 2.76, 2.78, 2.86, 2.98, 3.14, 3.30, 3.46, 3.58])
    zt_n = np.array([3.78, 3.86, 3.95, 4.02, 4.05, 4.05, 4.0, 3.94, 3.88, 3.82, 3.76])
    nr = _n(26, res, 20)
    R = p["prop_diameter"] / 2
    nb = 6
    ns = _n(10, res, 8)
    nsb = _n(9, res, 7)
    rs = R * (0.11 + 0.89 * (1 - np.cos(np.linspace(0, math.pi / 2, ns))) ** 0.8)
    rs = np.unique(np.r_[rs, 0.93 * R])
    b75 = p["blade_pitch_deg"]
    def chord(r):
        return 0.24 + 0.13 * math.sin(math.pi * min(r / R, 1.0) ** 0.8) - 0.16 * max(r / R - 0.85, 0) / 0.15
    def beta(r):
        return math.degrees(math.atan(math.tan(math.radians(b75)) * 0.75 * R / max(r, 0.2)))
    def tc(r):
        return 0.22 - 0.16 * min(r / R, 1.0) ** 0.7
    def cam(r):
        return 0.03
    def sweep(r):
        return 0.10 * max(r / R - 0.6, 0) ** 2 / 0.16
    for side, tag in ((1, "starboard"), (-1, "port")):
        yc = side * yn
        Vn, Fn = kit.body_of_sections(xs_n, hw_n, zb_n, zt_n, nr, 2.3)
        bodies.append(Body(f"nacelle_{tag}", Vn + [0, yc, 0], Fn, "atr_white", "nacelles"))
        # chin intake below the spinner: front face dark
        xi = np.array([8.82, 9.0, 9.4, 10.0, 10.8])
        hwi = np.array([0.20, 0.22, 0.24, 0.26, 0.20])
        zbi = np.array([2.62, 2.60, 2.62, 2.68, 2.80])
        zti = np.array([2.92, 2.94, 3.0, 3.1, 3.2])
        Vi, Fi = kit.body_of_sections(xi, hwi, zbi, zti, _n(20, res, 16), 3.0)
        ci = Vi[Fi].mean(1)
        mi = np.where(ci[:, 0] < 8.83, "anodized_black", "atr_white")
        bodies.append(Body(f"intake_{tag}", Vi + [0, yc, 0], Fi, mi, "nacelles"))
        # spinner
        xsp = np.linspace(7.82, 8.85, _n(8, res, 7))
        rsp = 0.385 * np.clip(1 - ((8.85 - xsp) / 1.03) ** 2.0, 0, 1) ** 0.5
        rsp[0] = 0.0
        Vs, Fs = kit.revolve(xsp, rsp, _n(20, res, 16), "x", center=(0, yc, zp))
        bodies.append(Body(f"spinner_{tag}", Vs, Fs, "wing_grey", "propellers"))
        # six blades
        for k in range(nb):
            Vb, Fb, ib, jb = _blade(rs, chord, beta, tc, cam, sweep, nsb, ax=(-1.0, 0, 0), sp=(0, 0, 1.0),
                                    sense=1, round_root=0.18 * R)
            az = p["prop_azimuth_deg"] + 360.0 / nb * k + (30.0 if side < 0 else 0.0)
            Vb = Vb @ kit.rot((1, 0, 0), az).T + [xp, yc, zp]
            rm = np.where(ib >= 0, rs[np.maximum(ib, 0)], R)
            bm = np.where(rm >= 0.93 * R - 1e-9, "prop_tip", "prop_black")
            bodies.append(Body(f"prop_{tag}_blade_{k + 1}", Vb, Fb, bm, "propellers"))
    # ---------------------------------------------------------------- vertical fin with dorsal fillet, T-tail
    zk = np.array([2.9, 3.85, 4.45, 7.48])
    lek = np.array([18.6, 19.7, 21.9, 24.45])
    chk = np.array([8.35, 7.25, 5.05, 2.15])
    tk = np.array([0.055, 0.075, 0.12, 0.12])
    zf = np.unique(np.r_[zk, 4.85, np.linspace(4.85, 7.48, _n(5, res, 4))])
    fst = [{"le": (np.interp(z, zk, lek), 0.0, z), "chord": np.interp(z, zk, chk), "t": np.interp(z, zk, tk)}
           for z in zf]
    Vf, Ff, i_f, _ = _lifting_surface(fst, nsec, frame=((1.0, 0, 0), (0, 1.0, 0)))
    cf = Vf[Ff].mean(1)
    fm = np.where(cf[:, 2] > 4.85, "livery_navy", "atr_white").astype(object)
    fm[(cf[:, 2] > 4.45) & (cf[:, 2] <= 4.85)] = "livery_teal"
    bodies.append(Body("fin", Vf, Ff, fm, "empennage"))
    bh = p["htail_span"] / 2
    zh = 7.50
    hst = []
    for yy in np.r_[-bh, -0.97 * bh, -0.5 * bh, 0.0, 0.5 * bh, 0.97 * bh, bh]:
        f = abs(yy) / bh
        c = 1.95 - 0.85 * f
        le = 24.50 + 0.80 * f
        tt = 0.11 * (1 - 0.5 * max(f - 0.97, 0) / 0.03)
        hst.append({"le": (le, yy, zh), "chord": c * (1 - 0.15 * max(f - 0.97, 0) / 0.03), "twist": -1.5, "t": tt})
    Vh, Fh, _, _ = _lifting_surface(hst, nsec, frame=((1.0, 0, 0), (0, 0, 1.0)))
    bodies.append(Body("horizontal_tail", Vh, Fh, "atr_white", "empennage"))
    bodies.append(Body("tail_bullet", *kit.ellipsoid((25.35, 0, 7.50), (1.35, 0.17, 0.15), _n(20, res, 16),
                                                        _n(10, res, 8)), "livery_navy", "empennage"))
    # ---------------------------------------------------------------- landing gear: sponsons, twin wheels
    x_mg, track, wb = 13.55, 4.10, 10.77
    rw, ww = 0.41, 0.23
    xsg = np.linspace(11.3, 16.8, _n(12, res, 9))
    s = (xsg - xsg[0]) / (xsg[-1] - xsg[0])
    hws = 0.03 + 0.62 * np.clip(np.sin(math.pi * s) ** 0.45, 0, 1)
    zbs = 0.86 + 0.75 * (1 - np.sin(math.pi * s) ** 0.5)
    zts = np.full_like(xsg, 2.05)
    for side, tag in ((1, "starboard"), (-1, "port")):
        Vsp, Fsp = kit.body_of_sections(xsg, hws, zbs, zts, _n(24, res, 18), 2.5)
        bodies.append(Body(f"sponson_{tag}", Vsp + [0, side * 1.52, 0], Fsp, "atr_white", "landing_gear"))
        yg = side * track / 2
        for k, dy in enumerate((-0.15, 0.15)):
            Vw_, Fw_, mw = _wheel((x_mg, yg + dy, rw), rw, ww, n=_n(18, res, 14), nm=_n(6, res, 5))
            bodies.append(Body(f"main_wheel_{tag}_{k + 1}", Vw_, Fw_, mw, "landing_gear", aero=False))
        bodies.append(Body(f"main_axle_{tag}", *kit.cylinder((x_mg, yg - 0.27, rw), (x_mg, yg + 0.27, rw), 0.05, 10),
                           "steel", "landing_gear", aero=False))
        bodies.append(Body(f"main_leg_{tag}", *kit.cylinder((x_mg + 0.05, yg, rw), (x_mg - 0.25, side * 1.85, 1.15),
                                                            0.08, 12), "steel", "landing_gear", aero=False))
    xng = x_mg - wb
    rn = 0.225
    for k, dy in enumerate((-0.17, 0.17)):
        Vw_, Fw_, mw = _wheel((xng, dy, rn), rn, 0.14, n=_n(16, res, 12), nm=_n(6, res, 5))
        bodies.append(Body(f"nose_wheel_{k + 1}", Vw_, Fw_, mw, "landing_gear", aero=False))
    bodies.append(Body("nose_leg", *kit.cylinder((xng, 0, rn), (xng - 0.1, 0, 1.45), 0.06, 12), "steel",
                       "landing_gear", aero=False))
    bodies.append(Body("nose_axle", *kit.cylinder((xng, -0.26, rn), (xng, 0.26, rn), 0.035, 10), "steel",
                       "landing_gear", aero=False))
    return bodies


register(GeometrySpec(
    name="turboprop_atr72", title="Regional twin turboprop (ATR 72-600 class)", category="aircraft",
    description=("High-wing regional twin turboprop on its gear: fuselage with cabin windows, cockpit glazing and a "
                 "livery, untapered inner / tapered outer wing with a morphing plain flap and leading-edge de-icing "
                 "boots, two nacelles with chin intakes and spinners, six-blade propellers lofted with constant "
                 "geometric pitch, T-tail with dorsal fin, main-gear sponsons with twin wheels and a twin nose "
                 "wheel."),
    builder=_atr,
    params=[
        Param("wing_span", 27.05, "m", (24.0, 30.0), "wing span (tip to tip)"),
        Param("wing_area", 61.0, "m^2", (50.0, 70.0), "reference wing area (sets the chords at a fixed taper)"),
        Param("flap_deg", 0.0, "deg", (0.0, 35.0), "flap deflection, inboard 70 % of the half-span "
                                                   "(ATR settings 0 / 15 / 30)"),
        Param("prop_diameter", 3.93, "m", (3.4, 4.2), "propeller diameter"),
        Param("blade_pitch_deg", 32.0, "deg", (10.0, 85.0), "blade angle at 0.75 R (constant geometric pitch "
                                                             "distribution; ~80 = feathered)"),
        Param("prop_azimuth_deg", 0.0, "deg", (0.0, 60.0), "propeller azimuth (blade 1 up at 0)"),
        Param("htail_span", 7.31, "m", (6.5, 8.0), "horizontal tail span"),
    ],
    dimensions={"length_x": (27.17, 0.02), "width_y": (27.05, 0.02), "height_z": (7.65, 0.02)},
    frame=FRAME_VEHICLE.replace("z = 0 on the ground for ground vehicles", "z = 0 on the ground (aircraft on its "
                                                                         "gear, tyres touching z = 0)"),
    sources=[Source("overall length 27.17 m, span 27.05 m, height 7.65 m, wing area 61.0 m^2, wheel track 4.10 m, "
                    "wheelbase 10.77 m, six-blade propellers of 3.93 m", ATR_FACTS),
             Source("propeller diameter 3.93 m, six composite blades", HS568F),
             Source("fuselage maximum width 2.77 m and height 2.62 m (published external diameter quoted 2.77-2.87 m;"
                    " 2.77 m used)", "estimate from ATR 72 airport-planning cross-section", estimate=True),
             Source("wing planform: constant chord to 4.6 m, taper 0.55 outboard, 3.1 deg LE sweep, 2 deg outer "
                    "dihedral, NACA 3418 -> 3413 stand-ins for the ATR sections, 1 deg -> -2 deg twist",
                    "engineering estimates from public three-views", estimate=True),
             Source("nacelle, sponson, fin / dorsal fillet, tailplane chords, blade chord / thickness / sweep, "
                    "main tyre 0.82 m and nose tyre 0.45 m diameters, gear position", "estimates from photographs "
                    "and three-views", estimate=True),
             Source("tailplane span 7.31 m", "ATR 72 three-view dimension (published in Jane's All the World's "
                    "Aircraft; treated as approximate)", estimate=True)],
    physics=["external aerodynamics of a high-wing turboprop (RANS, take-off / landing flap settings)",
             "propeller-wing interaction and slipstream (actuator disc / sliding-mesh blades)",
             "propeller aero-acoustics and cabin noise sources", "icing on the wing and tail leading edges",
             "ground effect and crosswind on the gear"],
    materials=ATR_MATERIALS, tags=("regional aircraft", "turboprop", "propeller", "high wing", "T-tail"),
    camera={"azimuth": 215.0, "elevation": 20.0},
))


# ============================================================================================ H145
H145_FACTS = ("Airbus Helicopters (2023). H145 technical data / H145 Technical Information brochure: main rotor "
              "diameter 11.00 m, length (rotors turning) 13.64 m, fuselage length 11.64 m, height 3.95 m. "
              "https://www.airbus.com/en/products-services/helicopters/civil-helicopters/h145")
H145_TCDS = ("EASA (2023). Type-Certificate Data Sheet R.010 (MBB-BK 117 / H145): four-blade bearingless main "
             "rotor, ducted (Fenestron) tail rotor on the D-3 / H145 variants")

H145_MATERIALS = {
    "heli_red": {"color": [0.72, 0.06, 0.07], "metallic": 0.0, "roughness": 0.28, "clearcoat": 1.0},
    "heli_white": {"color": [0.94, 0.95, 0.96], "metallic": 0.0, "roughness": 0.28, "clearcoat": 1.0},
    "heli_dark": {"color": [0.10, 0.11, 0.13], "metallic": 0.0, "roughness": 0.4, "clearcoat": 0.6},
    "rotor_grey": {"color": [0.22, 0.23, 0.25], "metallic": 0.0, "roughness": 0.45},
    "rotor_tip": {"color": [0.85, 0.75, 0.15], "metallic": 0.0, "roughness": 0.4},
    "skid_metal": {"color": [0.55, 0.57, 0.60], "metallic": 0.9, "roughness": 0.35},
}


def _h145_pod(res):
    from scipy.interpolate import PchipInterpolator
    xk = np.array([0.0, 0.04, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.6, 4.4, 5.0, 5.6, 6.2, 6.9])
    hwk = np.array([0.02, 0.25, 0.50, 0.68, 0.80, 0.86, 0.88, 0.88, 0.88, 0.86, 0.76, 0.58, 0.42, 0.28])
    zbk = np.array([0.98, 0.86, 0.74, 0.64, 0.58, 0.55, 0.54, 0.54, 0.54, 0.56, 0.70, 1.00, 1.28, 1.45])
    ztk = np.array([1.12, 1.36, 1.62, 1.85, 2.02, 2.12, 2.16, 2.16, 2.16, 2.14, 2.10, 2.00, 1.88, 1.78])
    marks = [0.10, 0.95, 1.40, 1.50, 2.0, 2.35, 2.55, 3.85, 4.05, 4.65]
    xs = np.unique(np.r_[np.linspace(0, 1, _n(8, res, 7)) ** 1.8 * 1.4, np.linspace(1.4, 6.9, _n(12, res, 9)),
                         marks])
    hw = PchipInterpolator(xk, hwk)(xs)
    zb = PchipInterpolator(xk, zbk)(xs)
    zt = PchipInterpolator(xk, ztk)(xs)
    e = np.interp(xs, [0, 1.5, 5, 6.9], [2.1, 2.6, 2.6, 2.2])
    th = _angles(_n(32, res, 28), extra_deg=(-40, -12, -6, 6, 12, 36, 44, 50, 62, 82))
    V, F, xm, tm = _skin(xs, hw, zb, zt, e, th)
    a = np.where(np.abs(tm) > 90, np.sign(tm) * (180 - np.abs(tm)), tm)
    mats = np.full(len(F), "heli_white", dtype=object)
    mats[(a < -12) & ((xm > 0.95) | (a < -40))] = "heli_red"
    glass = ((xm > 0.10) & (xm < 1.40) & (a > 6) & (a < 82))                  # windscreen (centre post above 82)
    glass |= (xm > 0.10) & (xm < 0.95) & (a > -40) & (a < -6)                 # chin windows
    glass |= (xm > 1.40) & (xm < 2.0) & (a > 62) & (a < 82)                   # roof windows
    glass |= (xm > 1.50) & (xm < 2.35) & (a > 12) & (a < 50)                  # cockpit door
    glass |= (xm > 2.55) & (xm < 3.85) & (a > 12) & (a < 44)                  # sliding cabin door
    glass |= (xm > 4.05) & (xm < 4.65) & (a > 12) & (a < 36)                  # aft cabin window
    mats[glass] = "glass"
    mats[(xm < 0.10) | np.isnan(xm)] = "heli_red"                            # nose cap
    return V, F, mats


def _h145(p, res):
    bodies = []
    V, F, m = _h145_pod(res)
    bodies.append(Body("fuselage_pod", V, F, m, "fuselage"))
    nt = _n(22, res, 16)
    # tail boom (slight upsweep), tapering
    xb = np.linspace(5.6, 10.35, _n(10, res, 7))
    sb = (xb - xb[0]) / (xb[-1] - xb[0])
    rb = 0.33 - 0.15 * sb
    zb_c = 1.55 + 0.30 * sb
    rings = []
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    for x, r, z in zip(xb, rb, zb_c, strict=False):
        rings.append(np.stack([np.full(nt, x), r * np.cos(th), z + 1.08 * r * np.sin(th)], 1))
    Vb, Fb, ib, jb = _loft_ij(rings)
    thm = (np.maximum(jb, 0) + 0.5) * 2 * math.pi / nt
    bm = np.where((jb >= 0) & (np.sin(thm) < -0.2), "heli_red", "heli_white")
    bodies.append(Body("tail_boom", Vb, Fb, bm, "fuselage"))
    # engine cowling and main-gearbox fairing on the roof
    xc = np.linspace(2.0, 6.6, _n(14, res, 10))
    sc = (xc - xc[0]) / (xc[-1] - xc[0])
    hwc = 0.03 + 0.62 * np.clip(np.sin(math.pi * sc ** 0.75), 0, 1) ** 0.4
    ztc = 1.90 + 0.72 * np.clip(np.sin(math.pi * sc ** 0.7), 0, 1) ** 0.45
    zbc = np.full_like(xc, 1.70)
    xc = np.unique(np.r_[xc, 2.75, 3.25])
    sc = (xc - xc[0]) / (xc[-1] - xc[0])
    hwc = 0.03 + 0.62 * np.clip(np.sin(math.pi * sc ** 0.75), 0, 1) ** 0.4
    ztc = 2.00 + 0.72 * np.clip(np.sin(math.pi * sc ** 0.7), 0, 1) ** 0.45
    zbc = np.full_like(xc, 1.90)
    Vc, Fc, xmc, tmc = _skin(xc, hwc, zbc, ztc, 3.0, _angles(_n(28, res, 22), extra_deg=(8, 34)))
    ac = np.where(np.abs(tmc) > 90, np.sign(tmc) * (180 - np.abs(tmc)), tmc)
    intake = (xmc > 2.75) & (xmc < 3.25) & (ac > 8) & (ac < 34)
    bodies.append(Body("engine_cowling", Vc, Fc, np.where(intake, "heli_dark", "heli_white").astype(object),
                       "fuselage"))
    for side in (1, -1):
        path = np.array([[5.30, side * 0.36, 2.32], [5.70, side * 0.48, 2.40], [6.05, side * 0.60, 2.52]])
        bodies.append(Body(f"exhaust_{'starboard' if side > 0 else 'port'}", *kit.tube(path, 0.11, _n(14, res, 10)),
                           "anodized_black", "fuselage", aero=False))
    # main rotor: mast, hub, four blades
    xh, zh = 3.50, 3.28
    Rm = p["rotor_diameter"] / 2
    bodies.append(Body("mast_fairing", *kit.revolve([2.5, 2.8, 3.02, 3.16], [0.42, 0.30, 0.17, 0.0],
                                                    _n(20, res, 16), "z", center=(xh, 0, 0)), "heli_white",
                       "rotor"))
    bodies.append(Body("rotor_hub", *kit.cylinder((xh, 0, 3.12), (xh, 0, 3.40), 0.24, _n(20, res, 16)), "rotor_grey",
                       "rotor"))
    bodies.append(Body("hub_cap", *kit.revolve([3.40, 3.46, 3.52], [0.20, 0.15, 0.0], _n(18, res, 14), "z",
                                               center=(xh, 0, 0)), "rotor_grey", "rotor"))
    coll, tw = p["collective_deg"], -10.0
    ns = _n(14, res, 10)
    nsb = _n(11, res, 8)
    r0 = 0.30
    rs = r0 + (Rm - r0) * (1 - np.cos(np.linspace(0, math.pi / 2, ns))) ** 0.7
    rs = np.unique(np.r_[rs, 1.6, 1.75, Rm - 0.35, Rm - 0.12, Rm - 0.04])
    c_bl = 0.31

    def chord(r):
        if r < 1.6:                       # flexbeam / control cuff
            return 0.20 + 0.09 * (r - r0) / (1.6 - r0)
        if r > Rm - 0.35:                 # tapered, slightly swept tip
            k = (r - (Rm - 0.35)) / 0.35
            return c_bl * (1 - 0.55 * k ** 1.5)
        return c_bl

    def beta(r):
        return coll + tw * (r / Rm - 0.75)
    def tc(r):
        return 0.30 if r < 1.6 else 0.12 if r < Rm - 0.35 else 0.12 - 0.03 * (r - (Rm - 0.35)) / 0.35
    def cam(r):
        return 0.0 if r < 1.6 else 0.01
    def sweep(r):
        return 0.10 * max(r - (Rm - 0.35), 0) / 0.35
    cone = p["coning_deg"]
    for k in range(4):
        Vb_, Fb_, ib_, _ = _blade(rs, chord, beta, tc, cam, sweep, nsb, ax=(0, 0, 1.0), sp=(1.0, 0, 0), sense=1,
                                  round_root=0.55)
        Vb_ = Vb_ @ kit.rot((0, 1, 0), -cone).T
        Vb_ = Vb_ @ kit.rot((0, 0, 1), 180.0 + p["rotor_azimuth_deg"] + 90.0 * k).T + [xh, 0, zh]
        rm = np.where(ib_ >= 0, rs[np.maximum(ib_, 0)], Rm)
        bm = np.where(rm >= Rm - 0.36, "rotor_tip", "rotor_grey").astype(object)
        bodies.append(Body(f"main_blade_{k + 1}", Vb_, Fb_, bm, "rotor"))
    # Fenestron shroud, fin, ventral fin, fan
    xfc, zfc = 10.82, 1.90
    rin, rout, hy = 0.56, 0.82, 0.17
    t = np.linspace(0, 2 * math.pi, _n(16, res, 12), endpoint=False)
    c, s = np.cos(t), np.sin(t)
    prof = np.stack([0.5 * (rin + rout) + 0.5 * (rout - rin) * np.sign(c) * np.abs(c) ** 0.5,
                     hy * np.sign(s) * np.abs(s) ** 0.5], 1)
    Vsh, Fsh = _revolve_loop(prof, _n(36, res, 28), (xfc, 0, zfc), "y")
    csh = Vsh[Fsh].mean(1)
    rad = np.hypot(csh[:, 0] - xfc, csh[:, 2] - zfc)
    bodies.append(Body("fenestron_shroud", Vsh, Fsh, np.where(rad < rin + 0.02, "heli_dark", "heli_red")
                       .astype(object), "tail"))
    nsec = _n(14, res, 10)
    fst = []
    for z, le, ch, tt in ((2.40, 10.15, 1.30, 0.20), (2.75, 10.50, 1.05, 0.16), (3.60, 10.90, 0.72, 0.13),
                          (3.95, 11.02, 0.58, 0.12)):
        fst.append({"le": (le, 0.0, z), "chord": ch, "t": tt, "m": 0.0})
    Vf, Ff, _, _ = _lifting_surface(fst, nsec, frame=((1.0, 0, 0), (0, 1.0, 0)))
    bodies.append(Body("fin", Vf, Ff, "heli_red", "tail"))
    vst = []
    for z, le, ch, tt in ((1.45, 10.25, 1.05, 0.18), (1.25, 10.62, 0.70, 0.14), (1.00, 10.90, 0.45, 0.12)):
        vst.append({"le": (le, 0.0, z), "chord": ch, "t": tt})
    Vv, Fv, _, _ = _lifting_surface(vst, nsec, frame=((1.0, 0, 0), (0, 1.0, 0)))
    bodies.append(Body("ventral_fin", Vv, Fv, "heli_red", "tail"))
    bodies.append(Body("tail_skid", *kit.tube(np.array([[10.85, 0, 1.08], [11.05, 0, 0.98], [11.25, 0, 0.95]]),
                                              0.025, 8), "skid_metal", "tail", aero=False))
    bodies.append(Body("fan_hub", *kit.cylinder((xfc, -0.12, zfc), (xfc, 0.10, zfc), 0.15, _n(18, res, 14)),
                       "rotor_grey", "tail"))
    bodies.append(Body("fan_strut", *kit.tube(np.array([[10.12, 0.02, zfc - 0.02], [10.45, 0.02, zfc],
                                                        [xfc, 0.02, zfc]]), 0.05, 10), "rotor_grey", "tail"))
    rf = np.linspace(0.10, rin - 0.02, _n(5, res, 4))
    for k in range(10):
        Vfb, Ffb, _, _ = _blade(rf, lambda r: 0.085, lambda r: 22.0 + 12 * (0.3 - r), lambda r: 0.10,
                                lambda r: 0.03, lambda r: 0.0, _n(8, res, 6), ax=(0, 1.0, 0), sp=(1.0, 0, 0))
        az = 360.0 / 10 * k + 7.0 * math.sin(k)          # uneven blade spacing (acoustic modulation)
        Vfb = Vfb @ kit.rot((0, 1, 0), az).T + [xfc, 0, zfc]
        bodies.append(Body(f"fan_blade_{k + 1}", Vfb, Ffb, "rotor_grey", "tail", aero=False))
    # horizontal stabiliser with endplates
    bs = 1.38
    hst = [{"le": (9.15 + 0.05 * abs(yy) / bs, yy, 1.80), "chord": 0.58 - 0.08 * abs(yy) / bs, "t": 0.13,
            "m": 0.03, "twist": -3.0, "p": 0.4} for yy in (-bs, -0.6, 0.0, 0.6, bs)]
    for st_ in hst:                       # inverted camber (down-load stabiliser)
        st_["m"] = 0.0
    Vh, Fh, _, _ = _lifting_surface(hst, nsec, frame=((1.0, 0, 0), (0, 0, 1.0)))
    bodies.append(Body("horizontal_stabiliser", Vh, Fh, "heli_white", "tail"))
    for side, tag in ((1, "starboard"), (-1, "port")):
        est = [{"le": (9.10 + 0.08 * f, side * (bs - 0.01) + side * 0.06 * f, 1.48 + 0.58 * f),
                "chord": 0.66 - 0.18 * f, "t": 0.08} for f in (0.0, 0.5, 1.0)]
        Ve, Fe, _, _ = _lifting_surface(est, _n(10, res, 8), frame=((1.0, 0, 0), (0, 1.0, 0)))
        bodies.append(Body(f"endplate_{tag}", Ve, Fe, "heli_red", "tail"))
    # skid landing gear: two skids, two arched cross tubes
    ys = 1.10
    ntb = _n(12, res, 10)
    for side, tag in ((1, "starboard"), (-1, "port")):
        u = np.linspace(0, 1, 6)
        nose = np.stack([0.95 - 0.55 * np.sin(u * math.pi / 2), np.full(6, side * ys),
                         0.045 + 0.30 * (1 - np.cos(u * math.pi / 2)) ** 1.3], 1)
        path = np.r_[np.array([[4.75, side * ys, 0.045], [3.5, side * ys, 0.045], [2.2, side * ys, 0.045]]), nose]
        bodies.append(Body(f"skid_{tag}", *kit.tube(path, 0.04, ntb), "skid_metal", "landing_gear", aero=False))
    for k, xx in enumerate((1.45, 3.85)):
        yy = np.linspace(-ys, ys, 13)
        zz = 0.045 + 0.58 * (1 - np.abs(yy / ys) ** 3.5)
        path = np.stack([np.full_like(yy, xx), yy, zz], 1)
        bodies.append(Body(f"cross_tube_{k + 1}", *kit.tube(path, 0.042, ntb), "skid_metal", "landing_gear",
                           aero=False))
    return bodies


register(GeometrySpec(
    name="helicopter_h145", title="Light twin helicopter (H145 class)", category="rotorcraft",
    description=("Light twin-engine helicopter: pod-and-boom fuselage with wrap-around glazing and a livery, engine "
                 "cowling with exhausts, four-blade bearingless main rotor (NACA 0012/1312-type sections, -10 deg "
                 "linear twist, collective, coning, azimuth), ducted Fenestron-type tail rotor with ten unevenly "
                 "spaced blades in a shroud with fin and ventral fin, horizontal stabiliser with endplates and a "
                 "skid landing gear; standing on the skids."),
    builder=_h145,
    params=[
        Param("rotor_diameter", 11.0, "m", (10.0, 12.0), "main-rotor diameter"),
        Param("collective_deg", 6.0, "deg", (-2.0, 16.0), "collective blade pitch at 0.75 R"),
        Param("rotor_azimuth_deg", 0.0, "deg", (0.0, 90.0), "rotor azimuth (0 = blade 1 over the nose)"),
        Param("coning_deg", 2.5, "deg", (-2.0, 6.0), "blade coning angle (negative = static droop)"),
    ],
    dimensions={"length_x": (13.64, 0.02), "width_y": (11.0, 0.02), "height_z": (3.95, 0.02)},
    frame=FRAME_VEHICLE.replace("z = 0 on the ground for ground vehicles", "z = 0 on the ground under the skids; "
                                "length includes the main rotor turning (blade 1 over the nose at azimuth 0)"),
    sources=[Source("main-rotor diameter 11.00 m, length rotors turning 13.64 m, fuselage length 11.64 m, height "
                    "3.95 m", H145_FACTS),
             Source("four-blade bearingless main rotor, Fenestron tail rotor", H145_TCDS),
             Source("blade chord 0.31 m, -10 deg twist, NACA 0012/1312 stand-ins (the actual DM-H3/H4 sections "
                    "are not reproduced), Fenestron ~1.1 m duct with 10 blades, rotor-hub height 3.28 m, cabin "
                    "width 1.74 m, skid track 2.2 m, stabiliser span 2.76 m", "estimates from photographs and "
                    "three-views", estimate=True)],
    physics=["rotor aerodynamics in hover and forward flight (BEM, actuator disc, blade-resolved CFD)",
             "rotor-fuselage interaction and download on the stabiliser", "ducted tail-rotor (Fenestron) "
             "performance and acoustics", "brown-out / downwash near the ground", "ground resonance on skids"],
    materials=H145_MATERIALS, tags=("helicopter", "rotor", "fenestron", "EMS"),
    camera={"azimuth": 215.0, "elevation": 20.0},
))


# ============================================================================================ quadrotor F450
F450_FACTS = ("DJI (2013). Flame Wheel F450 user manual / specifications: diagonal wheelbase 450 mm, frame weight "
              "282 g, recommended 10 x 4.5 in propellers and 2212 motors. https://www.dji.com/flame-wheel-arf")

DRONE_MATERIALS = {
    "arm_red": {"color": [0.78, 0.08, 0.06], "metallic": 0.0, "roughness": 0.4},
    "arm_white": {"color": [0.92, 0.92, 0.90], "metallic": 0.0, "roughness": 0.4},
    "canopy": {"color": [0.96, 0.96, 0.97], "metallic": 0.0, "roughness": 0.2, "clearcoat": 1.0},
    "prop_cw": {"color": [0.06, 0.06, 0.07], "metallic": 0.0, "roughness": 0.35},
    "prop_ccw": {"color": [0.10, 0.10, 0.11], "metallic": 0.0, "roughness": 0.35},
    "battery": {"color": [0.10, 0.25, 0.65], "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.5},
    "strap": {"color": [0.05, 0.05, 0.05], "metallic": 0.0, "roughness": 0.9},
    "foam": {"color": [0.08, 0.08, 0.08], "metallic": 0.0, "roughness": 0.9},
}


def _quad(p, res):
    bodies = []
    D = p["arm_diagonal"]
    Rarm = D / 2
    Rp = p["prop_diameter"] / 2
    P = p["prop_pitch"]
    z_bot = 0.140                                    # bottom plate underside above the ground
    t_pl = 0.0016
    h_arm = 0.030
    z_top = z_bot + t_pl + h_arm                    # top plate underside
    nt = _n(20, res, 14)
    # centre plates (PCB): octagon-ish with arm tabs
    ang = np.radians(np.arange(8) * 45 + 22.5)
    poly = np.stack([0.062 * np.cos(ang), 0.062 * np.sin(ang)], 1)
    for nm, z0 in (("bottom_plate", z_bot), ("top_plate", z_top)):
        V, F = kit.extrude(poly, z0, z0 + t_pl)
        bodies.append(Body(nm, V, F, "pcb", "frame"))
    dirs = [(-1, 1), (-1, -1), (1, -1), (1, 1)]          # front-right, front-left, rear-left, rear-right
    sense = [1, -1, 1, -1]
    ns = _n(12, res, 9)
    nsb = _n(9, res, 7)
    r_hub = 0.0085
    rs = r_hub + (Rp - r_hub) * (1 - np.cos(np.linspace(0, math.pi / 2, ns))) ** 0.75
    rs = np.unique(np.r_[0.006, rs])
    def chord(r):
        return (0.010 + 0.016 * math.sin(math.pi * min(r / Rp, 1.0) ** 0.8)) * (
            1 - 0.55 * max(r / Rp - 0.9, 0) / 0.1)
    def beta(r):
        return math.degrees(math.atan(P / (2 * math.pi * max(r, 0.012))))
    def tc(r):
        return 0.30 if r < 0.010 else 0.10 + 0.08 * max(0.0, 1 - r / (0.5 * Rp))
    def cam(r):
        return 0.04
    def sweep(r):
        return 0.004 * max(r / Rp - 0.6, 0) / 0.4
    az_off = [0.0, 110.0, 50.0, 150.0]
    for k, ((dx, dy), sn) in enumerate(zip(dirs, sense, strict=False)):
        u = np.array([dx, dy], float) / math.sqrt(2)
        phi = math.degrees(math.atan2(u[1], u[0]))
        front = dx < 0
        mat = "arm_red" if front else "arm_white"
        # arm: tapered beam from the plates to the motor mount
        L0, L1 = 0.035, Rarm
        V, F = kit.rounded_box((L1 - L0, 0.024, h_arm), ((L0 + L1) / 2, 0, z_bot + t_pl + h_arm / 2), 0.005, 2)
        xl = V[:, 0]
        f = np.clip((xl - L0) / (L1 - L0), 0, 1)
        zc = z_bot + t_pl + h_arm / 2
        V = V.copy()
        V[:, 2] = zc + (V[:, 2] - zc) * (1 - 0.45 * f)          # arm depth tapers toward the motor
        V[:, 1] *= 1 - 0.25 * f
        V = V @ kit.rot((0, 0, 1), phi).T
        bodies.append(Body(f"arm_{k + 1}", V, F, mat, "frame"))
        cx, cy = Rarm * u
        zt_arm = zc + h_arm / 2 * 0.55                          # arm top at the motor end
        # motor mount pad and motor (2212 outrunner)
        bodies.append(Body(f"motor_mount_{k + 1}", *kit.cylinder((cx, cy, zt_arm - 0.012), (cx, cy, zt_arm + 0.003),
                                                                0.019, nt), mat, "frame"))
        zm = zt_arm + 0.003
        xm = [zm, zm + 0.002, zm + 0.008, zm + 0.020, zm + 0.025, zm + 0.026]
        rm = [0.0135, 0.0140, 0.0140, 0.0140, 0.0120, 0.0]
        Vm, Fm = kit.revolve(xm, rm, nt, "z", center=(cx, cy, 0))
        cm = Vm[Fm].mean(1)
        mm = np.where((cm[:, 2] > zm + 0.009) & (cm[:, 2] < zm + 0.019), "anodized_black", "aluminium")
        bodies.append(Body(f"motor_{k + 1}", Vm, Fm, mm.astype(object), "propulsion"))
        zp = zm + 0.030
        bodies.append(Body(f"prop_hub_{k + 1}", *kit.cylinder((cx, cy, zm + 0.025), (cx, cy, zp + 0.004), 0.0085, nt),
                           "prop_cw" if sn > 0 else "prop_ccw", "propulsion"))
        bodies.append(Body(f"prop_nut_{k + 1}", *kit.revolve([zp + 0.004, zp + 0.008, zp + 0.013], [0.0055, 0.0045, 0.0],
                                                             _n(12, res, 10), "z", center=(cx, cy, 0)),
                           "aluminium", "propulsion", aero=False))
        for b in range(2):
            Vb, Fb, _, _ = _blade(rs, chord, beta, tc, cam, sweep, nsb, ax=(0, 0, 1.0), sp=(1.0, 0, 0), sense=sn,
                                  round_root=0.0)
            az = p["rotor_azimuth_deg"] + az_off[k] + 180.0 * b
            Vb = Vb @ kit.rot((0, 0, 1), az).T + [cx, cy, zp]
            bodies.append(Body(f"prop_{k + 1}_blade_{b + 1}", Vb, Fb, "prop_cw" if sn > 0 else "prop_ccw",
                               "propulsion"))
        # landing leg below the arm root, foam foot
        r_a, r_b = 0.075, 0.125
        a_ = np.linspace(0, 1, 7)
        path = np.stack([(r_a + (r_b - r_a) * a_ ** 1.5) * u[0], (r_a + (r_b - r_a) * a_ ** 1.5) * u[1],
                         z_bot + 0.004 - (z_bot - 0.012) * a_], 1)
        bodies.append(Body(f"leg_{k + 1}", *kit.tube(path, 0.0065, _n(10, res, 8)), "plastic", "landing_gear",
                           aero=False))
        bodies.append(Body(f"foot_{k + 1}", *kit.ellipsoid((r_b * u[0], r_b * u[1], 0.011), (0.011, 0.011, 0.011),
                                                            _n(12, res, 10), _n(7, res, 6)), "foam", "landing_gear",
                           aero=False))
        # ESC under the arm
        Ve, Fe = kit.rounded_box((0.045, 0.018, 0.008), (0.085, 0, z_bot + t_pl - 0.004), 0.002, 1)
        bodies.append(Body(f"esc_{k + 1}", *kit.tf((Ve, Fe), kit.rot((0, 0, 1), phi)), "plastic", "electronics",
                           aero=False))
    # canopy dome on the top plate
    zc0 = z_top + t_pl
    xs_c = np.linspace(0, 1, _n(9, res, 7))
    rr = np.sqrt(np.clip(1 - xs_c ** 2, 0, 1))
    Vc, Fc = kit.revolve(zc0 + 0.038 * xs_c, 0.058 * rr, _n(28, res, 20), "z")
    Vc = Vc * [1.18, 0.95, 1.0]
    bodies.append(Body("canopy", Vc, Fc, "canopy", "frame"))
    # GPS mast and puck
    bodies.append(Body("gps_mast", *kit.cylinder((0.045, 0, zc0 + 0.01), (0.045, 0, zc0 + 0.085), 0.003, 8),
                       "plastic", "electronics", aero=False))
    bodies.append(Body("gps_puck", *kit.rounded_box((0.05, 0.05, 0.012), (0.045, 0, zc0 + 0.091), 0.006, 2),
                       "anodized_black", "electronics", aero=False))
    # battery strapped under the bottom plate
    zb0 = z_bot - 0.027
    Vba, Fba = kit.rounded_box((0.135, 0.043, 0.026), (0.030, 0, zb0 + 0.013), 0.004, 2)
    cba = Vba[Fba].mean(1)
    mba = np.where(np.abs(cba[:, 0] + 0.005) < 0.009, "strap", "battery").astype(object)
    mba[np.abs(cba[:, 0] - 0.065) < 0.009] = "strap"
    bodies.append(Body("battery", Vba, Fba, mba, "electronics"))
    # 2-axis gimbal + camera under the nose
    gx = -0.078
    bodies.append(Body("gimbal_plate", *kit.rounded_box((0.040, 0.060, 0.004), (gx, 0, z_bot - 0.004), 0.0015, 1),
                       "anodized_black", "payload", aero=False))
    for sd in (1, -1):
        bodies.append(Body(f"gimbal_damper_{'r' if sd > 0 else 'l'}", *kit.cylinder(
            (gx, sd * 0.022, z_bot - 0.006), (gx, sd * 0.022, z_bot - 0.016), 0.006, 10), "foam", "payload",
            aero=False))
        bodies.append(Body(f"gimbal_arm_{'r' if sd > 0 else 'l'}", *kit.rounded_box(
            (0.012, 0.005, 0.045), (gx, sd * 0.030, z_bot - 0.040), 0.002, 1), "aluminium", "payload", aero=False))
    bodies.append(Body("gimbal_top", *kit.rounded_box((0.016, 0.065, 0.008), (gx, 0, z_bot - 0.020), 0.002, 1),
                       "aluminium", "payload", aero=False))
    bodies.append(Body("camera", *kit.rounded_box((0.040, 0.050, 0.034), (gx, 0, z_bot - 0.056), 0.005, 2),
                       "anodized_black", "payload", aero=False))
    bodies.append(Body("camera_lens", *kit.cylinder((gx - 0.018, 0, z_bot - 0.056), (gx - 0.032, 0, z_bot - 0.056),
                                                    0.012, nt, r1=0.010), "lens", "payload", aero=False))
    return bodies


register(GeometrySpec(
    name="quadrotor_drone", title="Quadcopter drone (F450 frame class)", category="aircraft",
    description=("X-configuration quadcopter on its landing legs: 450 mm diagonal frame with PCB centre plates, "
                 "red front / white rear arms, 2212-class outrunner motors, four counter-rotating two-blade 10 x 4.5 "
                 "in propellers (constant geometric pitch, cambered sections, swept tips), canopy, GPS mast, "
                 "battery and a two-axis camera gimbal."),
    builder=_quad,
    params=[
        Param("arm_diagonal", 0.45, "m", (0.35, 0.60), "motor-to-motor diagonal (wheelbase)"),
        Param("prop_diameter", 0.254, "m", (0.20, 0.30), "propeller diameter (10 in)"),
        Param("prop_pitch", 0.1143, "m", (0.06, 0.16), "propeller geometric pitch (4.5 in): blade angle "
                                                       "atan(P / 2 pi r)"),
        Param("rotor_azimuth_deg", 0.0, "deg", (0.0, 180.0), "azimuth added to every propeller"),
    ],
    dimensions={"length_x": (0.559, 0.03), "width_y": (0.498, 0.03), "height_z": (0.270, 0.03)},
    frame=("right-handed, metres: x aft (the nose is between the two red arms, at -x), y to starboard, z up; origin "
           "on the ground below the frame centre (feet on z = 0)"),
    sources=[Source("450 mm diagonal wheelbase, 10 x 4.5 in propellers, 2212-class motors", F450_FACTS),
             Source("constant geometric pitch: blade angle atan(P / (2 pi r))", "standard propeller nomenclature "
                    "(e.g. McCormick, B.W. (1995). Aerodynamics, Aeronautics and Flight Mechanics, 2nd ed., Wiley)"),
             Source("plate size, arm section, motor 28 x 26 mm, blade chord / thickness distribution, landing legs, "
                    "3S battery 135 x 43 x 26 mm, gimbal and camera sizes", "estimates from hobby-grade parts",
                    estimate=True)],
    physics=["multirotor aerodynamics (rotor-rotor and rotor-frame interaction, hover / forward flight)",
             "propeller noise", "ground effect and downwash", "flight-dynamics / control surrogates"],
    materials=DRONE_MATERIALS, tags=("drone", "multirotor", "UAV", "propeller"),
    camera={"azimuth": 215.0, "elevation": 25.0},
))
