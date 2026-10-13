"""Industrial and biomedical reference geometries: an electronics pin-fin heat sink, a closed centrifugal pump
impeller, the CAARC standard tall building in an urban block, and a human thoracic aorta lumen with the three
supra-aortic branches."""
from __future__ import annotations

import math

import numpy as np

from .. import kit, meshops
from ..core import Body, GeometrySpec, Param, Source, register


# ============================================================================================ shared helpers
def _n(base: float, res: float, lo: int) -> int:
    return max(lo, int(round(base * res)))


def _revolve_profile(prof, phis, inner2d=None, bore=()):
    """Closed surface of revolution about z of a closed (r, z) polygon ``prof`` sampled at the angles ``phis``.
    Profile vertices listed in ``bore`` are placed on the 2D contour ``inner2d`` (one point per angle) instead of
    the circle: a bore with a keyway."""
    P = np.asarray(prof, float)
    m, n = len(P), len(phis)
    c, s = np.cos(phis), np.sin(phis)
    V = np.empty((m, n, 3))
    for i, (r, z) in enumerate(P):
        if i in bore and inner2d is not None:
            V[i, :, 0], V[i, :, 1] = inner2d[:, 0], inner2d[:, 1]
        else:
            V[i, :, 0], V[i, :, 1] = r * c, r * s
        V[i, :, 2] = z
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    a = i * n + j
    b = ((i + 1) % m) * n + j
    cc = ((i + 1) % m) * n + (j + 1) % n
    d = i * n + (j + 1) % n
    F = np.concatenate([np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([a, cc, d], -1).reshape(-1, 3)])
    return meshops.orient_outward(V.reshape(-1, 3), F)


def _banded_prism(ring2d, z_levels, row_mat, col_mat, roof_mat):
    """Prism over a convex counter-clockwise footprint ``ring2d`` with horizontal grid lines at ``z_levels``.
    Side face (row i, segment j) gets ``row_mat(i, j)`` / ``col_mat``-driven materials; roof and floor get
    ``roof_mat``. Returns (V, F, materials)."""
    R = np.asarray(ring2d, float)
    n = len(R)
    Z = np.asarray(z_levels, float)
    nr = len(Z)
    V = np.vstack([np.c_[R, np.full(n, z)] for z in Z] + [np.r_[R.mean(0), Z[0]][None],
                                                          np.r_[R.mean(0), Z[-1]][None]])
    F, M = [], []
    for i in range(nr - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
            mm = col_mat(j) or row_mat(i)
            M += [mm, mm]
    cb, ct = nr * n, nr * n + 1
    for j in range(n):
        F.append([cb, (j + 1) % n, j])
        F.append([ct, (nr - 1) * n + j, (nr - 1) * n + (j + 1) % n])
        M += [roof_mat, roof_mat]
    F = np.asarray(F, np.int64)
    if meshops.volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F, np.asarray(M, dtype=object)


# ============================================================================================ 1. pin-fin heat sink
JONSSON = ("Jonsson, H., Moshfegh, B. (2001). Modeling of the thermal and hydraulic performance of plate fin, strip "
           "fin, and pin fin heat sinks - influence of flow bypass. IEEE Trans. Compon. Packag. Technol. 24(2), "
           "142-149. https://doi.org/10.1109/6144.926376")
SPARROW = ("Sparrow, E.M., Ramsey, J.W., Altemani, C.A.C. (1980). Experiments on in-line pin fin arrays and "
           "performance comparisons with staggered arrays. J. Heat Transfer 102(1), 44-50. "
           "https://doi.org/10.1115/1.3244247")


def _pin_fin_build(p, res):
    L, W, tb = p["base_length"], p["base_width"], p["base_thickness"]
    d, h, pitch = p["pin_diameter"], p["pin_height"], p["pitch"]
    shape = p["pin_shape"]
    bodies = []
    # base plate with small edge radius, bottom face at z = 0
    base = kit.rounded_box((L, W, tb), (0, 0, tb / 2), radius=min(0.0008, 0.3 * tb), n_side=2)
    bodies.append(Body("base_plate", *base, "heatsink_al", "heat_sink"))
    # pin array (pins root 0.4 mm inside the base: separate closed bodies penetrating the plate)
    nx = max(1, int(math.floor(L / pitch)))
    ny = max(1, int(math.floor(W / pitch)))
    npts = _n(16, res, 12)
    if shape == "circular":
        hw, hh, e = d / 2, d / 2, 2.0
    elif shape == "square":
        hw, hh, e = d / 2, d / 2, 8.0
    else:
        hw, hh, e = min(0.5 * d * p["pin_aspect"], 0.45 * pitch), d / 2, 2.0
    f = min(0.25 * d, 0.0006)                              # top edge round
    z0, z1 = tb - 0.0004, tb + h
    ring = kit.superellipse(hw, hh, npts, e)
    prof = [(z0, 1.0), (z1 - f, 1.0), (z1 - 0.3 * f, 1 - 0.55 * f / max(hh, 1e-9)),
            (z1, 1 - 1.0 * f / max(hh, 1e-9))]
    if res > 1.5:
        prof.insert(2, (z1 - 0.7 * f, 1 - 0.15 * f / hh))
        nz = int(round(4 * res))
        prof[1:1] = [(z, 1.0) for z in np.linspace(z0, z1 - f, nz + 2)[1:-1]]
    pins = []
    for iy in range(ny):
        stag = p["layout"] == "staggered" and iy % 2 == 1
        cols = nx - 1 if stag else nx
        y = (iy - (ny - 1) / 2) * pitch
        for ix in range(cols):
            x = (ix - (cols - 1) / 2) * pitch
            rings = [np.c_[ring * sc + (x, y), np.full(npts, z)] for z, sc in prof]
            pins.append(kit.loft(rings))
    bodies.append(Body("pin_fins", *kit.merge(pins), "heatsink_al", "heat_sink"))
    if p["heat_source"]:
        a, t = p["source_size"], p["source_thickness"]
        pad = kit.box((a, a, t), (0, 0, -t / 2))
        bodies.append(Body("heat_source", *pad, "heater_copper", "heat_source"))
    return bodies


register(GeometrySpec(
    name="pin_fin_heatsink", title="Pin-fin heat sink", category="thermal",
    description=("Electronics pin-fin heat sink: a 60 x 60 x 5 mm aluminium base plate carrying an in-line (or "
                 "staggered) array of circular, square or elliptic pin fins set by diameter, height and pitch, with a "
                 "copper heat-source pad under the base. Flow along x for forced-convection studies."),
    builder=_pin_fin_build,
    params=[
        Param("base_length", 0.060, "m", (0.02, 0.15), "base plate length along x (flow direction)"),
        Param("base_width", 0.060, "m", (0.02, 0.15), "base plate width along y"),
        Param("base_thickness", 0.005, "m", (0.002, 0.012), "base plate thickness"),
        Param("pin_diameter", 0.003, "m", (0.001, 0.008), "pin diameter (square: side; elliptic: minor axis)"),
        Param("pin_height", 0.020, "m", (0.005, 0.050), "pin height above the base"),
        Param("pitch", 0.006, "m", (0.003, 0.015), "pin centre-to-centre pitch in x and y (rows / columns follow)"),
        Param("pin_shape", "circular", "", None, "pin cross-section", choices=("circular", "square", "elliptic")),
        Param("pin_aspect", 1.6, "", (1.0, 3.0), "elliptic pins: major (x) / minor axis ratio"),
        Param("layout", "inline", "", None, "pin arrangement", choices=("inline", "staggered")),
        Param("heat_source", True, "", None, "add the copper heat-source pad under the base"),
        Param("source_size", 0.025, "m", (0.005, 0.10), "heat-source pad side"),
        Param("source_thickness", 0.0015, "m", (0.0005, 0.005), "heat-source pad thickness"),
    ],
    dimensions={"length_x": (0.060, 0.01), "width_y": (0.060, 0.01), "height_z": (0.0265, 0.01)},
    frame=("right-handed, metres: x along the base length (cooling-air flow direction), y across, z up along the "
           "pins; origin at the centre of the base plate's underside (the heat-source pad lies below, z < 0)"),
    sources=[Source("pin-fin heat-sink family and test configuration (square base with in-line pin arrays, "
                    "bypass flow, pin diameter / height / pitch as the design variables)", JONSSON),
             Source("in-line versus staggered pin-fin array arrangement", SPARROW),
             Source("base 60 x 60 x 5 mm, pins d = 3 mm, h = 20 mm, pitch 6 mm (10 x 10), 25 mm heat-source pad: "
                    "representative of the tested heat sinks, not the exact published specimens",
                    "engineering estimate", estimate=True)],
    physics=["conjugate heat transfer (pins + base, heat source)", "forced convection and pressure drop across "
             "pin arrays (inline vs staggered)", "flow bypass and fin-efficiency studies",
             "thermal design optimisation (pin diameter / pitch / height)"],
    materials={"heatsink_al": {"color": [0.80, 0.81, 0.83], "metallic": 1.0, "roughness": 0.32},
               "heater_copper": {"color": [0.85, 0.50, 0.32], "metallic": 1.0, "roughness": 0.3}},
    tags=("electronics cooling", "pin fin", "heat sink"),
    camera={"azimuth": 225.0, "elevation": 32.0},
))


# ============================================================================================ 2. centrifugal impeller
UBALDI = ("Ubaldi, M., Zunino, P., Barigozzi, G., Cattanei, A. (1996). An experimental investigation of stator "
          "induced unsteadiness on centrifugal impeller outflow. J. Turbomach. 118(1), 41-54. "
          "https://doi.org/10.1115/1.2836604 (ERCOFTAC centrifugal pump test case)")
GULICH = ("Gulich, J.F. (2014). Centrifugal Pumps, 3rd ed. Springer. https://doi.org/10.1007/978-3-642-40114-5 "
          "(impeller design: blade angles, log-spiral blades, shroud / hub contours)")
DIN6885 = "DIN 6885-1 (1968). Parallel keys, keyways, deep pattern (shaft 44-50 mm: key 14 x 9, hub depth 3.8 mm)"


def _imp_contours(p):
    """Meridional contours (r, z): hub z_h(r) and shroud z_s(r) of the flow channel, and their constants."""
    R2, b2 = p["outlet_diameter"] / 2, p["outlet_width"]
    Re = p["eye_diameter"] / 2
    A = min(0.05 * R2 / 0.21, 0.8 * (R2 - Re))
    Bz = 0.06 * R2 / 0.21
    r_h = 0.045 * R2 / 0.21
    r_hb = Re * 0.95
    z_hn = 0.05 * R2 / 0.21
    # the channel widens slightly towards the inlet (conical hub and shroud faces)
    return dict(R2=R2, b2=b2, Re=Re, A=A, Bz=Bz, r_h=r_h, r_hb=r_hb, z_hn=z_hn, dzh=0.0015 * R2 / 0.21,
                dzs=0.15 * b2)


def _z_shroud(r, c):
    """Shroud-side channel contour z_s(r): conical from (R2, b2) to (Re + A, b2 + dzs), then a quarter ellipse
    turning axial at the eye radius."""
    r0 = c["Re"] + c["A"]
    if r >= r0:
        return c["b2"] + c["dzs"] * (c["R2"] - r) / (c["R2"] - r0)
    t = np.clip((r - c["Re"]) / c["A"], 0, 1)
    return c["b2"] + c["dzs"] + c["Bz"] * (1 - np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1)))


def _z_hub(r, c):
    return c["dzh"] * (c["R2"] - r) / (c["R2"] - c["r_hb"]) if r >= c["r_hb"] else c["dzh"]


def _impeller_build(p, res):
    c = _imp_contours(p)
    R2, b2, Re, A, Bz = c["R2"], c["b2"], c["Re"], c["A"], c["Bz"]
    r_h, r_hb, z_hn = c["r_h"], c["r_hb"], c["z_hn"]
    sc = R2 / 0.21
    rb = p["bore_diameter"] / 2
    t_back, t_front = 0.008 * sc, 0.006 * sc
    nphi = _n(96, res, 48)
    ne = _n(8, res, 6)
    bodies = []
    # ---------------------------------------------------------------- hub + back shroud (one revolved solid)
    r_boss = rb + 0.030 * sc
    z_back = -0.040 * sc
    tq = np.linspace(0, math.pi / 2, ne + 2)[1:-1]
    dzh, dzs = c["dzh"], c["dzs"]
    hub_curve = [(r_h + (r_hb - r_h) * (1 - math.cos(t)), dzh + (z_hn - dzh) * (1 - math.sin(t)))
                 for t in tq][::-1]
    prof = ([(rb, z_back), (r_boss - 0.003 * sc, z_back), (r_boss, z_back + 0.003 * sc),
             (r_boss, -t_back - 0.010 * sc), (r_boss + 0.014 * sc, -1.3 * t_back), (R2, -t_back), (R2, 0.0),
             (r_hb, dzh)] + hub_curve +
            [(r_h, z_hn), (r_h - 0.004 * sc, z_hn + 0.004 * sc), (rb + 0.002 * sc, z_hn + 0.004 * sc),
             (rb, z_hn + 0.002 * sc)])
    bore_idx = (0, len(prof) - 1)
    if p["keyway"]:
        kw, kd = p["keyway_width"], p["keyway_depth"]
        phk = math.asin(min(kw / 2 / rb, 0.9))
        phk2 = math.atan2(kw / 2, rb + kd)
        uni = np.linspace(phk, 2 * math.pi - phk, nphi)
        phis = np.r_[phk2, phk, uni[1:-1], 2 * math.pi - phk, 2 * math.pi - phk2]
        inner = np.c_[rb * np.cos(phis), rb * np.sin(phis)]
        inner[0] = (rb + kd, kw / 2)
        inner[1] = (rb * math.cos(phk), kw / 2)
        inner[-2] = (rb * math.cos(phk), -kw / 2)
        inner[-1] = (rb + kd, -kw / 2)
    else:
        phis = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
        inner = None
    Vh, Fh = _revolve_profile(prof, phis, inner, bore_idx)
    bodies.append(Body("hub_back_shroud", Vh, Fh, "cast_bronze", "impeller"))
    # ---------------------------------------------------------------- front shroud with the eye
    L_neck = 0.020 * sc
    tq = np.linspace(0, math.pi / 2, _n(14, res, 10))
    ztop = b2 + dzs + Bz
    ell = [(Re + A * (1 - math.cos(t)), b2 + dzs + Bz * (1 - math.sin(t))) for t in tq][::-1]  # -> (Re, ztop)
    rmid = Re + A + 0.5 * (R2 - Re - A)
    inner_c = np.array([(R2, b2), (rmid, _z_shroud(rmid, c))] + ell +
                       [(Re, ztop + 0.5 * L_neck), (Re, ztop + L_neck)])
    tg = np.gradient(inner_c, axis=0)
    nrm = np.c_[tg[:, 1], -tg[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    thick = t_front * (0.8 + 0.2 * np.clip((R2 - inner_c[:, 0]) / (R2 - Re), 0, 1))   # tapered shroud
    thick[-2:] = 1.6 * t_front                                    # wear-ring neck
    outer_c = inner_c + thick[:, None] * nrm
    outer_c[0] = (R2, b2 + 0.8 * t_front)
    outer_c[-1] = (Re + 1.6 * t_front, ztop + L_neck)
    outer_c[-2, 0] = Re + 1.6 * t_front
    prof_s = np.vstack([inner_c, outer_c[::-1]])
    phis_s = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
    if p["front_shroud"]:
        bodies.append(Body("front_shroud", *_revolve_profile(prof_s, phis_s), "cast_bronze", "impeller"))
    # ---------------------------------------------------------------- blades (log-spiral camber, 2D blades)
    Z = p["blade_count"]
    b1, b2a = math.radians(p["beta1_deg"]), math.radians(p["beta2_deg"])
    tb = p["blade_thickness"]
    r1h, r1s = Re * 1.10, Re * 1.02                       # leading edge: further into the eye at the shroud
    rr = np.linspace(min(r1h, r1s), R2, 400)
    beta = b1 + (b2a - b1) * (rr - rr[0]) / (R2 - rr[0])
    dth = 1.0 / (rr * np.tan(beta))
    th = -np.r_[0, np.cumsum(0.5 * (dth[1:] + dth[:-1]) * np.diff(rr))]   # backward curved (rotation +theta)
    nc = _n(26, res, 18)
    ns = _n(3, res, 3)
    xi = 1 - np.cos(0.5 * math.pi * np.linspace(0, 1, nc))      # clustered at the leading edge
    pen_h, pen_s = 0.5 * t_back, 0.45 * t_front
    span = np.linspace(0, 1, ns + 1)
    for k in range(Z):
        rot0 = 2 * math.pi * k / Z
        rings = []
        for x in xi:
            half = 0.5 * tb * (0.30 + 0.70 * min(1.0, math.sqrt(x / 0.12)))
            side_p, side_s = [], []
            for s in span:
                r = r1h + (r1s - r1h) * s
                r = r + x * (R2 - r)
                t = np.interp(r, rr, th) + rot0
                dt = -np.interp(r, rr, dth)
                tx, ty = math.cos(t) - r * math.sin(t) * dt, math.sin(t) + r * math.cos(t) * dt
                ln = math.hypot(tx, ty)
                nx_, ny_ = -ty / ln, tx / ln
                zl = _z_hub(r, c) - pen_h
                zh = float(_z_shroud(r, c)) + pen_s
                z = zl + s * (zh - zl)
                cx, cy = r * math.cos(t), r * math.sin(t)
                for sgn, lst in ((1, side_p), (-1, side_s)):
                    px, py = cx + sgn * half * nx_, cy + sgn * half * ny_
                    rq = math.hypot(px, py)
                    if rq > R2:                       # trailing edge turned to the outlet diameter
                        px, py = px * R2 / rq, py * R2 / rq
                    lst.append((px, py, z))
            rings.append(np.array(side_p + side_s[::-1]))
        bodies.append(Body(f"blade_{k + 1}", *kit.loft(np.asarray(rings)), "cast_bronze_blade", "impeller"))
    return bodies


register(GeometrySpec(
    name="centrifugal_impeller", title="Closed centrifugal pump impeller", category="turbomachinery",
    description=("Closed (shrouded) centrifugal pump impeller of the ERCOFTAC test-case size: D2 = 420 mm, 7 "
                 "backward-curved log-spiral blades between inlet angle beta1 and outlet angle beta2, hub with keyed "
                 "bore, back shroud and front shroud with suction eye and wear-ring neck. Blades are separate bodies "
                 "penetrating both shrouds."),
    builder=_impeller_build,
    params=[
        Param("outlet_diameter", 0.42, "m", (0.15, 1.0), "impeller outlet diameter D2 (scales the meridional shape)"),
        Param("outlet_width", 0.040, "m", (0.01, 0.12), "blade height at the outlet b2"),
        Param("eye_diameter", 0.20, "m", (0.06, 0.5), "suction-eye diameter (front shroud inner diameter)"),
        Param("blade_count", 7, "", (3, 12), "number of blades"),
        Param("beta1_deg", 20.0, "deg", (10.0, 40.0), "blade inlet angle from the tangential direction"),
        Param("beta2_deg", 22.5, "deg", (12.0, 45.0), "blade outlet angle from the tangential direction"),
        Param("blade_thickness", 0.006, "m", (0.002, 0.02), "blade thickness (rounded, thinner leading edge)"),
        Param("bore_diameter", 0.050, "m", (0.015, 0.12), "shaft bore diameter"),
        Param("keyway", True, "", None, "cut a parallel-key keyway into the bore"),
        Param("keyway_width", 0.014, "m", (0.003, 0.03), "keyway width"),
        Param("keyway_depth", 0.0038, "m", (0.001, 0.01), "keyway depth into the hub"),
        Param("front_shroud", True, "", None, "closed impeller (False: semi-open, blades exposed)"),
    ],
    dimensions={"length_x": (0.42, 0.01), "width_y": (0.42, 0.01), "height_z": (0.166, 0.01)},
    frame=("right-handed, metres: z the rotation axis pointing into the suction eye (upstream), origin on the axis "
           "in the hub-side plane of the outlet (back-shroud inner face); the impeller rotates in +theta about +z, "
           "blades are backward-curved"),
    sources=[Source("outlet diameter D2 = 420 mm and 7 blades of the ERCOFTAC centrifugal-pump test impeller "
                    "(recalled from the published description; check against the paper before quantitative use)",
                    UBALDI, estimate=True),
             Source("b2 = 40 mm, eye 200 mm, beta1 = 20 deg, beta2 = 22.5 deg, blade thickness 6 mm, shroud / hub "
                    "contours, wrap angle: generic design within the usual ranges", GULICH, estimate=True),
             Source("keyway 14 mm wide, 3.8 mm deep in the hub for a 50 mm shaft", DIN6885)],
    physics=["pump performance (head, efficiency) with frozen-rotor / sliding-mesh CFD",
             "impeller-diffuser interaction and unsteady blade loading", "cavitation at the blade leading edge",
             "rotordynamic and structural (FEA) analysis of the impeller"],
    materials={"cast_bronze": {"color": [0.72, 0.53, 0.33], "metallic": 1.0, "roughness": 0.38},
               "cast_bronze_blade": {"color": [0.78, 0.60, 0.38], "metallic": 1.0, "roughness": 0.32}},
    tags=("pump", "impeller", "turbomachinery", "ERCOFTAC"),
    camera={"azimuth": 215.0, "elevation": 38.0},
))


# ============================================================================================ 3. CAARC building
MELBOURNE = ("Melbourne, W.H. (1980). Comparison of measurements on the CAARC standard tall building model in "
             "simulated model wind flows. J. Wind Eng. Ind. Aerodyn. 6(1-2), 73-88. "
             "https://doi.org/10.1016/0167-6105(80)90023-9")

_FACADES = ["facade_concrete", "facade_brick", "facade_beige", "facade_dark", "facade_white"]


def _caarc_build(p, res):
    H, B, D = p["height"], p["width"], p["depth"]
    bodies = []
    # ---------------------------------------------------------------- the tower: 50 storeys of 3.66 m
    nfl = max(1, int(round(H / 3.6576)))
    fh = H / nfl
    zl = [0.0]
    for k in range(nfl):
        zl += [k * fh + 0.38 * fh, (k + 1) * fh]
    nbx = _n(4, res, 4)                   # bays across the depth (x) face
    nby = _n(6, res, 6)                   # bays across the wide (y) face
    wm = 0.9                              # mullion / column width [m]

    def side(p0, p1, nb):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / L
        pts = []
        for b in range(nb):
            a = b * L / nb
            pts += [p0 + a * u, p0 + (a + wm) * u]
        return pts
    hx, hy = D / 2, B / 2
    ring = (side((-hx, -hy), (hx, -hy), nbx) + side((hx, -hy), (hx, hy), nby) +
            side((hx, hy), (-hx, hy), nbx) + side((-hx, hy), (-hx, -hy), nby))

    def row_t(i):
        if i == 0 or i >= len(zl) - 3:
            return "facade_panel"
        return "facade_panel" if i % 2 == 0 else "glass_curtain"
    V, F, M = _banded_prism(ring, zl, row_t, lambda j: "mullion" if j % 2 == 0 else None, "roof")
    bodies.append(Body("caarc_tower", V, F, M, "tower"))
    # ---------------------------------------------------------------- urban surroundings (deterministic)
    Rg = p["ground_radius"]
    if p["surroundings"]:
        rng = np.random.default_rng(p["seed"])
        pitch = 55.0
        nlot = int(Rg // pitch) + 1
        clear = 0.5 * math.hypot(B, D) + 35.0
        city = []
        for ix in range(-nlot, nlot + 1):
            for iy in range(-nlot, nlot + 1):
                u = rng.random(6)
                cx, cy = ix * pitch, iy * pitch
                if ix == 0 and iy == 0:
                    continue
                if u[0] > p["density"]:
                    continue
                lx, ly = 22.0 + 16.0 * u[1], 22.0 + 16.0 * u[2]
                hgt = 12.0 + 33.0 * u[3] ** 1.4
                cx += (u[4] - 0.5) * (pitch - lx - 12.0)
                cy += (u[5] - 0.5) * (pitch - ly - 12.0)
                corner = math.hypot(abs(cx) + lx / 2, abs(cy) + ly / 2)
                if corner > Rg - 8.0 or math.hypot(cx, cy) - 0.5 * math.hypot(lx, ly) < clear - 20.0:
                    continue
                if abs(cx) - lx / 2 < hx + 25.0 and abs(cy) - ly / 2 < hy + 25.0:
                    continue
                city.append((cx, cy, lx, ly, hgt, _FACADES[int(rng.integers(len(_FACADES)))]))
        for k, (cx, cy, lx, ly, hgt, fac) in enumerate(city):
            fl = max(2, int(round(hgt / 3.5)))
            fhh = hgt / fl
            zz = [0.0]
            for q in range(fl):
                zz += [q * fhh + 0.45 * fhh, q * fhh + 0.85 * fhh, (q + 1) * fhh]
            zz[-2] = hgt - 0.6
            ringc = [(cx - lx / 2, cy - ly / 2), (cx + lx / 2, cy - ly / 2), (cx + lx / 2, cy + ly / 2),
                     (cx - lx / 2, cy + ly / 2)]

            def rmat(i, fac=fac, n=len(zz)):  # noqa: B008
                return "glass_window" if (i % 3 == 1 and 0 < i < n - 3) else fac
            Vb, Fb, Mb = _banded_prism(ringc, zz, rmat, lambda j: None, "roof")
            bodies.append(Body(f"building_{k + 1:02d}", Vb, Fb, Mb, "surroundings"))
    # ---------------------------------------------------------------- turntable / ground (top at z = 0)
    ng = _n(96, res, 64)
    th = np.linspace(0, 2 * math.pi, ng, endpoint=False)
    disc = np.c_[Rg * np.cos(th), Rg * np.sin(th)]
    gv, gf = kit.extrude(disc, -1.0, 0.0)
    bodies.append(Body("ground", gv, gf, "asphalt", "ground", aero=False))
    R = kit.rot((0, 0, 1), p["orientation_deg"])
    for b in bodies:
        b.vertices = b.vertices @ R.T
    return bodies


register(GeometrySpec(
    name="caarc_building", title="CAARC standard tall building", category="civil",
    description=("The CAARC standard tall building for wind-loading comparisons: a 30.48 x 45.72 x 182.88 m "
                 "(100 x 150 x 600 ft) prism with a 50-storey curtain-wall facade, in a deterministic urban block "
                 "of lower buildings on a 300 m-radius turntable ground (top at z = 0)."),
    builder=_caarc_build,
    params=[
        Param("height", 182.88, "m", (60.0, 400.0), "tower height (600 ft)"),
        Param("width", 45.72, "m", (10.0, 100.0), "wide-face width along y, normal to the wind at 0 deg (150 ft)"),
        Param("depth", 30.48, "m", (10.0, 100.0), "along-wind depth along x (100 ft)"),
        Param("orientation_deg", 0.0, "deg", (-180.0, 180.0), "wind incidence: turntable rotation about z"),
        Param("surroundings", True, "", None, "add the urban block of lower buildings"),
        Param("density", 0.7, "", (0.0, 1.0), "fraction of city lots that carry a building"),
        Param("seed", 1980, "", (0, 2 ** 31 - 1), "seed of the deterministic city generator"),
        Param("ground_radius", 300.0, "m", (100.0, 1000.0), "turntable / ground disc radius"),
    ],
    dimensions={"length_x": (600.0, 0.01), "width_y": (600.0, 0.01), "height_z": (183.88, 0.01)},
    frame=("right-handed, metres, full scale: x the wind direction at 0 deg incidence (normal to the 45.72 m wide "
           "face), y across, z up; origin at the tower base centre on the ground (z = 0 is the ground top; the "
           "ground disc is 1 m thick)"),
    sources=[Source("100 x 150 x 600 ft (30.48 x 45.72 x 182.88 m) CAARC tall-building model, wind normal to the "
                    "wide face at 0 deg", MELBOURNE),
             Source("50 storeys of 12 ft, curtain-wall bays, spandrel bands (visual facade only)",
                    "engineering estimate", estimate=True),
             Source("surrounding urban block (55 m street grid, 12-45 m buildings, density 0.7), 300 m ground disc",
                    "generic, generated (not a published configuration)", estimate=True)],
    physics=["mean and fluctuating wind pressures on a tall building (CAARC benchmark)",
             "along / across-wind base moments and vortex shedding", "LES of the atmospheric boundary layer with "
             "urban roughness", "pedestrian-level wind comfort around the tower"],
    materials={"glass_curtain": {"color": [0.10, 0.16, 0.22], "metallic": 0.3, "roughness": 0.06, "clearcoat": 1.0},
               "glass_window": {"color": [0.08, 0.11, 0.15], "metallic": 0.2, "roughness": 0.08, "clearcoat": 1.0},
               "facade_panel": {"color": [0.80, 0.80, 0.78], "metallic": 0.1, "roughness": 0.5},
               "mullion": {"color": [0.62, 0.64, 0.66], "metallic": 0.8, "roughness": 0.35},
               "roof": {"color": [0.42, 0.43, 0.44], "metallic": 0.0, "roughness": 0.85},
               "facade_concrete": {"color": [0.66, 0.65, 0.62], "metallic": 0.0, "roughness": 0.85},
               "facade_brick": {"color": [0.58, 0.32, 0.24], "metallic": 0.0, "roughness": 0.85},
               "facade_beige": {"color": [0.82, 0.74, 0.60], "metallic": 0.0, "roughness": 0.8},
               "facade_dark": {"color": [0.30, 0.31, 0.33], "metallic": 0.1, "roughness": 0.6},
               "facade_white": {"color": [0.90, 0.89, 0.86], "metallic": 0.0, "roughness": 0.7},
               "asphalt": {"color": [0.30, 0.31, 0.31], "metallic": 0.0, "roughness": 0.95}},
    tags=("wind engineering", "tall building", "CAARC", "urban"),
    camera={"azimuth": 215.0, "elevation": 22.0},
))


# ============================================================================================ 4. aortic arch
HAGER = ("Hager, A., Kaemmerer, H., Rapp-Bernhardt, U., Bluecher, S., Rapp, K., Bernhardt, T.M., Galanski, M., "
         "Hess, J. (2002). Diameters of the thoracic aorta throughout life as measured with helical computed "
         "tomography. J. Thorac. Cardiovasc. Surg. 123(6), 1060-1066. https://doi.org/10.1067/mtc.2002.122310")
REDHEUIL = ("Redheuil, A., Yu, W.C., Mousseaux, E., et al. (2011). Age-related changes in aortic arch geometry: "
            "relationship with proximal aortic function and left ventricular mass and remodeling. J. Am. Coll. "
            "Cardiol. 58(12), 1262-1270. https://doi.org/10.1016/j.jacc.2011.06.012")


def _frames(P):
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    seed = np.array([1.0, 0, 0]) if abs(T[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    nrm = np.cross(T[0], seed)
    nrm /= np.linalg.norm(nrm)
    N, Bn = [], []
    for t in T:
        nrm = nrm - np.dot(nrm, t) * t
        nrm /= np.linalg.norm(nrm)
        N.append(nrm)
        Bn.append(np.cross(t, nrm))
    return T, np.array(N), np.array(Bn)


def _sweep(P, rfun, n):
    """Closed (flat-capped) tube along ``P`` with radius ``rfun(k, theta)`` at station k."""
    _, N, Bn = _frames(P)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = [P[k] + rfun(k, th)[:, None] * (np.cos(th)[:, None] * N[k] + np.sin(th)[:, None] * Bn[k])
             for k in range(len(P))]
    return kit.loft(np.asarray(rings))


def _resample(C, m):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    su = np.linspace(0, s[-1], m)
    return np.stack([np.interp(su, s, C[:, i]) for i in range(3)], 1), su


_BIF_OFFSET = 1.2         # daughter vessels start this many trunk radii before the trunk end
_FLARE = (0.35, 6.0)       # ostium flare: relative radius increase at the aortic wall, decay length [mm]


def _aorta_build(p, res):
    from scipy.interpolate import CubicSpline
    mm = 1e-3
    Ra = p["arch_radius"] / mm                                    # work in mm, convert at the end
    al = math.radians(p["arch_plane_deg"])
    u = np.array([math.sin(al), math.cos(al), 0.0])               # in-plane horizontal: posterior-left
    w = np.array([math.cos(al), -math.sin(al), 0.0])              # arch-plane normal (patient left-anterior)
    k = Ra / 32.0
    # (a: along u, z, out-of-plane) control points of the centreline, arch scaled by the arch radius
    L_asc = p["ascending_length"] / mm
    cp = np.array([(0, 0, 0), (-2.5, 0.35 * L_asc, 0), (-3.5, 0.7 * L_asc, 0), (-2.0, L_asc, 0),
                   (6 * k, L_asc + 20 * k, 0.5), (22 * k, L_asc + 32 * k, 1.0), (40 * k, L_asc + 33 * k, 1.5),
                   (56 * k, L_asc + 23 * k, 2.0), (64 * k, L_asc + 4 * k, 3.0), (66 * k, L_asc - 30, 5.0),
                   (66 * k, L_asc - 100, 7.0), (65 * k, L_asc - 170, 9.0),
                   (64 * k, L_asc - p["descending_length"] / mm, 10.0)])
    sp = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cp, axis=0), axis=1))]
    cs = CubicSpline(sp, cp, bc_type="natural")
    C2 = cs(np.linspace(0, sp[-1], 600))
    C2, s = _resample(C2, _n(110, res, 80))
    C = C2[:, 0:1] * u + C2[:, 1:2] * np.array([0, 0, 1.0]) + C2[:, 2:3] * w
    # radius along the arc length (mm)
    d_as, d_ar, d_de = p["ascending_diameter"] / mm, p["arch_diameter"] / mm, p["descending_diameter"] / mm
    s_arch = L_asc + 45 * k
    knots = [0, 25, 0.6 * L_asc, L_asc + 10, s_arch, s_arch + 55 * k, s_arch + 100, s[-1]]
    dia = [0.78 * d_as, 0.92 * d_as, d_as, 0.5 * (d_as + d_ar), d_ar, 0.95 * d_de, d_de, 0.93 * d_de]
    rb = 0.5 * np.interp(s, knots, dia)
    sinus = p["sinus_diameter"] / mm
    bump = np.exp(-((s - 12.0) / 8.5) ** 2)
    amp = np.maximum(0.5 * sinus - rb, 0) * 1.25
    nth = _n(48, res, 36)

    def r_main(i, th):
        return rb[i] + amp[i] * bump[i] * (0.2 + 0.8 * (0.5 + 0.5 * np.cos(3 * th)) ** 1.5)
    tubes = [("aorta", _sweep(C, r_main, nth))]
    # supra-aortic branches: take-off on the arch roof, then rising towards the neck / arm
    T, _, _ = _frames(C)
    zc = np.array([0, 0, 1.0])
    specs = [("brachiocephalic_trunk", 0.28, p["bct_diameter"], p["bct_angle_deg"], 30.0),
             ("left_common_carotid", 0.50, p["lcca_diameter"], p["lcca_angle_deg"], 52.0),
             ("left_subclavian", 0.68, p["lsa_diameter"], p["lsa_angle_deg"], 46.0)]
    i_top0 = int(np.searchsorted(s, L_asc + 8 * k))
    i_top1 = int(np.searchsorted(s, s_arch + 60 * k))
    nb = _n(16, res, 12)
    nbt = _n(24, res, 20)
    for name, frac, dmm, ang, Lb in specs:
        i = int(round(i_top0 + frac * (i_top1 - i_top0)))
        P0 = C[i]
        up = zc - np.dot(zc, T[i]) * T[i]
        up /= np.linalg.norm(up)
        a = math.radians(ang)
        dfin = np.array([math.sin(a), 0.12 * math.sin(a), math.cos(a)])   # tilt in the frontal plane (x left)
        dfin /= np.linalg.norm(dfin)
        rr = 0.5 * dmm / mm
        P1 = P0 + up * (rb[i] + 8.0)
        P2 = P1 + dfin * Lb
        t = np.linspace(0, 1, nb)[:, None]
        path = (1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t ** 2 * P2
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        flare = 1.0 + _FLARE[0] * np.exp(-np.maximum(dist - rb[i], 0) / _FLARE[1])     # ostium funnel
        flare = np.minimum(flare, (rb[i] - 1.0) / rr)
        rad = rr * flare
        bif = name == "brachiocephalic_trunk" and p["bct_bifurcation"]
        if bif:                               # domed end, covered by the two daughter vessels
            ph = np.linspace(0, 1.25, 5)[1:]
            path = np.vstack([path, P2 + dfin * (rr * np.sin(ph))[:, None]])
            rad = np.r_[rad, rr * np.cos(ph)]
        tubes.append((name, _sweep(path, lambda q, th, rad=rad: np.full_like(th, rad[q]), nbt)))
        if bif:
            for dname, dd, dang, Ld in (("right_subclavian", p["rsa_diameter"], -68.0, 30.0),
                                        ("right_common_carotid", p["rcca_diameter"], -6.0, 36.0)):
                a2 = math.radians(dang)
                d2 = np.array([math.sin(a2), 0.15 * abs(math.sin(a2)), math.cos(a2)])
                d2 /= np.linalg.norm(d2)
                rd = 0.5 * dd / mm
                Q0 = P2 - dfin * _BIF_OFFSET * rr          # daughters diverge from the trunk axis
                Q2 = P2 + d2 * Ld
                Q1 = 0.5 * (Q0 + Q2) + 0.08 * Ld * dfin
                pth = (1 - t) ** 2 * Q0 + 2 * t * (1 - t) * Q1 + t ** 2 * Q2
                tubes.append((dname, _sweep(pth, lambda q, th, rd=rd: np.full_like(th, rd), nbt)))
    tubes = [(nm, (V * mm, F)) for nm, (V, F) in tubes]
    if p["fused"]:
        V, F = meshops.union([m for _, m in tubes])
        V, F = meshops.weld(V, F)
        return [Body("aortic_lumen", V, F, "lumen", "lumen")]
    return [Body(nm, V, F, "lumen", "lumen") for nm, (V, F) in tubes]


register(GeometrySpec(
    name="aortic_arch", title="Human thoracic aorta (arch)", category="biomedical",
    description=("Adult human thoracic aorta lumen: aortic root with three-lobed sinus of Valsalva, ascending aorta, "
                 "arch and descending thoracic aorta with the brachiocephalic trunk, left common carotid and left "
                 "subclavian arteries, fused (manifold3d union) into one watertight lumen with flat-capped inlet "
                 "and outlets."),
    builder=_aorta_build,
    params=[
        Param("ascending_diameter", 0.030, "m", (0.020, 0.050), "mid-ascending aorta diameter"),
        Param("sinus_diameter", 0.034, "m", (0.024, 0.060), "diameter across the sinuses of Valsalva"),
        Param("arch_diameter", 0.026, "m", (0.018, 0.040), "arch diameter (between the branches)"),
        Param("descending_diameter", 0.023, "m", (0.015, 0.035), "descending thoracic aorta diameter"),
        Param("arch_radius", 0.032, "m", (0.022, 0.045), "radius of curvature of the arch centreline"),
        Param("ascending_length", 0.070, "m", (0.040, 0.090), "root-to-arch length of the ascending aorta"),
        Param("descending_length", 0.200, "m", (0.120, 0.300), "drop of the descending aorta below the arch start"),
        Param("arch_plane_deg", 40.0, "deg", (0.0, 70.0), "rotation of the arch plane from sagittal towards the left"),
        Param("bct_diameter", 0.012, "m", (0.007, 0.018), "brachiocephalic trunk diameter"),
        Param("lcca_diameter", 0.008, "m", (0.005, 0.012), "left common carotid diameter"),
        Param("lsa_diameter", 0.010, "m", (0.006, 0.014), "left subclavian diameter"),
        Param("bct_angle_deg", -28.0, "deg", (-60.0, 20.0), "brachiocephalic tilt from vertical (- = rightwards)"),
        Param("lcca_angle_deg", 6.0, "deg", (-30.0, 40.0), "left common carotid tilt from vertical (+ = leftwards)"),
        Param("lsa_angle_deg", 35.0, "deg", (0.0, 70.0), "left subclavian tilt from vertical (+ = leftwards)"),
        Param("bct_bifurcation", True, "", None, "split the brachiocephalic trunk into the right subclavian "
              "and right common carotid arteries"),
        Param("rsa_diameter", 0.009, "m", (0.005, 0.013), "right subclavian diameter"),
        Param("rcca_diameter", 0.007, "m", (0.004, 0.011), "right common carotid diameter"),
        Param("fused", True, "", None, "one fused lumen (True) or the overlapping closed tubes (False)"),
    ],
    dimensions={"length_x": (0.1134, 0.02), "width_y": (0.0766, 0.02), "height_z": (0.3114, 0.02)},
    frame=("right-handed patient frame, metres: x towards the patient's left, y posterior, z cranial; origin at the "
           "centre of the aortic-root inlet (annulus plane)"),
    sources=[Source("thoracic aortic diameters by segment (sinus ~33-36 mm, ascending ~29-32 mm, arch ~26-28 mm, "
                    "descending ~23-26 mm in adults)", HAGER),
             Source("arch geometry (arch width / height, curvature radius ~30-35 mm)", REDHEUIL),
             Source("chosen defaults within the published ranges, branch diameters 12 / 8 / 10 mm, branch take-off "
                    "angles, lengths, sinus lobing, arch plane rotation and the centreline shape",
                    "anatomical estimate", estimate=True)],
    physics=["hemodynamics: pulsatile blood flow, wall shear stress, secondary flow in the arch",
             "aneurysm / coarctation and stent-graft planning studies", "fluid-structure interaction of the aortic "
             "wall", "particle (embolus) transport into the supra-aortic branches"],
    materials={"lumen": {"color": [0.72, 0.16, 0.14], "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.6}},
    tags=("cardiovascular", "aorta", "hemodynamics"),
    camera={"azimuth": 320.0, "elevation": 10.0},
))
