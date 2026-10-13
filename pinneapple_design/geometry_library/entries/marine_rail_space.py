"""Marine, rail and space entries: the KRISO Container Ship (KCS) at full scale, the analytic Wigley parabolic hull,
a high-speed train (ICE 3 / CRH3 class) on track, and a Falcon 9 Block 5 class two-stage launcher."""
from __future__ import annotations

import math

import numpy as np

from .. import kit, meshops
from ..core import FRAME_VEHICLE, Body, GeometrySpec, Param, Source, register


def _n(base: float, res: float, lo: int = 3) -> int:
    return max(lo, int(round(base * res)))


def _centres(V, F):
    return np.asarray(V)[np.asarray(F)].mean(1)


def _normals(V, F):
    return meshops.face_normals(np.asarray(V, float), np.asarray(F, np.int64))


def _boxes(items, R=None):
    """Merge non-overlapping boxes ``(size, centre)`` into one mesh."""
    return kit.merge([kit.box(s, c, R) for s, c in items])


def _resample(poly, n, lam=1.2):
    """``n + 1`` points along a polyline, spaced by arc length plus ``lam`` x turning angle (corners get points)."""
    P = np.asarray(poly, float)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.r_[True, d > 1e-9]
    P = P[keep]
    if len(P) < 2:
        return np.repeat(P[:1], n + 1, 0)
    seg = np.diff(P, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    ang = np.zeros(len(ds))
    if len(ds) > 1:
        u = seg / ds[:, None]
        c = np.clip(np.sum(u[1:] * u[:-1], 1), -1, 1)
        a = np.arccos(c)
        ang[1:] += 0.5 * a
        ang[:-1] += 0.5 * a
    S = ds.sum()
    A = ang.sum()
    m = ds / S + (lam * ang / A if A > 1e-6 else 0.0)
    cm = np.r_[0, np.cumsum(m)]
    t = np.linspace(0, cm[-1], n + 1)
    return np.stack([np.interp(t, cm, P[:, k]) for k in range(P.shape[1])], 1)


# ============================================================================================ materials
MATERIALS = {
    # ship
    "antifouling_red": {"color": [0.55, 0.09, 0.07], "metallic": 0.0, "roughness": 0.6},
    "hull_navy": {"color": [0.08, 0.13, 0.24], "metallic": 0.0, "roughness": 0.45, "clearcoat": 0.3},
    "deck_green": {"color": [0.25, 0.36, 0.27], "metallic": 0.0, "roughness": 0.75},
    "superstructure_white": {"color": [0.92, 0.92, 0.90], "metallic": 0.0, "roughness": 0.4},
    "funnel_red": {"color": [0.72, 0.13, 0.10], "metallic": 0.0, "roughness": 0.45},
    "funnel_black": {"color": [0.05, 0.05, 0.05], "metallic": 0.0, "roughness": 0.6},
    "lifeboat_orange": {"color": [0.95, 0.42, 0.05], "metallic": 0.0, "roughness": 0.4},
    "bronze": {"color": [0.78, 0.58, 0.32], "metallic": 1.0, "roughness": 0.25},
    "hatch_grey": {"color": [0.42, 0.44, 0.45], "metallic": 0.0, "roughness": 0.7},
    "box_red": {"color": [0.62, 0.14, 0.11], "metallic": 0.2, "roughness": 0.6},
    "box_blue": {"color": [0.10, 0.28, 0.55], "metallic": 0.2, "roughness": 0.6},
    "box_lightblue": {"color": [0.35, 0.62, 0.80], "metallic": 0.2, "roughness": 0.6},
    "box_green": {"color": [0.13, 0.42, 0.25], "metallic": 0.2, "roughness": 0.6},
    "box_orange": {"color": [0.88, 0.45, 0.12], "metallic": 0.2, "roughness": 0.6},
    "box_grey": {"color": [0.55, 0.57, 0.58], "metallic": 0.2, "roughness": 0.6},
    "box_white": {"color": [0.85, 0.85, 0.83], "metallic": 0.2, "roughness": 0.6},
    "box_brown": {"color": [0.42, 0.25, 0.15], "metallic": 0.2, "roughness": 0.6},
    "box_yellow": {"color": [0.90, 0.72, 0.15], "metallic": 0.2, "roughness": 0.6},
    "bridge_glass": {"color": [0.03, 0.05, 0.07], "metallic": 0.1, "roughness": 0.05, "clearcoat": 1.0},
    # model hull
    "model_hull": {"color": [0.95, 0.83, 0.30], "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.6},
    "model_hull_wet": {"color": [0.80, 0.18, 0.12], "metallic": 0.0, "roughness": 0.35, "clearcoat": 0.6},
    "model_deck": {"color": [0.78, 0.78, 0.76], "metallic": 0.0, "roughness": 0.5},
    # train
    "train_white": {"color": [0.95, 0.95, 0.94], "metallic": 0.0, "roughness": 0.25, "clearcoat": 1.0},
    "train_red": {"color": [0.78, 0.06, 0.08], "metallic": 0.0, "roughness": 0.3, "clearcoat": 1.0},
    "train_grey": {"color": [0.30, 0.31, 0.33], "metallic": 0.2, "roughness": 0.5},
    "bogie_dark": {"color": [0.12, 0.12, 0.13], "metallic": 0.4, "roughness": 0.55},
    "rail_steel": {"color": [0.55, 0.55, 0.56], "metallic": 1.0, "roughness": 0.35},
    "sleeper_concrete": {"color": [0.58, 0.57, 0.55], "metallic": 0.0, "roughness": 0.9},
    "insulator": {"color": [0.55, 0.20, 0.12], "metallic": 0.0, "roughness": 0.35},
    # launcher
    "lv_white": {"color": [0.93, 0.93, 0.92], "metallic": 0.0, "roughness": 0.35},
    "lv_black": {"color": [0.04, 0.04, 0.045], "metallic": 0.0, "roughness": 0.55},
    "carbon_leg": {"color": [0.08, 0.08, 0.09], "metallic": 0.2, "roughness": 0.45},
    "titanium": {"color": [0.55, 0.55, 0.57], "metallic": 1.0, "roughness": 0.45},
    "nozzle_dark": {"color": [0.14, 0.13, 0.13], "metallic": 0.8, "roughness": 0.5},
    "heat_shield": {"color": [0.20, 0.19, 0.18], "metallic": 0.0, "roughness": 0.8},
}


# ============================================================================================ Wigley hull
WIGLEY = ("Wigley, W.C.S. (1934). A comparison of experiment and calculated wave-profiles and wave-resistances for "
          "a form having parabolic waterlines. Proc. Royal Society of London A 144(851), 144-159. "
          "https://doi.org/10.1098/rspa.1934.0040")
ITTC_WIGLEY = ("Kajitani, H., Miyata, H., Ikehata, M., Tanaka, H., Adachi, H., Namimatsu, M., Ogiwara, S. (1983). "
               "The summary of the cooperative experiment on Wigley parabolic model in Japan. Proc. 2nd DTNSRDC "
               "Workshop on Ship Wave-Resistance Computations, Bethesda, 5-35 (ITTC Resistance Committee benchmark, "
               "L/B = 10, B/T = 1.6)")


def _wigley(p, res):
    L, B, T, f = p["length"], p["beam"], p["draft"], p["freeboard"]
    nx = 2 * _n(30, res, 12)
    nz = _n(12, res, 6)
    nf = _n(4, res, 2)
    u = np.linspace(0, 1, nx + 1)
    xs = -L / 2 + L * 0.5 * (1 - np.cos(np.pi * u))
    xs[0], xs[-1] = -L / 2, L / 2
    zw = -T * np.cos(0.5 * np.pi * np.linspace(0, 1, nz + 1))
    zw[0], zw[-1] = -T, 0.0
    zs = np.r_[zw, np.linspace(0, f, nf + 1)[1:]]
    nzt = len(zs)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Y = 0.5 * B * (1 - (2 * X / L) ** 2) * np.where(Z < 0, 1 - (Z / T) ** 2, 1.0)
    on_cp = np.zeros_like(Y, bool)
    on_cp[0, :] = on_cp[-1, :] = True
    on_cp[:, 0] = True
    Y[on_cp] = 0.0
    sid = np.arange(Y.size).reshape(Y.shape)
    V = [np.stack([X, Y, Z], -1).reshape(-1, 3)]
    pid = sid.copy()
    nxt = Y.size
    for i in range(nx + 1):
        for j in range(nzt):
            if not on_cp[i, j]:
                pid[i, j] = nxt
                nxt += 1
    Vp = np.stack([X, -Y, Z], -1)[~on_cp]
    # order Vp exactly as the ids were assigned (row-major over the non-centre-plane vertices)
    V.append(Vp.reshape(-1, 3))
    V = np.vstack(V)
    F = []
    for ids, sgn in ((sid, 1), (pid, -1)):
        for i in range(nx):
            for j in range(nzt - 1):
                a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
                ys = [Y[i, j], Y[i + 1, j], Y[i + 1, j + 1], Y[i, j + 1]]
                k = int(np.argmax(ys))
                if k in (0, 2):
                    tri = [(a, c, b), (a, d, c)]
                else:
                    tri = [(b, d, c), (b, a, d)]
                if sgn < 0:
                    tri = [t[::-1] for t in tri]
                F += tri
    top = nzt - 1
    for i in range(nx):
        s0, s1, p0, p1 = sid[i, top], sid[i + 1, top], pid[i, top], pid[i + 1, top]
        F += [(s0, p1, s1), (s0, p0, p1)]
    F = np.array([t for t in F if len(set(t)) == 3], np.int64)
    V, F = meshops.fix_winding(V, F)
    c = _centres(V, F)
    nrm = _normals(V, F)
    mats = np.where(c[:, 2] < 0, "model_hull_wet", "model_hull").astype(object)
    mats[(nrm[:, 2] > 0.99) & (c[:, 2] > f - 1e-9)] = "model_deck"
    return [Body("hull", V, F, mats, "hull")]


register(GeometrySpec(
    name="wigley_hull", title="Wigley parabolic hull", category="marine",
    description=("The analytic Wigley hull y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2) below the waterline, continued "
                 "wall-sided up to a flat deck at the freeboard: a closed solid (Cb = 4/9). The classic ITTC "
                 "benchmark for wave resistance, sinkage / trim and potential-flow / RANS free-surface codes."),
    builder=_wigley,
    params=[
        Param("length", 1.0, "m", (0.5, 10.0), "length between perpendiculars L (stem to stern)"),
        Param("beam", 0.1, "m", (0.03, 2.0), "beam B at the waterline (standard L/B = 10)"),
        Param("draft", 0.0625, "m", (0.02, 1.5), "draft T (standard B/T = 1.6)"),
        Param("freeboard", 0.04, "m", (0.005, 1.0), "height of the flat deck above the waterline"),
    ],
    dimensions={"length_x": (1.0, 0.01), "width_y": (0.1, 0.01), "height_z": (0.1025, 0.01)},
    frame=("right-handed, metres: x aft, y to starboard, z up; origin at midships on the centre plane in the calm "
           "waterline (bow at x = -L/2, stern at x = +L/2, keel at z = -T)"),
    sources=[Source("parabolic waterlines and frames y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)", WIGLEY),
             Source("benchmark proportions L/B = 10, B/T = 1.6", ITTC_WIGLEY),
             Source("default L = 1 m (non-dimensional reference) and freeboard 0.04 L, wall-sided above the waterline",
                    "choice for a normalised closed solid", estimate=True)],
    physics=["wave resistance and wave pattern (Michell thin-ship theory, Rankine panels)",
             "RANS / VOF free-surface CFD validation (resistance, sinkage and trim)",
             "seakeeping: heave / pitch response in head waves", "mesh-convergence benchmark for hull CFD"],
    materials=MATERIALS, tags=("ship hydrodynamics", "benchmark", "analytic hull"),
    camera={"azimuth": 215.0, "elevation": 20.0},
))


# ============================================================================================ Falcon 9 class
SPACEX_F9 = ("SpaceX (2024). Falcon 9 vehicle overview (height 70 m, diameter 3.7 m, nine Merlin engines on the "
             "first stage, one Merlin Vacuum on the second). https://www.spacex.com/vehicles/falcon-9/")
SPACEX_UG = ("SpaceX (2021). Falcon User's Guide, sections 1-3 and 5 (fairing 5.2 m outer diameter, 13.1 m "
             "height; vehicle configuration). https://www.spacex.com/media/falcon-users-guide-2021-09.pdf")

F9_Z = {"base": 1.35, "s1_top": 42.6, "inter_top": 47.3, "s2_top": 56.9, "fairing_cyl": 58.3, "ogive": 63.6,
        "tip": 70.0}


def _grid_fin(res, span=1.5, width=1.3, depth=0.22, n_cells=8):
    """Lattice grid fin as one closed manifold: a frame plus two families of 45-deg slats (boolean union).
    Local frame: u (span) along +z, v (width) along +y, w (thickness) along +x; root edge at u = 0."""
    import manifold3d as m3d

    def man(mesh):
        V, F = meshops.weld(*mesh)
        return m3d.Manifold(m3d.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                                     tri_verts=np.ascontiguousarray(F, np.uint32)))
    tb = 0.06       # frame bar
    ts = 0.022      # slat thickness
    frame = [((depth, width, tb), (0, 0, tb / 2)), ((depth, width, tb), (0, 0, span - tb / 2)),
             ((depth, tb, span), (0, width / 2 - tb / 2, span / 2)), ((depth, tb, span), (0, -width / 2 + tb / 2, span / 2))]
    parts = [man(kit.box(s, c)) for s, c in frame]
    pitch = width / n_cells * math.sqrt(2)
    Lsl = 1.5 * math.hypot(span, width)
    slats = []
    for sgn in (1, -1):
        R = kit.rot((1, 0, 0), 45 * sgn)
        k0 = int(math.ceil(Lsl / pitch))
        for k in range(-k0, k0 + 1):
            # slat centre lines offset perpendicular to their direction
            n_dir = R @ np.array([0, 1.0, 0])
            c = np.array([0, 0, span / 2]) + k * pitch * n_dir
            slats.append(man(kit.tf(kit.box((depth * 0.92, ts, Lsl)), R, c)))
    inner = man(kit.box((depth, width - 1.5 * tb, span - 1.5 * tb), (0, 0, span / 2)))
    lattice = m3d.Manifold.batch_boolean(slats, m3d.OpType.Add) ^ inner
    out = m3d.Manifold.batch_boolean(parts + [lattice], m3d.OpType.Add)
    m = out.to_mesh()
    return np.asarray(m.vert_properties[:, :3], float), np.asarray(m.tri_verts, np.int64)


def _falcon9(p, res):
    z = F9_Z
    r = 3.66 / 2
    nt = _n(48, res, 32)
    bodies = []
    # core stack: first stage (white), interstage (black), second stage (white)
    zs = np.unique(np.r_[np.linspace(z["base"], z["s1_top"], _n(10, res, 6)), np.linspace(z["s1_top"], z["inter_top"], 3),
                         np.linspace(z["inter_top"], z["s2_top"], _n(4, res, 3))])
    rs = np.full(len(zs), r)
    Vc, Fc = kit.revolve(zs, rs, nt, axis="z")
    # revolve about z maps (x, r) -> (r cos, r sin, x)
    c = _centres(Vc, Fc)
    nrm = _normals(Vc, Fc)
    mats = np.where((c[:, 2] > z["s1_top"]) & (c[:, 2] < z["inter_top"]), "lv_black", "lv_white").astype(object)
    mats[nrm[:, 2] < -0.9] = "heat_shield"
    bodies.append(Body("core_stages", Vc, Fc, mats, "vehicle"))
    # payload fairing (5.2 m): boattail cone, cylinder, tangent ogive with a rounded tip
    if p["fairing"]:
        R = 5.2 / 2
        Lo = z["tip"] - z["ogive"]
        rho = (R ** 2 + Lo ** 2) / (2 * R)
        s = np.linspace(0, 1, _n(16, res, 10)) ** 0.8
        xo = s * (Lo - 0.12)                            # distance from the ogive base
        ro = np.sqrt(rho ** 2 - xo ** 2) + R - rho
        rn = ro[-1]
        cap = np.linspace(0, math.pi / 2, _n(5, res, 3))[1:]
        zcap = z["ogive"] + xo[-1] + rn * 0.35 * np.sin(cap) / max(math.sin(cap[-1]), 1e-9)
        rcap = rn * np.cos(cap)
        zf = np.r_[z["s2_top"] - 0.3, z["s2_top"], z["fairing_cyl"], z["ogive"] + xo, zcap[:-1], z["tip"]]
        rf = np.r_[r - 0.02, r - 0.02, R, ro, rcap[:-1], 0.0]
        Vf, Ff = kit.revolve(zf, rf, nt, axis="z")
        bodies.append(Body("payload_fairing", Vf, Ff, "lv_white", "vehicle"))
    else:
        # second-stage forward dome, payload adapter and a generic payload
        Vd, Fd = kit.revolve(np.r_[z["s2_top"] - 0.1, z["s2_top"] + np.linspace(0, 0.8, 5)],
                             np.r_[r, r * np.sqrt(1 - np.linspace(0, 0.8, 5) ** 2 / 0.8 ** 2 * 0.75)], nt, axis="z")
        bodies.append(Body("stage2_dome", Vd, Fd, "lv_white", "vehicle"))
        bodies.append(Body("payload_adapter", *kit.cylinder((0, 0, z["s2_top"] + 0.5), (0, 0, z["s2_top"] + 1.6),
                                                            1.3, nt, 0.65), "aluminium", "payload"))
        sat = kit.rounded_box((2.6, 2.6, 4.0), (0, 0, z["s2_top"] + 3.6), radius=0.08, n_side=2)
        bodies.append(Body("payload", *sat, "mli_gold", "payload"))
    # nine Merlin bells in the octaweb (one centre, eight on a ring)
    re_, rt = 0.43, 0.16
    zb = np.r_[0.0, 0.25, 0.55, 0.85, 1.05, 1.15, 1.25, 1.6]
    rb = np.r_[re_, re_ * 0.86, re_ * 0.66, re_ * 0.46, rt * 1.15, rt, rt * 1.25, 0.24]
    nn = _n(20, res, 14)
    bells = []
    for k in range(9):
        cx, cy = (0.0, 0.0) if k == 0 else (1.2 * math.cos(2 * math.pi * (k - 1) / 8 + math.pi / 8),
                                           1.2 * math.sin(2 * math.pi * (k - 1) / 8 + math.pi / 8))
        bells.append(kit.revolve(zb, rb, nn, axis="z", center=(cx, cy, 0)))
    bodies.append(Body("merlin_engines", *kit.merge(bells), "nozzle_dark", "engines"))
    # cable raceway along the first stage
    bodies.append(Body("raceway", *kit.box((0.22, 0.30, z["s1_top"] - 4.5), (0, r + 0.08, (z["s1_top"] + 4.5) / 2 - 0.5)),
                       "lv_white", "vehicle", aero=False))
    # landing legs: hinge at the base, stowed along the stage (0 deg) or deployed (~112 deg out and down)
    leg_len, legw, legt = 9.6, 1.25, 0.40
    nl = _n(10, res, 6)
    u = np.linspace(0, 1, nl)
    alpha = 115.0 if p["legs_deployed"] else 0.0
    hinge_r, hinge_z = r + 0.05, 1.7
    for k in range(4):
        az = math.radians(45 + 90 * k)
        rings = []
        for ui in u:
            th = legt * (0.55 + 0.45 * math.sin(0.5 * math.pi * min(ui / 0.3, 1.0))) * (1 - 0.7 * ui ** 8)
            ww = legw * (1 - 0.55 * ui) * (1 - 0.5 * ui ** 10)
            sq = kit.superellipse(ww / 2, th / 2, 12, 5.0)
            # local: s along the leg (+z stowed), y' tangential, w radial
            rings.append(np.c_[th / 2 + sq[:, 1], sq[:, 0], np.full(12, ui * leg_len)])
        V, F = kit.loft(np.array(rings), "cap", "cap")
        V = V @ kit.rot((0, 1, 0), alpha).T            # rotate outward about the tangential hinge axis
        V = V + [hinge_r, 0, hinge_z]
        V = V @ kit.rot((0, 0, 1), math.degrees(az)).T
        bodies.append(Body(f"landing_leg_{k + 1}", V, F, "carbon_leg", "legs"))
        if p["legs_deployed"]:
            a = math.radians(alpha)
            pl = np.array([hinge_r + 0.62 * leg_len * math.sin(a), 0, hinge_z + 0.62 * leg_len * math.cos(a)])
            pb = np.array([r - 0.05, 0, hinge_z + 5.2])
            Rz = kit.rot((0, 0, 1), math.degrees(az))
            bodies.append(Body(f"leg_strut_{k + 1}", *kit.cylinder(Rz @ pb, Rz @ pl, 0.11, 12), "titanium", "legs"))
            foot = np.array([hinge_r + leg_len * math.sin(a) + 0.1, 0, hinge_z + leg_len * math.cos(a) - 0.05])
            bodies.append(Body(f"leg_foot_{k + 1}", *kit.cylinder(Rz @ (foot + [0, 0, -0.12]), Rz @ (foot + [0, 0, 0.12]),
                                                                  0.45, 16), "titanium", "legs"))
    # grid fins on the interstage, hinged at their root, 0 deg stowed (span up along the body), 90 deg deployed
    Vg, Fg = _grid_fin(res)
    dep = p["grid_fin_angle"]
    for k in range(4):
        az = 45 + 90 * k
        V = Vg @ kit.rot((0, 1, 0), dep).T          # span (+z) towards +x (radial) as it deploys
        V = V + [r + 0.16, 0, 45.6]
        V = V @ kit.rot((0, 0, 1), az).T
        bodies.append(Body(f"grid_fin_{k + 1}", V, Fg.copy(), "titanium", "grid_fins"))
        Vm, Fm = kit.box((0.36, 0.5, 0.35), (r + 0.06, 0, 45.6))
        bodies.append(Body(f"grid_fin_actuator_{k + 1}", Vm @ kit.rot((0, 0, 1), az).T, Fm, "lv_black", "grid_fins"))
    return bodies


register(GeometrySpec(
    name="falcon9_booster", title="Two-stage launcher (Falcon 9 Block 5 class)", category="space",
    description=("Two-stage medium-lift launcher: 3.7 m core with a white first stage, black interstage carrying four "
                 "titanium lattice grid fins, white second stage and a 5.2 m x 13.1 m ogive payload fairing; nine "
                 "Merlin-class bells in an octaweb and four folding carbon landing legs. Grid-fin angle, leg "
                 "deployment and fairing on / off for ascent and booster-return aerodynamics."),
    builder=_falcon9,
    params=[
        Param("grid_fin_angle", 0.0, "deg", (0.0, 90.0), "grid-fin deployment angle (0 = stowed against the "
              "interstage, 90 = deployed perpendicular to the flow)"),
        Param("legs_deployed", False, "", None, "landing legs deployed (115 deg, feet below the engines)"),
        Param("fairing", True, "", None, "payload fairing on (off: second-stage dome, adapter and a payload)"),
    ],
    dimensions={"length_x": (5.2, 0.02), "width_y": (5.2, 0.02), "height_z": (70.0, 0.01)},
    frame=("right-handed, metres: z up along the vehicle axis, x / y lateral; origin on the axis in the plane of the "
           "first-stage nozzle exits; legs and grid fins at 45 / 135 / 225 / 315 deg azimuth"),
    sources=[Source("height 70 m, core diameter 3.7 m, nine first-stage engines, black interstage, grid fins and "
                    "four landing legs", SPACEX_F9),
             Source("payload fairing outer diameter 5.2 m, height 13.1 m", SPACEX_UG),
             Source("stage break-down (first stage to 42.6 m, interstage to 47.3 m, second stage to 56.9 m), ogive "
                    "shape, bell contour (exit diameter 0.86 m), octaweb ring radius 1.2 m", "estimates from "
                    "published photographs and the overall dimensions", estimate=True),
             Source("grid fin 1.5 x 1.3 x 0.22 m with a 45-deg lattice, leg length 9.6 m, deployed leg angle "
                    "115 deg, leg / fin azimuths", "estimates from photographs", estimate=True)],
    physics=["ascent aerodynamics (axial force, transonic buffet on the fairing / boattail)",
             "booster re-entry and landing burn: grid-fin control authority, base heating of the octaweb",
             "supersonic retro-propulsion plume-body interaction", "structural loads: wind / max-q bending"],
    materials=MATERIALS, tags=("rocket", "reusable booster", "grid fins"),
    camera={"azimuth": 215.0, "elevation": 10.0},
))
