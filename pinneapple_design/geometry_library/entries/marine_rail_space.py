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
    # landing legs: hinge at the base, stowed along the stage (0 deg) or deployed (115 deg, out and down)
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


# ============================================================================================ high-speed train
SIEMENS_VELARO = ("Siemens Mobility (2018). Velaro D / ICE 3 (BR 407) and Velaro China (CRH3) technical data: car "
                  "body width 2,950 mm, height above top of rail 3,890 mm, end-car length 25,675 mm, intermediate-car "
                  "length 24,775 mm, bogie-centre distance 17,375 mm, wheel diameter 920 mm (new). Velaro product "
                  "data sheets, https://www.mobility.siemens.com/global/en/portfolio/rail/rolling-stock/high-speed-"
                  "and-intercity-trains/velaro.html")
EN14067 = ("CEN (2018). EN 14067-4: Railway applications - Aerodynamics - Part 4: Requirements and test procedures "
           "for aerodynamics on open track (top of rail as the reference plane, standard gauge 1435 mm)")
UIC60 = "CEN (2017). EN 13674-1: Railway applications - Track - Rail - Vignole railway rails 46 kg/m and above (60E1)"

TRAIN_W, TRAIN_H, TRAIN_ZB = 2.95, 3.89, 0.42
END_CAR, MID_CAR, GAP = 25.675, 24.775, 0.60
TR_EB, TR_ET = 6.0, 3.4                                     # super-ellipse exponents of the body section


def _theta_of_z(z, side=1):
    """Ring angle on the full body section at height ``z`` (starboard side; ``side=-1``: port)."""
    zc, hh = 0.5 * (TRAIN_ZB + TRAIN_H), 0.5 * (TRAIN_H - TRAIN_ZB)
    e = TR_ET if z > zc else TR_EB
    s = math.copysign(abs((z - zc) / hh) ** (e / 2), z - zc)
    th = math.asin(np.clip(s, -1, 1))
    return th if side > 0 else math.pi - th


def _train_thetas(res):
    n = 2 * _n(22, res, 16)
    base = np.linspace(0, 2 * math.pi, n, endpoint=False)
    forced = []
    for z in (0.80, 1.02, 1.30, 2.02, 2.74, 3.10):
        forced += [_theta_of_z(z, 1), _theta_of_z(z, -1)]
    forced += [math.radians(a) for a in (38.0, 142.0, -30.0, -14.0, 194.0, 210.0)]
    forced = np.mod(np.array(forced), 2 * math.pi)
    d = np.abs((base[:, None] - forced[None, :] + math.pi) % (2 * math.pi) - math.pi).min(1)
    th = np.sort(np.r_[base[d > 0.55 * (2 * math.pi / n)], forced])
    return th


def _ring(x, hw, zb, zt, eb, et, th):
    c, s = np.cos(th), np.sin(th)
    zc, hh = 0.5 * (zb + zt), 0.5 * (zt - zb)
    y = hw * np.sign(c) * np.abs(c) ** (2 / np.where(s > 0, et, eb))
    z = zc + hh * np.sign(s) * np.abs(s) ** (2 / np.where(s > 0, et, eb))
    return np.c_[np.full(len(th), x), y, z]


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _car(kind, Lcar, Ln, res, th):
    """One car body as a closed loft, x in [0, Lcar] (head car: nose at x = 0). Returns (V, F, face materials)."""
    W2, ZT, ZB = TRAIN_W / 2, TRAIN_H, TRAIN_ZB
    z_tip = 0.95
    xs, prm = [], []
    if kind == "head":
        k = _n(22, res, 16)
        xi = (1 - np.cos(0.5 * math.pi * np.linspace(0, 1, k))) ** 0.8
        xi = np.unique(np.round(np.r_[xi[1:], 0.004, 0.08, 0.15, 0.24, 0.47, 0.78, 1.0], 6))
        for q in xi:
            hw = W2 * (1 - (1 - q) ** 2.4) ** 0.5
            zb = ZB + (z_tip - ZB) * (1 - math.sqrt(max(0.0, 1 - (1 - min(q / 0.25, 1.0)) ** 2)))
            zt = z_tip + (ZT - z_tip) * (1 - (1 - q) ** 1.6) ** 0.6
            w = float(_smooth(q / 0.9))
            xs.append(q * Ln)
            prm.append((hw, zb, zt, 2.5 + (TR_EB - 2.5) * w, 2.2 + (TR_ET - 2.2) * w))
        x_win0 = Ln + 2.4
    else:
        x_win0 = 2.1
    x_win1 = Lcar - 1.9
    pitch, ww = 1.45, 1.0
    nw = int((x_win1 - x_win0 + (pitch - ww)) // pitch)
    x0w = x_win0 + 0.5 * ((x_win1 - x_win0) - (nw * pitch - (pitch - ww)))
    wins = [(x0w + i * pitch, x0w + i * pitch + ww) for i in range(nw)]
    body_x = [x for wv in wins for x in wv]
    xstart = xs[-1] if xs else 0.0
    ends = [Lcar - 0.25, Lcar - 0.08, Lcar]
    starts = [] if kind == "head" else [0.0, 0.08, 0.25]
    bx = np.unique(np.r_[starts, xstart + 0.5, body_x, ends])
    bx = bx[bx > xstart + 1e-6] if kind == "head" else bx
    for x in bx:
        d_end = min(Lcar - x, x if kind != "head" else 1e9)
        sc = 1.0 - 0.05 * (1 - min(d_end / 0.25, 1.0)) ** 2
        zc = 0.5 * (ZB + ZT)
        hh = 0.5 * (ZT - ZB) * sc
        xs.append(x)
        prm.append((W2 * sc, zc - hh, zc + hh, TR_EB, TR_ET))
    rings = [_ring(x, *pp, th) for x, pp in zip(xs, prm, strict=False)]
    nr, n = len(rings), len(th)
    V, F = kit.loft(np.array(rings))
    if kind == "head":
        V[nr * n] = (0.0, 0.0, z_tip)                            # pointed / rounded nose tip (pole)
    # face materials from (station interval, ring interval)
    xs = np.asarray(xs)
    i = np.r_[np.repeat(np.arange(nr - 1), n), np.repeat(np.arange(nr - 1), n)]
    j = np.r_[np.tile(np.arange(n), nr - 1), np.tile(np.arange(n), nr - 1)]
    xm = 0.5 * (xs[i] + xs[i + 1])
    tm = np.mod(0.5 * (th[j] + np.r_[th[1:], th[0] + 2 * math.pi][j]), 2 * math.pi)
    ts = np.where(tm > math.pi, tm - 2 * math.pi, tm)                         # (-pi, pi]
    tr = np.where(np.cos(tm) >= 0, ts, np.sign(ts) * math.pi - ts)           # folded onto the starboard side
    tz = {z: _theta_of_z(z) for z in (0.80, 1.02, 1.30, 2.02, 2.74)}
    mats = np.full(len(xm), "train_white", dtype=object)
    mats[tr < tz[0.80]] = "train_grey"
    stripe = (tr > tz[1.02]) & (tr < tz[1.30])
    mats[stripe & (xm > 0.2 * Ln if kind == "head" else True)] = "train_red"
    inwin = np.zeros(len(xm), bool)
    for a, b in wins:
        inwin |= (xm > a) & (xm < b)
    mats[inwin & (tr > tz[2.02]) & (tr < tz[2.74])] = "glass"
    if kind == "head":
        q = xm / Ln
        mats[(q > 0.47) & (q < 0.78) & (tm > math.radians(38)) & (tm < math.radians(142))] = "glass"
        mats[(q > 0.08) & (q < 0.15) & (tr > math.radians(-30)) & (tr < math.radians(-14))] = "led"
        # cab side window behind the windscreen
        mats[(xm > 0.80 * Ln) & (xm < Ln + 1.4) & (tr > tz[2.02]) & (tr < tz[2.74]) & (np.cos(tm) != 0)] = "glass"
    mats = np.r_[mats, np.full(len(F) - len(mats), "train_white", dtype=object)]
    return V, F, mats


def _bogie(xc, res):
    """Two wheelsets (920 mm wheels on standard gauge) and the two side frames, centred at x = ``xc``."""
    nw = _n(20, res, 14)
    rw, yw, tw = 0.46, 0.7535, 0.135
    parts = []
    for dx in (-1.25, 1.25):
        for sy in (-1, 1):
            parts.append(kit.cylinder((xc + dx, sy * (yw - tw / 2), rw), (xc + dx, sy * (yw + tw / 2), rw), rw, nw))
    for sy in (-1, 1):
        parts.append(kit.rounded_box((3.6, 0.18, 0.36), (xc, sy * 0.98, 0.50), radius=0.06, n_side=2))
    return kit.merge(parts)


def _pantograph(x0, res):
    nt = _n(10, res, 8)
    zr = TRAIN_H - 0.02
    out = []
    ins = kit.merge([kit.cylinder((x0 + dx, dy, zr - 0.05), (x0 + dx, dy, zr + 0.28), 0.075, nt)
                     for dx, dy in ((0.0, 0.45), (0.0, -0.45), (1.6, 0.0))])
    out.append(("pantograph_insulators", ins, "insulator"))
    out.append(("pantograph_base", kit.box((1.9, 1.10, 0.08), (x0 + 0.8, 0, zr + 0.31)), "steel"))
    out.append(("pantograph_lower_arm", kit.tube([(x0 + 0.1, 0, zr + 0.36), (x0 + 1.0, 0, zr + 0.46),
                                                  (x0 + 1.9, 0, zr + 0.58)], 0.05, nt), "steel"))
    out.append(("pantograph_upper_arm", kit.tube([(x0 + 1.86, 0, zr + 0.60), (x0 + 1.0, 0, zr + 0.66),
                                                  (x0 + 0.15, 0, zr + 0.72)], 0.035, nt), "steel"))
    out.append(("pantograph_head", kit.box((0.22, 1.60, 0.05), (x0 + 0.15, 0, zr + 0.78)), "steel"))
    horns = kit.merge([kit.tube([(x0 + 0.15, s * 0.74, zr + 0.78), (x0 + 0.15, s * 0.92, zr + 0.76),
                                 (x0 + 0.15, s * 1.00, zr + 0.69)], 0.022, 6) for s in (-1, 1)])
    out.append(("pantograph_horns", horns, "steel"))
    return out


def _train(p, res):
    Ln, nc = p["nose_length"], int(p["n_cars"])
    th = _train_thetas(res)
    bodies = []
    kinds = ["head"] if nc == 1 else (["head", "tail"] if nc == 2 else ["head", "middle", "tail"])
    x = 0.0
    spans = []
    for k, kind in enumerate(kinds):
        Lcar = MID_CAR if kind == "middle" else END_CAR
        V, F, mats = _car("middle" if kind == "middle" else "head", Lcar, Ln, res, th)
        if kind == "tail":
            V, F = kit.mirror((V, F), 0)
            V = V + [Lcar, 0, 0]
        V = V + [x, 0, 0]
        bodies.append(Body(f"car_{k + 1}_body", V, F, mats, f"car_{k + 1}"))
        # bogies at the published bogie-centre distance, centred on the car
        bog = kit.merge([_bogie(x + Lcar / 2 + s * 17.375 / 2, res) for s in (-1, 1)])
        bodies.append(Body(f"car_{k + 1}_bogies", *bog, "bogie_dark", f"car_{k + 1}", aero=False))
        fair = kit.merge([kit.rounded_box((3.9, 0.06, 0.40), (x + Lcar / 2 + s * 17.375 / 2, sy * 1.28, 0.57),
                                          radius=0.04, n_side=2) for s in (-1, 1) for sy in (-1, 1)])
        bodies.append(Body(f"car_{k + 1}_bogie_fairings", *fair, "train_grey", f"car_{k + 1}"))
        spans.append((x, x + Lcar))
        x += Lcar + GAP
    Ltot = x - GAP
    for k in range(len(spans) - 1):
        a, b = spans[k][1], spans[k + 1][0]
        gw = kit.rounded_box((b - a + 0.6, 2.60, 3.0), ((a + b) / 2, 0, 0.62 + 1.5), radius=0.3, n_side=2)
        bodies.append(Body(f"gangway_{k + 1}", *gw, "train_grey", "gangways"))
    # pantograph (lowered) on the roof of the second car, or of the head car when it runs alone
    car_p = spans[1] if nc >= 2 else spans[0]
    xp = car_p[0] + (6.0 if nc >= 2 else END_CAR - 7.5)
    for nm, mesh, mat in _pantograph(xp, res):
        bodies.append(Body(nm, *mesh, mat, "pantograph"))
    # track: two 60E1-like rails (top of rail at z = 0) on concrete sleepers at 0.6 m pitch
    prof = np.array([[-0.075, -0.172], [0.075, -0.172], [0.075, -0.160], [0.0085, -0.140], [0.0085, -0.051],
                     [0.036, -0.045], [0.036, 0.0], [-0.036, 0.0], [-0.036, -0.045], [-0.0085, -0.051],
                     [-0.0085, -0.140], [-0.075, -0.160]])
    rails = kit.merge([kit.extrude(prof + [s * 0.7535, 0], 0.0, Ltot, plane="yz") for s in (-1, 1)])
    bodies.append(Body("rails", *rails, "rail_steel", "track", aero=False))
    ns = int(Ltot // 0.6)
    xs0 = 0.5 * (Ltot - (ns - 1) * 0.6)
    sl = _boxes([((0.26, 2.6, 0.22), (xs0 + i * 0.6, 0, -0.172 - 0.11)) for i in range(ns)])
    bodies.append(Body("sleepers", *sl, "sleeper_concrete", "track", aero=False))
    return bodies


register(GeometrySpec(
    name="high_speed_train", title="High-speed train (ICE 3 / CRH3 class)", category="rail",
    description=("Streamlined head car with a long nose, wrap-around windscreen and headlights, passenger window "
                 "band and red livery stripe, bogies with 920 mm wheelsets behind fairings, gangway in the inter-car "
                 "gap, a lowered pantograph on the roof and a trailing end car, on two rails and sleepers (top of "
                 "rail z = 0). Nose length and number of cars (1-3) as parameters, zero yaw."),
    builder=_train,
    params=[
        Param("nose_length", 10.0, "m", (6.0, 15.0), "length of the streamlined nose (tip to full section)"),
        Param("n_cars", 2, "-", (1, 3), "cars: 1 = head car, 2 = head + trailing end car, 3 = head + "
              "intermediate + trailing end car"),
    ],
    dimensions={"length_x": (51.95, 0.01), "width_y": (2.95, 0.02), "height_z": (5.07, 0.02)},
    frame=FRAME_VEHICLE.replace("z = 0 on the ground for ground vehicles", "z = 0 at the top of rail (EN 14067 "
                                "reference plane); rails and sleepers lie below z = 0"),
    sources=[Source("car width 2.95 m, height 3.89 m above top of rail, end car 25.675 m, intermediate car 24.775 m, "
                    "bogie-centre distance 17.375 m, wheel diameter 0.92 m", SIEMENS_VELARO),
             Source("top of rail as the aerodynamic reference plane, standard gauge 1.435 m", EN14067),
             Source("rail profile (60E1-like, 172 mm high, 72 mm head, 150 mm foot)", UIC60),
             Source("nose profile and plan-form curves, windscreen extent, window pitch 1.45 m, inter-car gap 0.6 m, "
                    "bogie wheelbase 2.5 m, pantograph geometry, sleeper 2.6 x 0.26 x 0.22 m at 0.6 m",
                    "estimates from photographs of ICE 3 / CRH3 trains", estimate=True)],
    physics=["train aerodynamics: drag, slipstream and the nose pressure pulse (EN 14067)",
             "crosswind stability (side force and roll moment; yaw by rotating the inflow)",
             "tunnel entry micro-pressure waves and passing-train loads",
             "pantograph aero-acoustics and under-body / ballast flow"],
    materials=MATERIALS, tags=("high-speed rail", "train aerodynamics", "slipstream"),
    camera={"azimuth": 208.0, "elevation": 12.0},
))


# ============================================================================================ KCS container ship
KIM2001 = ("Kim, W.J., Van, S.H., Kim, D.H. (2001). Measurement of flows around modern commercial ship models. "
           "Experiments in Fluids 31(5), 567-578. https://doi.org/10.1007/s003480100332")
TOKYO2015 = ("Hino, T., Stern, F., Larsson, L., Visonneau, M., Hirata, N., Kim, J. (eds.) (2021). Numerical Ship "
             "Hydrodynamics: An Assessment of the Tokyo 2015 Workshop. Lecture Notes in Applied and Computational "
             "Mechanics 94, Springer. https://doi.org/10.1007/978-3-030-47572-7 (KCS: Lpp 230 m, B 32.2 m, "
             "T 10.8 m, D 19.0 m, Cb 0.6505, propeller D 7.9 m, 5 blades, scale 1:31.6)")
SIMMAN = ("SIMMAN 2008 / 2014 workshop on verification and validation of ship manoeuvring simulation methods: KCS "
          "geometry and conditions. http://www.simman2014.dk/ship-data/moeri-container-ship-kcs/")

KCS_L, KCS_B, KCS_T, KCS_D = 230.0, 32.2, 10.8, 19.0
KCS_XT = 236.0                      # transom, aft of the AP
KCS_RAKE = 0.55                     # stem rake above the waterline (m forward per m up)
KCS_PROP = (223.6, 4.4)             # propeller centre (x, z above the keel)
KCS_BULB = (7.5, 11.0, 5.6, 3.9)  # bulb: centre x, half-length, centre z, half-height
KCS_BOX = (12.192, 2.438, 2.591)    # 40-ft ISO container
BOX_COLOURS = ("box_red", "box_blue", "box_lightblue", "box_green", "box_orange", "box_grey", "box_white",
               "box_brown", "box_yellow", "box_blue", "box_red", "box_grey")


def _kcs_deck(x):
    return KCS_D + 1.6 * float(np.clip((28.0 - x) / 28.0, 0, 1)) ** 2          # forecastle sheer


def _kcs_W(x, z):
    """Half-breadth of the waterline at height z (above the keel) at station x (aft of the FP)."""
    Bh = KCS_B / 2
    s = float(np.clip((z - KCS_T) / (KCS_D - KCS_T), 0, 1.3))
    xa, xb = 0.44 * KCS_L, 0.53 * KCS_L
    if x < xa:
        xs = -KCS_RAKE * max(z - KCS_T, 0.0)
        u = float(np.clip((x - xs) / (xa - xs), 0, 1))
        return Bh * (1 - (1 - u) ** (1.6 + 1.2 * s))
    if x <= xb:
        return Bh
    v = float(np.clip((x - xb) / (KCS_XT - xb), 0, 1))
    wt = 0.50 + 0.32 * min(s, 1.0)
    return Bh * (1 - (1 - wt) * v ** (2.1 - 0.3 * min(s, 1.0)))


def _kcs_zk(x):
    """Height of the bottom of station x above the keel: stem, forefoot, flat keel, skeg cut-up, counter."""
    if x < 0:
        return KCS_T + (-x) / KCS_RAKE
    if x < 18:
        # forefoot faired into the underside of the bulb
        xc, a, zc, rz = KCS_BULB
        bb = zc - rz * math.sqrt(max(0.0, 1 - ((x - xc) / a) ** 2))
        return (bb + 0.8) * (1 - float(_smooth((x - 2.0) / 16.0)))
    if x <= 200:
        return 0.0
    if x <= 221.0:
        return 3.4 * ((x - 200) / 21) ** 1.6
    return 9.4 + 1.8 * (x - 221.9) / (KCS_XT - 221.9)


def _kcs_e(x):
    """Super-ellipse exponent of the underwater section (10 = midship box, 1 = V, < 1 = hollow skeg)."""
    if x < 14:
        return 2.0
    if x < 101:
        return 2.0 + 8.0 * float(_smooth((x - 14) / 87))
    if x < 122:
        return 10.0
    if x < 200:
        return 10.0 - 8.8 * float(_smooth((x - 122) / 78)) ** 0.7
    if x <= 221.0:
        return 1.2 - 0.6 * (x - 200) / 21
    return 4.0


def _kcs_half(x, z, zk=None, e=None):
    zk = _kcs_zk(x) if zk is None else zk
    if z < KCS_T and zk < KCS_T:
        e = _kcs_e(x) if e is None else e
        t = float(np.clip((z - zk) / (KCS_T - zk), 0, 1))
        return _kcs_W(x, KCS_T) * (1 - (1 - t) ** e) ** (1 / e)
    return _kcs_W(x, z)


def _kcs_section(x, zw, nA, nB, nd):
    """Closed ring of hull station x (hull frame), with a ring point on the local waterline z = zw."""
    zk, e, zd = _kcs_zk(x), _kcs_e(x), _kcs_deck(x)
    yk = 0.12
    pts = [(0.0, zk)]
    if zk < KCS_T:
        t = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, 160)))
        for z in zk + t * (KCS_T - zk):
            pts.append((max(_kcs_half(x, z, zk, e), yk), z))
        z_up = np.linspace(KCS_T, zd, 50)[1:]
    else:
        z_up = np.linspace(zk, zd, 50)
    for z in z_up:
        pts.append((max(_kcs_W(x, z), yk), z))
    P = np.array(pts)
    if zk < zw < zd:
        k = int(np.searchsorted(P[1:, 1], zw)) + 1
        k = max(k, 2)
        a, b = P[k - 1], P[k]
        w = (zw - a[1]) / max(b[1] - a[1], 1e-12)
        pw = a + w * (b - a)
        A = np.vstack([P[:k], pw])
        B = np.vstack([pw, P[k:]])
    else:
        A, B = P[:2], P[1:]
    sa, sb = _resample(A, nA), _resample(B, nB)
    stb = np.vstack([sa, sb[1:]])
    bd = stb[-1, 0]
    deck = np.c_[bd * (1 - 2 * np.arange(1, nd + 1) / (nd + 1)), np.full(nd, zd)]
    port = stb[1:][::-1] * [-1, 1]
    yz = np.vstack([stb, deck, port])
    return np.c_[np.full(len(yz), x), yz]


def _kcs_stations(res):
    x0 = -KCS_RAKE * (_kcs_deck(-5.0) - 0.45 - KCS_T)
    seg = [(x0, 0.0, 6), (0.0, 18.0, 8), (18.0, 92.0, 11), (92.0, 129.0, 3), (129.0, 200.0, 12), (200.0, 220.9, 8)]
    xs = []
    for a, b, k in seg:
        kk = _n(k, res, 2)
        u = np.linspace(0, 1, kk + 1)[:-1]
        if a == x0:
            u = 1 - (1 - u) ** 1.5
        xs += list(a + (b - a) * u)
    xs += [220.9] + list(np.linspace(221.9, KCS_XT, _n(6, res, 4)))
    return np.array(xs)


def _kcs_propeller(res, nb=5, D=7.9):
    """Five-blade fixed-pitch propeller (P/D ~ 1.0, skewed elliptic blades) about the +x axis through the origin."""
    R = D / 2
    rh = 0.18 * R
    ns, nsec = _n(9, res, 7), _n(11, res, 8)
    xr = np.linspace(0, 1, ns)
    xr = 1 - (1 - xr) ** 1.3
    rr = rh * 0.85 + (R - rh * 0.85) * xr
    blades = []
    for k in range(nb):
        rings = []
        for x, r in zip(xr, rr, strict=False):
            c = D * (0.21 + 0.17 * math.sin(math.pi * min(x, 0.999) ** 0.75)) * math.sqrt(max(1 - 0.94 * x ** 2.2, 0.02))
            tc = 0.20 - 0.155 * x
            phi = math.atan(0.997 * D / (2 * math.pi * max(r, 1e-6)))
            e1 = np.array([math.sin(phi), -math.cos(phi), 0.0])
            e2 = np.array([math.cos(phi), math.sin(phi), 0.0])
            sec = kit.naca4(0.02, 0.45, tc, nsec)
            ring = kit.section_ring(sec, c, 0.0, np.array([0, 0, r]) - 0.5 * c * e1, frame=(e1, e2), pivot=0.5)
            skew = 22.0 * x ** 1.8
            ring = ring @ kit.rot((1, 0, 0), skew + 360.0 * k / nb).T
            rings.append(ring)
        blades.append(kit.loft(np.array(rings)))
    return kit.merge(blades)


def _kcs(p, res):
    Td, trim, tiers = p["draft"], p["trim"], int(p["container_tiers"])
    th = math.atan(trim / KCS_L)
    R = kit.rot((0, 1, 0), math.degrees(th))
    piv = np.array([KCS_L / 2, 0.0, Td])

    def W(mesh):
        V, F = mesh
        return (np.asarray(V, float) - piv) @ R.T + [KCS_L / 2, 0, 0], F

    bodies = []
    # ---------------------------------------------------------------- hull
    nA, nB, nd = _n(9, res, 6), _n(7, res, 5), _n(3, res, 2)
    rings = [_kcs_section(x, Td + (x - KCS_L / 2) * math.tan(th), nA, nB, nd) for x in _kcs_stations(res)]
    Vh, Fh = W(kit.loft(np.array(rings)))
    c, nrm = _centres(Vh, Fh), _normals(Vh, Fh)
    mats = np.where(c[:, 2] < -1e-6, "antifouling_red", "hull_navy").astype(object)
    mats[(nrm[:, 2] > 0.9) & (c[:, 2] > KCS_D - Td - 1.5)] = "deck_green"
    bodies.append(Body("hull", Vh, Fh, mats, "hull"))
    # bulbous bow
    Vb, Fb = W(kit.ellipsoid((KCS_BULB[0], 0, KCS_BULB[2]), (KCS_BULB[1], 2.6, KCS_BULB[3]), _n(28, res, 20), _n(14, res, 10)))
    bodies.append(Body("bulbous_bow", Vb, Fb, "antifouling_red", "hull"))
    # ---------------------------------------------------------------- appendages
    xp, zp = KCS_PROP
    hx = np.r_[220.2, 221.0, np.linspace(222.0, 225.0, 3), 225.6, 226.1, 226.4]
    hr = np.r_[0.75, 0.78, 0.80, 0.80, 0.78, 0.62, 0.35, 0.0]
    hub = kit.revolve(hx, hr, _n(20, res, 14), axis="x", center=(0, 0, zp))
    bodies.append(Body("propeller_hub", *W(hub), "bronze", "propeller"))
    Vp, Fp = _kcs_propeller(res)
    bodies.append(Body("propeller_blades", *W((Vp + [xp, 0, zp], Fp)), "bronze", "propeller"))
    nfoil = _n(14, res, 10)
    rud = kit.wing([{"le": (227.7, 0, 0.9), "chord": 5.3, "t": 0.18}, {"le": (227.5, 0, 6.0), "chord": 5.7, "t": 0.18},
                    {"le": (227.3, 0, 11.6), "chord": 6.0, "t": 0.18}], nfoil, frame=((1, 0, 0), (0, 1, 0)))
    bodies.append(Body("rudder", *W(rud), "antifouling_red", "rudder"))
    # ---------------------------------------------------------------- deck structures (hull frame, z above keel)
    D = KCS_D
    chev = np.array([[16.6, 0.0], [20.3, 8.6], [20.75, 8.6], [17.05, 0.0], [20.75, -8.6], [20.3, -8.6]])
    bodies.append(Body("breakwater", *W(kit.extrude(chev, _kcs_deck(18) - 0.3, _kcs_deck(18) + 3.2)),
                       "superstructure_white", "deck"))
    bodies.append(Body("foremast", *W(kit.cylinder((7.0, 0, _kcs_deck(7) - 0.3), (7.0, 0, 31.0), 0.35, 12, 0.22)),
                       "superstructure_white", "deck", aero=False))
    # accommodation block, bridge deck with wings, mast and radar
    xa0, xa1, ha, ztop = 186.5, 197.5, 12.0, D + 6 * 2.9
    bodies.append(Body("deckhouse", *W(kit.box((xa1 - xa0, 2 * ha, ztop - D + 0.5), ((xa0 + xa1) / 2, 0, (ztop + D - 0.5) / 2))),
                       "superstructure_white", "superstructure"))
    zb0, zb1 = ztop, ztop + 3.0
    bodies.append(Body("bridge", *W(kit.box((8.5, KCS_B - 0.3, zb1 - zb0), (186.0 + 4.25, 0, (zb0 + zb1) / 2))),
                       "superstructure_white", "superstructure"))
    win = []
    for k in range(6):
        z0 = D + 2.9 * k + 1.0
        for yc in np.linspace(-10.4, 10.4, 9):
            win.append(((0.16, 1.5, 1.0), (xa0 - 0.04, yc, z0 + 0.5)))
        for xc in (188.6, 191.2, 193.8, 196.2):
            for sy in (-1, 1):
                win.append(((1.3, 0.16, 1.0), (xc, sy * (ha + 0.04), z0 + 0.5)))
    bodies.append(Body("deckhouse_windows", *W(_boxes(win)), "bridge_glass", "superstructure", aero=False))
    bg = [((0.16, KCS_B - 0.6, 1.5), (185.96, 0, zb0 + 1.55))]
    for sy in (-1, 1):
        bg.append(((6.5, 0.16, 1.5), (190.4, sy * (KCS_B / 2 - 0.11), zb0 + 1.55)))
    bodies.append(Body("bridge_windows", *W(_boxes(bg)), "bridge_glass", "superstructure", aero=False))
    bodies.append(Body("main_mast", *W(kit.cylinder((191.0, 0, zb1 - 0.2), (191.0, 0, 47.0), 0.45, 14, 0.25)),
                       "superstructure_white", "superstructure", aero=False))
    bodies.append(Body("radar", *W(kit.box((0.5, 6.0, 0.3), (191.0, 0, 44.0))), "funnel_black", "superstructure",
                       aero=False))
    for sy in (-1, 1):
        lb = kit.ellipsoid((192.0, sy * 13.45, D + 7.0), (4.3, 1.35, 1.25), _n(18, res, 12), _n(8, res, 6))
        bodies.append(Body(f"lifeboat_{'stb' if sy > 0 else 'port'}", *W(lb), "lifeboat_orange", "superstructure",
                           aero=False))
    # engine casing and funnel (raked aft, black top)
    bodies.append(Body("engine_casing", *W(kit.box((14.0, 12.0, 9.0), (208.0, 0, D + 4.0))), "superstructure_white",
                       "superstructure"))
    nf = _n(28, res, 20)
    fr = []
    for z in np.r_[np.linspace(D + 7.0, 41.8, _n(5, res, 3)), 44.0]:
        se = kit.superellipse(4.2, 2.7, nf, 2.6)
        fr.append(np.c_[208.0 + 0.12 * (z - D) + se[:, 0], se[:, 1], np.full(nf, z)])
    Vf, Ff = kit.loft(np.array(fr))
    cf = _centres(Vf, Ff)
    bodies.append(Body("funnel", *W((Vf, Ff)), np.where(cf[:, 2] > 41.8, "funnel_black", "funnel_red"),
                       "superstructure"))
    # ---------------------------------------------------------------- hatch covers and containers on deck
    rng = np.random.default_rng(7)
    Lb, Wb, Hb = KCS_BOX
    pitch_x, pitch_y = Lb + 0.75, 2.46
    bays = [22.0 + i * pitch_x for i in range(int((184.0 - 22.0 + 0.75) // pitch_x))] + [214.6]
    hatch, boxes, bmat = [], [], []
    for bi, xb0 in enumerate(bays):
        xb1 = xb0 + Lb
        half = min(_kcs_W(x, _kcs_deck(x)) for x in np.linspace(xb0, xb1, 7))
        zdk = max(_kcs_deck(xb0), _kcs_deck(xb1))
        hatch.append(((Lb + 0.3, 2 * (half - 0.9), 1.6), ((xb0 + xb1) / 2, 0, zdk + 0.8 - 0.3)))
        nrow = min(13, int(2 * (half - 0.05) // pitch_y))
        nt = tiers if 0 < bi < len(bays) - 1 else max(tiers - 1, 0)
        for t in range(nt):
            for r in range(nrow):
                yc = (r - (nrow - 1) / 2) * pitch_y
                if t == nt - 1 and nrow > 4 and (r == 0 or r == nrow - 1) and rng.random() < 0.5:
                    continue
                boxes.append(((Lb, Wb, Hb), ((xb0 + xb1) / 2, yc, zdk + 1.3 + 0.03 + t * (Hb + 0.03) + Hb / 2)))
                bmat.append(BOX_COLOURS[int(rng.integers(len(BOX_COLOURS)))])
    bodies.append(Body("hatch_covers", *W(_boxes(hatch)), "hatch_grey", "cargo"))
    if boxes:
        Vc, Fc = W(_boxes(boxes))
        bodies.append(Body("containers", Vc, Fc, np.repeat(np.array(bmat, dtype=object), 12), "cargo"))
    return bodies


register(GeometrySpec(
    name="container_ship_kcs", title="KRISO Container Ship (KCS), full scale", category="marine",
    description=("3600-TEU class container ship at full scale (Lpp 230 m, B 32.2 m, T 10.8 m, Cb ~0.65) with "
                 "bulbous bow, transom stern, skeg, 5-blade propeller and rudder, aft deckhouse with bridge "
                 "windows and wings, funnel, hatch covers and stacked containers. The hull form is a parametric "
                 "approximation of the KCS lines (the official offsets are not reproduced); draft, trim and "
                 "container tiers are parameters."),
    builder=_kcs,
    params=[
        Param("draft", 10.8, "m", (7.0, 12.5), "mean draft at midships (the hull floats with z = 0 on the calm water)"),
        Param("trim", 0.0, "m", (-4.0, 4.0), "trim = aft minus forward draft (positive: by the stern)"),
        Param("container_tiers", 4, "-", (0, 6), "tiers of 40-ft containers stacked on the hatch covers"),
    ],
    dimensions={"length_x": (241.1, 0.01), "width_y": (32.2, 0.01), "height_z": (47.0, 0.01)},
    frame=("right-handed, metres: x aft from the forward perpendicular (FP at x = 0, AP at x = 230), y to "
           "starboard, z up; z = 0 on the calm waterline (the keel at midships is at z = -draft)"),
    sources=[Source("KCS: Lpp 230 m, B 32.2 m, T 10.8 m, D 19.0 m, Cb 0.651, bulbous bow and transom stern",
                    KIM2001),
             Source("KCS full-scale data and propeller (D = 7.9 m, 5 blades, P/D ~ 1.0)", TOKYO2015),
             Source("KCS manoeuvring definition (rudder, propeller)", SIMMAN),
             Source("hull form: super-ellipse sections on parametric waterlines tuned to Cb ~ 0.65, stem rake, "
                    "forefoot, skeg and counter; the official KCS offsets are NOT reproduced", "parametric "
                    "approximation", estimate=True),
             Source("bulb, rudder (5.3-6.0 m chord, NACA 0018), propeller blade outline and skew, deckhouse, bridge, "
                    "funnel, lifeboats, hatch covers, 13-row container stowage (40-ft ISO boxes 12.19 x 2.44 x "
                    "2.59 m), propeller centre 4.4 m above the keel", "estimates for a 3600-TEU class ship "
                    "(the KCS was conceived without superstructure)", estimate=True)],
    physics=["calm-water resistance and nominal / effective wake (RANS, free surface)",
             "self-propulsion with the 5-blade propeller and rudder (SIMMAN / Tokyo 2015 cases)",
             "seakeeping and added resistance in waves; trim optimisation",
             "wind loads on the superstructure and container stacks"],
    materials=MATERIALS, tags=("ship hydrodynamics", "container ship", "benchmark hull"),
    camera={"azimuth": 212.0, "elevation": 16.0},
))
