"""Parametric vehicle CAD: closed-mesh primitives for rover-class vehicles, parts with materials and masses, mass
properties (centre of mass and inertia from the meshes) and assembly into bodies for multibody simulation.

    from pinneapple_design.geometry.vehicle_cad import VehicleCAD, beveled_box, rover_wheel, solar_panel
    cad = VehicleCAD("my rover")
    cad.add("chassis", "body", beveled_box((1.2, 0.9, 0.5), (0, 0, 0.8), 0.03), "white_paint", mass=200.0)
    cad.add("wheel_FL", "wheel_FL", *rover_wheel(0.25, 0.2, grousers=24, grouser_height=0.025))  # several parts
    props = cad.mass_properties()          # per body: mass, com, inertia about the com (body frame = world here)
    cad.scene().save("rover.glb")          # glTF with PBR materials, .stl, .usda

A *part* is one closed triangle mesh with a material and a mass (uniform density inside the mesh); a *body* is a
group of parts that move together (chassis, a suspension arm, a wheel). Parts that are thin plates are still closed
boxes so volume, centroid and inertia are well defined (divergence theorem, Eberly's polyhedral mass properties).

Coordinates: metres, z up, x forward, y left. Every function returns ``(vertices (V, 3), faces (F, 3))`` or, for the
assemblies (wheel, panel, camera head ...), a list of ``(suffix, (V, F), material, mass_share)``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------- basic closed primitives
def _faces_grid(nr, nc, closed=True):
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc if closed else nc - 1), indexing="ij")
    a = i * nc + j
    b = i * nc + (j + 1) % nc
    return np.concatenate([np.stack([a, b, b + nc], -1).reshape(-1, 3), np.stack([a, b + nc, a + nc], -1)
                           .reshape(-1, 3)])


def orient_outward(mesh: Mesh) -> Mesh:
    """Flip the winding of a closed mesh whose signed volume is negative."""
    V, F = mesh
    if signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def signed_volume(V, F) -> float:
    t = V[F]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def box(size, center=(0, 0, 0), R=None) -> Mesh:
    sx, sy, sz = np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, d in q for t in ([a, b, c], [a, c, d])], np.int64)
    if R is not None:
        v = v @ np.asarray(R).T
    return orient_outward((v + np.asarray(center, float), f))


def beveled_box(size, center=(0, 0, 0), bevel: float = 0.02, n: int = 3, R=None) -> Mesh:
    """Box with rounded edges of radius ``bevel`` (convex hull of eight corner spheres)."""
    from scipy.spatial import ConvexHull
    half = np.asarray(size, float) / 2 - bevel
    if np.any(half <= 0):
        return box(size, center, R)
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 4 * n, endpoint=False), np.linspace(0, np.pi, 2 * n + 1))
    s = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1).reshape(-1, 3)
    pts = np.concatenate([np.asarray(c) * half + bevel * s for c in
                          [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]])
    pts = np.unique(np.round(pts, 9), axis=0)
    h = ConvexHull(pts)
    V = pts
    t = pts[h.simplices]
    flip = np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), h.equations[:, :3]) < 0
    h.simplices[flip] = h.simplices[flip][:, ::-1]               # qhull facets are not consistently wound
    used = np.unique(h.simplices)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[h.simplices]
    if R is not None:
        V = V @ np.asarray(R).T
    return orient_outward((V + np.asarray(center, float), F))


def cylinder(p0, p1, radius, n: int = 32) -> Mesh:
    """Closed cylinder (or frustum, ``radius=(r0, r1)``) between two points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r0, r1 = (radius, radius) if np.isscalar(radius) else radius
    ax = p1 - p0
    L = np.linalg.norm(ax)
    ax = ax / L
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ax, seed)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    V = np.vstack([ring(p0, r0), ring(p1, r1), p0, p1])
    F = _faces_grid(2, n)
    c0, c1 = 2 * n, 2 * n + 1
    fan = [[c0, (j + 1) % n, j] for j in range(n)] + [[c1, n + j, n + (j + 1) % n] for j in range(n)]
    return orient_outward((V, np.vstack([F, fan])))


def tube_shell(axis_p0, axis_p1, r_in, r_out, n: int = 48) -> Mesh:
    """Closed annular tube (a wheel rim): inner and outer cylinders with annular end faces."""
    p0, p1 = np.asarray(axis_p0, float), np.asarray(axis_p1, float)
    ax = (p1 - p0) / np.linalg.norm(p1 - p0)
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(ax, seed)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    V = np.vstack([ring(p0, r_out), ring(p1, r_out), ring(p1, r_in), ring(p0, r_in)])
    F = _ring_faces(4, n)
    return orient_outward((V, F))


def _ring_faces(n_rings, n):
    """Faces joining ``n_rings`` rings cyclically (ring k to k+1, last to first)."""
    out = []
    for k in range(n_rings):
        a0, b0 = k * n, ((k + 1) % n_rings) * n
        for j in range(n):
            a, b = a0 + j, a0 + (j + 1) % n
            c, d = b0 + j, b0 + (j + 1) % n
            out += [[a, b, d], [a, d, c]]
    return np.array(out, np.int64)


def revolve_closed(profile, axis_origin=(0, 0, 0), axis=(0, 0, 1), n: int = 48) -> Mesh:
    """Closed solid of revolution of a closed polygon ``profile`` of (r, h) points about ``axis``."""
    p = np.asarray(profile, float)
    o, a = np.asarray(axis_origin, float), np.asarray(axis, float) / np.linalg.norm(axis)
    seed = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, seed)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = (o + p[:, 1, None, None] * a + p[:, 0, None, None] * (np.cos(th)[None, :, None] * u
                                                              + np.sin(th)[None, :, None] * w)).reshape(-1, 3)
    return orient_outward((V, _ring_faces(len(p), n)))


def paraboloid_dish(diameter: float, depth: float, thickness: float, center, axis, n: int = 40, m: int = 8) -> Mesh:
    """Dish antenna reflector: a paraboloid shell of ``depth`` over ``diameter`` (its vertex at ``center``)."""
    r = np.linspace(0, diameter / 2, m)
    z = depth * (r / (diameter / 2)) ** 2
    prof = np.r_[np.c_[r, z], np.c_[r[::-1], z[::-1] - thickness]]
    prof[0, 0] = prof[-1, 0] = 1e-4
    return revolve_closed(prof, center, axis, n)


def loft(rings, cap: str = "fan") -> Mesh:
    """Closed solid skinned through ``rings`` (m, n, 3): m closed polylines of n points each, in corresponding order.

    Consecutive rings are joined by quads; both ends are capped, either with a fan from the ring centroid
    (``cap="fan"``, right for convex or star-shaped rings) or with a strip joining point ``j`` to point ``n-1-j``
    (``cap="strip"``, right for thin rings made of an outward and a return polyline, n even)."""
    P = np.asarray(rings, float)
    m, n = P.shape[:2]
    V = P.reshape(-1, 3)
    F = []
    for k in range(m - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    top = (m - 1) * n
    if cap == "strip":
        for i in range(n // 2 - 1):
            p, q, r, s = i, i + 1, n - 2 - i, n - 1 - i
            F += [[q, p, s], [q, s, r], [top + p, top + q, top + r], [top + p, top + r, top + s]]
    else:
        c0, c1 = len(V), len(V) + 1
        V = np.vstack([V, P[0].mean(0), P[-1].mean(0)])
        F += [[c0, (j + 1) % n, j] for j in range(n)] + [[c1, top + j, top + (j + 1) % n] for j in range(n)]
    return orient_outward((V, np.asarray(F, np.int64)))


def offset_polygon(poly, d: float) -> np.ndarray:
    """Convex polygon (k, 2) moved inward by ``d`` along every edge normal (outward for ``d < 0``)."""
    p = np.asarray(poly, float)
    if _area2(p) < 0:
        p = p[::-1]
    e = np.roll(p, -1, 0) - p
    nrm = np.c_[-e[:, 1], e[:, 0]] / np.linalg.norm(e, axis=1)[:, None]     # inward normals of a CCW polygon
    out = []
    for i in range(len(p)):
        a0, n0, t0 = p[i - 1] + d * nrm[i - 1], nrm[i - 1], e[i - 1]
        a1, t1 = p[i] + d * nrm[i], e[i]
        A = np.array([t0, -t1]).T
        if abs(np.linalg.det(A)) < 1e-12:
            out.append(p[i] + d * n0)
        else:
            s = np.linalg.solve(A, a1 - a0)
            out.append(a0 + s[0] * t0)
    return np.asarray(out)


def _area2(p):
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def chamfered_rect(sx: float, sy: float, c: float = 0.0) -> np.ndarray:
    """Rectangle ``sx`` x ``sy`` centred at the origin with its four corners cut at 45 deg by ``c`` (CCW)."""
    hx, hy = sx / 2, sy / 2
    c = min(c, 0.49 * min(sx, sy))
    if c <= 0:
        return np.array([[hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
    return np.array([[hx, -hy + c], [hx, hy - c], [hx - c, hy], [-hx + c, hy], [-hx, hy - c], [-hx, -hy + c],
                     [-hx + c, -hy], [hx - c, -hy]])


def extrude_polygon(poly, z0: float, z1: float, chamfer_top: float = 0.0, chamfer_bottom: float = 0.0,
                    R=None, t=(0, 0, 0)) -> Mesh:
    """Prism of the convex polygon ``poly`` (k, 2) from ``z0`` to ``z1``, its top and bottom edges chamfered at 45 deg
    by ``chamfer_top`` / ``chamfer_bottom`` (the cap polygons are inset by that amount), then rotated by ``R`` and
    moved by ``t``. Use it for body panels with bevelled vertical edges (cut the polygon corners) and a bevelled
    top."""
    p = np.asarray(poly, float)
    if _area2(p) < 0:
        p = p[::-1]
    lev = []
    if chamfer_bottom > 0:
        lev.append((offset_polygon(p, chamfer_bottom), z0))
    lev.append((p, z0 + chamfer_bottom))
    lev.append((p, z1 - chamfer_top))
    if chamfer_top > 0:
        lev.append((offset_polygon(p, chamfer_top), z1))
    rings = [np.c_[q, np.full(len(q), z)] for q, z in lev]
    V, F = loft(rings)
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(t, float), F


def chamfered_box(size, center=(0, 0, 0), edge: float = 0.0, top: float = 0.0, bottom: float = 0.0, R=None) -> Mesh:
    """Box with flat 45 deg chamfers: ``edge`` on the four vertical edges, ``top`` / ``bottom`` on the edges of the
    top and bottom faces (a machined or sheet-metal look, as opposed to the rounded :func:`beveled_box`)."""
    sx, sy, sz = size
    V, F = extrude_polygon(chamfered_rect(sx, sy, edge), -sz / 2, sz / 2, top, bottom)
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(center, float), F


def _frame(axis, up):
    e = np.asarray(axis, float)
    e = e / np.linalg.norm(e)
    u = np.asarray(up, float) - np.dot(up, e) * e
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    u = u / np.linalg.norm(u)
    return e, u, np.cross(e, u)


def box_beam(p0, p1, section0, section1=None, up=(0, 0, 1), chamfer: float = 0.0) -> Mesh:
    """Straight (optionally tapered) box-section beam from ``p0`` to ``p1``. ``section`` = (width, height): height
    along ``up`` (projected normal to the beam), width across; ``section1`` tapers it to the far end; ``chamfer``
    cuts the four long edges (as a fraction of the smaller side of each section)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    _, u, v = _frame(p1 - p0, up)
    s1 = section0 if section1 is None else section1
    rings = []
    for p, (w, h) in ((p0, section0), (p1, s1)):
        q = chamfered_rect(w, h, chamfer * min(w, h))
        rings.append(p + q[:, :1] * v + q[:, 1:] * u)
    return loft(rings)


def ladder_rail(p0, p1, width: float, rail=(0.03, 0.04), rungs: int = 4, side=(0, 1, 0), rung=(0.02, 0.02)) -> Mesh:
    """Ladder / guide-rail frame: two parallel box rails ``width`` apart (centre to centre) along ``side``, from
    ``p0`` to ``p1``, tied by ``rungs`` cross members (deployment rails, linear stages, truss frames)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s = np.asarray(side, float) - np.dot(side, p1 - p0) * (p1 - p0) / np.dot(p1 - p0, p1 - p0)
    s = s / np.linalg.norm(s) * width / 2
    up = np.cross(p1 - p0, s)
    parts = [box_beam(p0 + k * s, p1 + k * s, rail, up=up) for k in (-1, 1)]
    for i in range(rungs):
        f = (i + 0.5) / rungs
        c = p0 + f * (p1 - p0)
        parts.append(box_beam(c - s, c + s, rung, up=up))
    return merge(parts)


def helix_sweep(origin, axis, r_in: float, r_out: float, thickness: float, pitch: float, length: float,
                n_per_turn: int = 24, phase: float = 0.0) -> Mesh:
    """Helical flight (auger flute, screw thread): a rectangular section from ``r_in`` to ``r_out`` and
    ``thickness`` along the axis swept along a helix of ``pitch`` for ``length`` from ``origin`` along ``axis``."""
    o = np.asarray(origin, float)
    a, u, w = _frame(axis, (1, 0, 0) if abs(np.asarray(axis, float)[0]) < 0.9 else (0, 1, 0))
    turns = length / pitch
    m = max(2, int(math.ceil(turns * n_per_turn)) + 1)
    rings = []
    for k in range(m):
        s = k / (m - 1)
        phi = phase + 2 * math.pi * turns * s
        rad = math.cos(phi) * u + math.sin(phi) * w
        c = o + s * length * a
        rings.append([c + r_in * rad - thickness / 2 * a, c + r_out * rad - thickness / 2 * a,
                      c + r_out * rad + thickness / 2 * a, c + r_in * rad + thickness / 2 * a])
    return loft(rings)


def mli_blanket(width: float, height: float, thickness: float = 0.012, wrinkle: float = 0.004, n=(28, 14),
                seed: int = 0) -> Mesh:
    """Multi-layer-insulation blanket in the x-z plane centred at the origin: a slab whose outer face (-y, like
    :func:`solar_panel`) is crumpled by ``wrinkle`` [m] of smooth random relief, the back face flat."""
    rng = np.random.default_rng(seed)
    nx, nz = n
    x = np.linspace(-width / 2, width / 2, nx)
    z = np.linspace(-height / 2, height / 2, nz)
    modes = [(rng.uniform(2, 9) / width, rng.uniform(1, 6) / height, rng.uniform(0, 2 * math.pi), rng.uniform(.3, 1))
             for _ in range(6)]
    rings = []
    for zk in z:
        relief = sum(a * np.sin(2 * math.pi * (fx * x + fz * zk) + ph) for fx, fz, ph, a in modes)
        relief = wrinkle * (0.5 + 0.5 * relief / sum(m[3] for m in modes))
        edge = np.minimum(1.0, np.minimum(np.abs(x - x[0]), np.abs(x - x[-1])) / (0.04 * width + 1e-9))
        edge *= min(1.0, min(abs(zk - z[0]), abs(zk - z[-1])) / (0.04 * height + 1e-9))
        front = np.c_[x, -thickness / 2 - relief * edge, np.full(nx, zk)]
        back = np.c_[x[::-1], np.full(nx, thickness / 2), np.full(nx, zk)]
        rings.append(np.vstack([front, back]))
    return loft(rings, cap="strip")


def rotate(mesh: Mesh, R, about=(0, 0, 0)) -> Mesh:
    V, F = mesh
    c = np.asarray(about, float)
    return (V - c) @ np.asarray(R).T + c, F


def translate(mesh: Mesh, t) -> Mesh:
    return mesh[0] + np.asarray(t, float), mesh[1]


def rot(axis, angle_rad):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * K @ K


def merge(meshes: Sequence[Mesh]) -> Mesh:
    vs, fs, off = [], [], 0
    for V, F in meshes:
        vs.append(np.asarray(V, float))
        fs.append(np.asarray(F) + off)
        off += len(V)
    return np.vstack(vs), np.vstack(fs)


# ---------------------------------------------------------------------- mass properties
def mesh_mass_properties(V, F, mass: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre of mass and inertia tensor about it [kg m^2] of a closed mesh of uniform density holding ``mass``,
    and its volume [m^3] (Eberly, "Polyhedral mass properties", 2002)."""
    V, F = np.asarray(V, float), np.asarray(F)
    t = V[F]
    x0, x1, x2 = t[:, 0], t[:, 1], t[:, 2]
    d = np.cross(x1 - x0, x2 - x0)
    f1 = x0 + x1 + x2
    f2 = x0 * x0 + x1 * (x0 + x1) + x2 * f1
    f3 = x0 ** 3 + x1 * (x0 * x0 + x0 * x1 + x1 * x1) + x2 * f2
    g0 = f2 + x0 * (f1 + x0)
    g1 = f2 + x1 * (f1 + x1)
    g2 = f2 + x2 * (f1 + x2)
    vol = (d[:, 0] * f1[:, 0]).sum() / 6.0
    first = (d * f2).sum(0) / 24.0
    second = (d * f3).sum(0) / 60.0
    yx = (d[:, 0] * (x0[:, 1] * g0[:, 0] + x1[:, 1] * g1[:, 0] + x2[:, 1] * g2[:, 0])).sum() / 120.0
    zy = (d[:, 1] * (x0[:, 2] * g0[:, 1] + x1[:, 2] * g1[:, 1] + x2[:, 2] * g2[:, 1])).sum() / 120.0
    xz = (d[:, 2] * (x0[:, 0] * g0[:, 2] + x1[:, 0] * g1[:, 2] + x2[:, 0] * g2[:, 2])).sum() / 120.0
    if abs(vol) < 1e-15:
        c = V.mean(0)
        return c, np.zeros((3, 3)), 0.0
    sgn = 1.0 if vol > 0 else -1.0
    vol, first, second, yx, zy, xz = (sgn * v for v in (vol, first, second, yx, zy, xz))
    c = first / vol
    rho = mass / vol
    Ixx = rho * (second[1] + second[2])
    Iyy = rho * (second[2] + second[0])
    Izz = rho * (second[0] + second[1])
    Ixy, Iyz, Ixz = -rho * yx, -rho * zy, -rho * xz
    Io = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    Ic = Io - mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c))       # parallel axis back to the centroid
    return c, Ic, vol


def combine_mass(items) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass, centre of mass and inertia about it of (mass, com, inertia about own com) items."""
    m = sum(i[0] for i in items)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = sum(i[0] * np.asarray(i[1]) for i in items) / m
    inertia = np.zeros((3, 3))
    for mi, ci, Ii in items:
        r = np.asarray(ci) - c
        inertia += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return float(m), c, inertia


# ---------------------------------------------------------------------- assemblies
def rover_wheel(radius: float, width: float, *, rim_thickness: float = 0.004, grousers: int = 0,
                grouser_height: float = 0.0, grouser_thickness: float = 0.004, grouser_shape: str = "straight",
                chevron_angle_deg: float = 20.0, spokes: int = 8, spoke_width: float = 0.03, hub_radius: float = 0.07,
                hub_length: float | None = None, n: int = 72, inner_side: int = -1):
    """A rigid rover wheel in its own frame (axle along y, centre at the origin): rim shell, grousers (straight lugs
    or chevrons), a spoked web on the inner side, hub (drive actuator housing). Returns parts
    ``(suffix, mesh, material, mass_share)``; mass shares are by volume x material density ratio."""
    parts = []
    rim = tube_shell((0, -width / 2, 0), (0, width / 2, 0), radius - rim_thickness, radius, n)
    parts.append(("rim", rim, "wheel_metal", 0.0))
    for k in range(grousers):
        a = 2 * math.pi * k / grousers
        Rk = rot((0, 1, 0), a)
        if grouser_shape == "chevron":
            half = width / 2
            sweep = math.tan(math.radians(chevron_angle_deg)) * half
            L = math.hypot(half, sweep)
            for side in (-1, 1):
                ang = side * math.atan2(sweep, half)
                g = box((grouser_thickness, L, grouser_height), (0, 0, radius + grouser_height / 2 - 0.0005),
                        rot((0, 0, 1), ang))
                g = translate(g, (-sweep / 2, side * half / 2, 0))
                parts.append((f"grouser{k}_{side}", rotate(g, Rk), "wheel_metal", 0.0))
        else:
            g = box((grouser_thickness, width, grouser_height), (0, 0, radius + grouser_height / 2 - 0.0005))
            parts.append((f"grouser{k}", rotate(g, Rk), "wheel_metal", 0.0))
    y_web = inner_side * (width / 2 - 0.01)
    web = []
    for k in range(spokes):
        a = 2 * math.pi * (k + 0.5) / spokes
        r_mid = 0.5 * (hub_radius + radius - rim_thickness)
        sp = box((spoke_width, 0.006, radius - rim_thickness - hub_radius * 0.8), (0, y_web, r_mid))
        web.append(rotate(sp, rot((0, 1, 0), a)))
    parts.append(("spokes", merge(web), "wheel_metal", 0.0))
    ring = tube_shell((0, y_web - 0.004, 0), (0, y_web + 0.004, 0), radius - rim_thickness - 0.012,
                      radius - rim_thickness + 1e-4, n)
    parts.append(("web_ring", ring, "wheel_metal", 0.0))
    hl = hub_length if hub_length is not None else 0.75 * width
    parts.append(("hub", cylinder((0, inner_side * hl * 0.9, 0), (0, -inner_side * 0.1 * hl, 0), hub_radius, 32),
                  "anodized_black", 0.0))
    parts.append(("hub_cap", cylinder((0, -inner_side * 0.1 * hl, 0), (0, -inner_side * 0.14 * hl, 0),
                                      hub_radius * 0.6, 24), "aluminium", 0.0))
    return _shares(parts)


def solar_panel(width: float, height: float, thickness: float = 0.025, cells=(6, 4), gap: float = 0.012,
                frame: float = 0.03):
    """Vertical solar panel in the x-z plane (facing -y... rotate as needed), centred at the origin: aluminium frame
    and substrate plus a grid of cells standing proud by 1 mm."""
    parts = [("substrate", box((width, thickness, height), (0, 0, 0)), "anodized_black", 0.0)]
    nx, nz = cells
    cw = (width - 2 * frame - (nx - 1) * gap) / nx
    ch = (height - 2 * frame - (nz - 1) * gap) / nz
    cs = []
    for i in range(nx):
        for j in range(nz):
            x = -width / 2 + frame + cw / 2 + i * (cw + gap)
            z = -height / 2 + frame + ch / 2 + j * (ch + gap)
            cs.append(box((cw, 0.002, ch), (x, -thickness / 2 - 0.0008, z)))
    parts.append(("cells", merge(cs), "solar_cell", 0.0))
    parts.append(("frame_top", box((width, thickness * 1.2, frame), (0, 0, height / 2 - frame / 2)), "aluminium", 0.0))
    parts.append(("frame_bot", box((width, thickness * 1.2, frame), (0, 0, -height / 2 + frame / 2)), "aluminium",
                  0.0))
    return _shares(parts)


def radiator(width: float, depth: float, thickness: float = 0.02, fins: int = 12, fin_height: float = 0.03):
    """Horizontal radiator plate (x-y plane, top at z = 0) with ``fins`` parallel fins along x."""
    parts = [("plate", box((width, depth, thickness), (0, 0, -thickness / 2)), "radiator", 0.0)]
    fs = [box((width * 0.96, 0.004, fin_height), (0, -depth / 2 + (k + 0.5) * depth / fins, fin_height / 2))
          for k in range(fins)]
    parts.append(("fins", merge(fs), "radiator", 0.0))
    return _shares(parts)


def camera_head(width: float, depth: float, height: float, baseline: float, lens_radius: float = 0.022,
                lights: int = 2):
    """Mast camera head facing +x: housing, a stereo pair of lenses ``baseline`` apart, LED light bars."""
    parts = [("housing", beveled_box((depth, width, height), (0, 0, 0), 0.015), "white_paint", 0.0)]
    for s in (-1, 1):
        y = s * baseline / 2
        parts.append((f"lens_barrel_{'L' if s > 0 else 'R'}", cylinder((depth / 2 - 0.005, y, 0.01),
                                                                     (depth / 2 + 0.03, y, 0.01), lens_radius * 1.35,
                                                                     24), "anodized_black", 0.0))
        parts.append((f"lens_{'L' if s > 0 else 'R'}", cylinder((depth / 2 + 0.03, y, 0.01),
                                                              (depth / 2 + 0.034, y, 0.01), lens_radius, 24),
                      "lens", 0.0))
    for k in range(lights):
        y = (k - (lights - 1) / 2) * width * 0.55
        parts.append((f"light{k}", box((0.012, width * 0.22, 0.03), (depth / 2 + 0.006, y, -height / 2 + 0.03)),
                      "led", 0.0))
    return _shares(parts)


def actuator(p0, p1, radius: float, cap_radius: float | None = None, cap_length: float | None = None,
             flange: float = 0.012, housing: str = "white_paint", trim: str = "anodized_black"):
    """Rotary actuator in assembly coordinates: a cylindrical housing from ``p0`` to ``p1`` (output end at ``p0``),
    flange rings at both ends and a motor / gearbox cap beyond ``p1`` (BLDC motor plus strain-wave gear class
    joints: suspension, steering, mast pan)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    cr = 0.62 * radius if cap_radius is None else cap_radius
    cl = 0.45 * radius if cap_length is None else cap_length
    parts = [("housing", cylinder(p0, p1, radius, 32), housing, 0.0),
             ("flange0", cylinder(p0 - 0.002 * a, p0 + flange * a, radius * 1.12, 32), trim, 0.0),
             ("flange1", cylinder(p1 - flange * a, p1 + 0.002 * a, radius * 1.12, 32), trim, 0.0),
             ("cap", cylinder(p1, p1 + cl * a, (cr, 0.8 * cr), 28), trim, 0.0)]
    return _shares(parts)


def pan_tilt_head(width: float = 0.36, depth: float = 0.15, height: float = 0.12, baseline: float = 0.24,
                  lens_radius: float = 0.02, lights: int = 6, aft_camera: bool = True, pan_height: float = 0.06):
    """Mast head on a pan-tilt unit, in its own frame (pan axis = z through the origin, looking +x): pan drum, a
    U-yoke carrying the tilt axle, a camera box with a stereo pair of lenses ``baseline`` apart, a row of
    ``lights`` LEDs under them and an optional aft-looking camera. The top of the head is at
    ``pan_height + 0.02 + height``."""
    zt = pan_height + 0.02 + height / 2                      # tilt axis height
    yo = width / 2 + 0.018
    parts = [("pan", cylinder((0, 0, 0), (0, 0, pan_height), 0.055, 32), "anodized_black", 0.0),
             ("yoke_base", chamfered_box((0.07, 2 * yo + 0.02, 0.02), (0, 0, pan_height + 0.01), 0.01),
              "white_paint", 0.0)]
    for s in (-1, 1):
        parts.append((f"yoke_{'L' if s > 0 else 'R'}", chamfered_box((0.07, 0.014, zt - pan_height + 0.035),
                                                                     (0, s * yo, 0.5 * (pan_height + zt + 0.035)),
                                                                     0.02), "white_paint", 0.0))
    parts.append(("tilt_axle", cylinder((0, -yo - 0.02, zt), (0, yo + 0.02, zt), 0.022, 24), "anodized_black", 0.0))
    parts.append(("housing", chamfered_box((depth, width, height), (0, 0, zt), 0.02, 0.012, 0.012), "white_paint",
                  0.0))
    zc = zt + 0.15 * height
    for s in (-1, 1):
        y = s * baseline / 2
        tag = "L" if s > 0 else "R"
        parts.append((f"navcam_barrel_{tag}", cylinder((depth / 2 - 0.004, y, zc), (depth / 2 + 0.028, y, zc),
                                                       lens_radius * 1.35, 24), "anodized_black", 0.0))
        parts.append((f"navcam_lens_{tag}", cylinder((depth / 2 + 0.028, y, zc), (depth / 2 + 0.032, y, zc),
                                                     lens_radius, 24), "lens", 0.0))
    if aft_camera:
        parts.append(("aftcam_barrel", cylinder((-depth / 2 + 0.004, 0, zc), (-depth / 2 - 0.024, 0, zc),
                                                lens_radius * 1.25, 24), "anodized_black", 0.0))
        parts.append(("aftcam_lens", cylinder((-depth / 2 - 0.024, 0, zc), (-depth / 2 - 0.028, 0, zc),
                                              lens_radius * 0.9, 24), "lens", 0.0))
    if lights:
        pitch = 0.8 * width / max(lights, 1)
        parts.append(("light_bar", chamfered_box((0.012, 0.86 * width, 0.028), (depth / 2 + 0.006, 0,
                                                                                 zt - 0.27 * height), 0.004),
                      "anodized_black", 0.0))
        leds = [box((0.006, 0.5 * pitch, 0.016), (depth / 2 + 0.014, (k - (lights - 1) / 2) * pitch,
                                                    zt - 0.27 * height)) for k in range(lights)]
        parts.append(("leds", merge(leds), "led", 0.0))
    return _shares(parts)


def drill_assembly(rail_height: float = 1.30, rail_width: float = 0.16, rail_offset: float = 0.075,
                   auger_radius: float = 0.0125, flight_radius: float = 0.021, pitch: float = 0.035,
                   head=(0.15, 0.15, 0.22), carriage: float = 1.0, footpad_radius: float = 0.06):
    """Rotary-percussive planetary drill with a deployment rail, in its own frame: drill axis = z through the origin,
    footpad bottom at z = 0, the rail frame (and the mount face) on the -x side. Parts: a ladder guide rail of
    ``rail_height`` with a backing deployment stage, the drill head (rotary-percussive housing and motor) riding on
    a carriage, the auger stem with its helical flight and the bit, a cuttings brush station and chute near the
    bottom, and the footpad ring. ``carriage`` in [0, 1] places the head along its feed stroke (1 = stowed at the
    top, 0 = fully fed: the feed stroke is ``rail_height - head height - 0.10``; the travel of the deployment
    stage, which lowers the rail and footpad to the ground, is not modelled)."""
    hx, hy, hz = head
    zr0, zr1 = 0.03, rail_height
    xr = -rail_offset
    z_head_top_stowed = rail_height - 0.02
    stroke = rail_height - hz - 0.10
    dz = (1.0 - carriage) * stroke
    zh1 = z_head_top_stowed - dz
    zh0 = zh1 - hz
    auger_len = z_head_top_stowed - hz - 0.012           # stowed: bit tip just above the bottom of the footpad
    zb = zh0 - auger_len                                  # bit tip
    bit_len = 0.045
    parts = [
        ("rail", ladder_rail((xr, 0, zr0), (xr, 0, zr1), rail_width, rail=(0.03, 0.035), rungs=6), "aluminium", 0.0),
        ("deploy_stage", chamfered_box((0.05, rail_width + 0.04, 0.92 * rail_height),
                                       (xr - 0.04, 0, 0.5 * (zr0 + zr1)), 0.015), "white_paint", 0.0),
        ("deploy_motor", cylinder((xr - 0.035, 0, zr1 - 0.01), (xr - 0.035, 0, zr1 + 0.07), 0.035, 24),
         "anodized_black", 0.0),
        ("carriage", chamfered_box((rail_offset + 0.02, rail_width + 0.05, 0.09), (xr / 2, 0, zh0 + 0.6 * hz), 0.01),
         "anodized_black", 0.0),
        ("head", chamfered_box((hx, hy, hz), (0, 0, 0.5 * (zh0 + zh1)), 0.022, 0.015, 0.01), "white_paint", 0.0),
        ("rotary_motor", cylinder((0, 0, zh1), (0, 0, zh1 + 0.06), (0.045, 0.04), 28), "anodized_black", 0.0),
        ("percussor", cylinder((0.0, 0.0, zh0 - 0.035), (0, 0, zh0 + 0.002), (0.028, 0.04), 24), "aluminium", 0.0),
        ("stem", cylinder((0, 0, zb + bit_len), (0, 0, zh0 - 0.03), auger_radius, 20), "steel", 0.0),
        ("flight", helix_sweep((0, 0, zb + bit_len + 0.01), (0, 0, 1), auger_radius * 0.8, flight_radius, 0.004,
                               pitch, auger_len - bit_len - 0.06, 16), "steel", 0.0),
        ("bit", cylinder((0, 0, zb + bit_len), (0, 0, zb), (flight_radius * 1.05, 0.004), 20), "steel", 0.0),
        # cuttings brush (a rotating brush sweeps the flutes) and the chute that drops the cuttings to the ground
        ("brush_housing", chamfered_box((0.09, 0.075, 0.10), (0.0, -(flight_radius + 0.045), 0.16), 0.012),
         "white_paint", 0.0),
        ("brush", cylinder((-0.035, -flight_radius - 0.012, 0.16), (0.035, -flight_radius - 0.012, 0.16), 0.022, 20),
         "anodized_black", 0.0),
        ("chute", box_beam((0.02, -(flight_radius + 0.05), 0.11), (0.10, -(flight_radius + 0.11), 0.03),
                           (0.05, 0.012), up=(0, 0, 1)), "aluminium", 0.0),
        ("footpad", tube_shell((0, 0, 0), (0, 0, 0.025), footpad_radius - 0.022, footpad_radius, 32), "aluminium",
         0.0),
    ]
    for s in (-1, 1):
        parts.append((f"footpad_strut_{'L' if s > 0 else 'R'}",
                      box_beam((xr, s * rail_width / 2, zr0 + 0.01), (-0.6 * footpad_radius, s * 0.03, 0.012),
                               (0.018, 0.018)), "aluminium", 0.0))
    return _shares(parts)


_DENSITY = {"wheel_metal": 2700.0, "aluminium": 2700.0, "anodized_black": 2700.0, "solar_cell": 2300.0,
            "radiator": 2700.0, "white_paint": 1500.0, "lens": 2500.0, "led": 1500.0, "steel": 7800.0,
            "mli_gold": 300.0, "mli_silver": 300.0}


def _shares(parts):
    vols = []
    for _name, (V, F), mat, _ in parts:
        vols.append(abs(signed_volume(V, F)) * _DENSITY.get(mat, 1500.0))
    tot = sum(vols) or 1.0
    return [(n, m, mat, v / tot) for (n, m, mat, _), v in zip(parts, vols, strict=True)]


# ---------------------------------------------------------------------- the assembly
@dataclass
class CADPart:
    name: str
    body: str
    vertices: np.ndarray
    faces: np.ndarray
    material: str
    mass: float = 0.0
    group: str = ""

    def mass_properties(self):
        return mesh_mass_properties(self.vertices, self.faces, self.mass)


@dataclass
class VehicleCAD:
    """Parts grouped into bodies; every coordinate is in one assembly frame (the vehicle at its reference pose)."""
    title: str = "vehicle"
    parts: list[CADPart] = field(default_factory=list)
    frames: dict = field(default_factory=dict)        # body -> (R, p) of its joint frame in the assembly frame
    notes: dict = field(default_factory=dict)

    def add(self, name: str, body: str, mesh: Mesh, material: str, mass: float = 0.0, group: str = "") -> CADPart:
        p = CADPart(name, body, np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64), material, float(mass),
                    group or body)
        self.parts.append(p)
        return p

    def add_assembly(self, prefix: str, body: str, parts, mass: float, R=None, t=(0, 0, 0), group: str = ""):
        """Add an assembly (list of (suffix, mesh, material, mass_share)) rotated by R, moved by t, with ``mass``
        split by the shares."""
        R = np.eye(3) if R is None else np.asarray(R)
        for suffix, (V, F), mat, share in parts:
            self.add(f"{prefix}_{suffix}", body, (V @ R.T + np.asarray(t, float), F), mat, mass * share, group)

    def bodies(self) -> list[str]:
        return list(dict.fromkeys(p.body for p in self.parts))

    def body_parts(self, body: str) -> list[CADPart]:
        return [p for p in self.parts if p.body == body]

    def mass_properties(self) -> dict:
        """Per body: ``mass`` [kg], ``com`` [m] and ``inertia`` about the com [kg m^2], in the assembly frame."""
        out = {}
        for b in self.bodies():
            items = []
            for p in self.body_parts(b):
                if p.mass > 0:
                    c, inertia, _ = p.mass_properties()
                    items.append((p.mass, c, inertia))
            m, c, inertia = combine_mass(items)
            out[b] = {"mass": m, "com": c, "inertia": inertia}
        return out

    def total(self) -> dict:
        mp = self.mass_properties()
        m, c, inertia = combine_mass([(v["mass"], v["com"], v["inertia"]) for v in mp.values() if v["mass"] > 0])
        return {"mass": m, "com": c, "inertia": inertia}

    def body_mesh(self, body: str) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
        return [(p.name, p.vertices, p.faces, p.material) for p in self.body_parts(body)]

    def posed(self, transforms: dict) -> list[CADPart]:
        """Parts moved rigidly: ``transforms[body] = (R, t)`` maps assembly coordinates of that body to the world."""
        out = []
        for p in self.parts:
            R, t = transforms.get(p.body, (np.eye(3), np.zeros(3)))
            out.append(CADPart(p.name, p.body, p.vertices @ np.asarray(R).T + t, p.faces, p.material, p.mass,
                               p.group))
        return out

    def scene(self, transforms: dict | None = None, title: str | None = None):
        from pinneapple_tools.visualization.studio.scene import Scene, Surface
        parts = self.posed(transforms) if transforms else self.parts
        return Scene([Surface(p.name, p.vertices, p.faces, p.material, group=p.group) for p in parts],
                     title=title or self.title)

    def export(self, path: str, transforms: dict | None = None) -> str:
        """glTF (.glb, PBR materials), STL or USD by extension."""
        return self.scene(transforms).save(path)

    def bounds(self):
        V = np.concatenate([p.vertices for p in self.parts])
        return V.min(0), V.max(0)


__all__ = ["VehicleCAD", "CADPart", "box", "beveled_box", "chamfered_box", "chamfered_rect", "extrude_polygon",
           "offset_polygon", "loft", "box_beam", "ladder_rail", "helix_sweep", "mli_blanket", "cylinder",
           "tube_shell", "revolve_closed", "paraboloid_dish", "rover_wheel", "solar_panel", "radiator", "camera_head",
           "pan_tilt_head", "actuator", "drill_assembly", "mesh_mass_properties", "combine_mass", "rot", "rotate",
           "translate", "merge", "signed_volume", "orient_outward"]
