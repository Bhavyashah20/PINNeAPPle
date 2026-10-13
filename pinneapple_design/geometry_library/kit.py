"""Closed-surface building blocks for the library entries (z up, metres, outward normals).

Every function returns ``(V, F)``; every surface is closed (watertight) unless the name says otherwise:
lofts through rings with capped or pole ends, solids of revolution, NACA sections and lofted wings / blades,
rounded boxes, cylinders, frusta, ellipsoids, swept tubes, extruded polygons (ear-clipped caps), rotations.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .meshops import concat, orient_outward, volume

Mesh = tuple[np.ndarray, np.ndarray]


# ============================================================================================ transforms
def rot(axis, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def tf(mesh: Mesh, R=None, t=(0, 0, 0), about=(0, 0, 0), scale=1.0) -> Mesh:
    V, F = mesh
    V = (np.asarray(V, float) - about) * scale
    if R is not None:
        V = V @ np.asarray(R).T
    return V + np.asarray(about, float) + np.asarray(t, float), F


def mirror(mesh: Mesh, axis: int = 1) -> Mesh:
    V, F = mesh
    V = np.asarray(V, float).copy()
    V[:, axis] *= -1
    return V, F[:, ::-1].copy()


def with_mirror(mesh: Mesh, axis: int = 1) -> list[Mesh]:
    return [mesh, mirror(mesh, axis)]


def merge(meshes: Iterable[Mesh]) -> Mesh:
    return concat(list(meshes))


# ============================================================================================ lofts
def _grid(nr: int, n: int) -> np.ndarray:
    i = np.arange(nr - 1)[:, None]
    j = np.arange(n)[None, :]
    a = i * n + j
    b = i * n + (j + 1) % n
    f1 = np.stack([a, b, a + n], -1).reshape(-1, 3)
    f2 = np.stack([b, b + n, a + n], -1).reshape(-1, 3)
    return np.concatenate([f1, f2])


def _plane_coords(P: np.ndarray):
    c = P.mean(0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    return (P - c) @ vt[:2].T, float(sv[-1] / max(sv[0], 1e-300))


def ear_clip_fast(P2: np.ndarray) -> np.ndarray:
    """Ear clipping of a simple polygon (counter-clockwise), vectorised point-in-triangle tests; (n - 2, 3)."""
    idx = list(range(len(P2)))
    tris = []
    def cr(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    guard = 0
    while len(idx) > 3 and guard < 4 * len(P2):
        guard += 1
        I = np.asarray(idx)  # noqa: E741
        A, B, C = P2[np.roll(I, 1)], P2[I], P2[np.roll(I, -1)]
        convex = cr(A, B, C) > 1e-14 * max(float(np.abs(P2).max()) ** 2, 1e-300)
        reflex = I[~convex]
        clipped = False
        for k in np.where(convex)[0]:
            a, b, c = A[k], B[k], C[k]
            others = reflex[(reflex != I[k - 1]) & (reflex != I[(k + 1) % len(I)])]
            if len(others):
                Q = P2[others]
                inside = (cr(a, b, Q) >= 0) & (cr(b, c, Q) >= 0) & (cr(c, a, Q) >= 0)
                if inside.any():
                    continue
            tris.append((int(I[k - 1]), int(I[k]), int(I[(k + 1) % len(I)])))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            raise ValueError("ear clipping failed (polygon not simple)")
    tris.append(tuple(idx))
    return np.asarray(tris, np.int64)


def _cap(ring: np.ndarray, base: int, centre: int, flip: bool) -> np.ndarray:
    """Triangles closing a ring: a fan to the centre vertex when the ring is star-shaped from its centroid (the
    usual case: circles, ellipses, convex sections), otherwise an ear-clipped planar triangulation (concave caps
    such as cambered airfoils, where a fan would fold over itself)."""
    n = len(ring)
    j = np.arange(n)
    fan = np.stack([np.full(n, centre), base + (j + 1) % n, base + j], -1)
    P2, flat = _plane_coords(ring)
    Q2 = np.roll(P2, -1, axis=0)
    c = P2[:, 0] * Q2[:, 1] - P2[:, 1] * Q2[:, 0]
    star = (c > 0).all() or (c < 0).all()
    if not star and flat < 1e-6:
        try:
            ccw = c.sum() > 0
            T = ear_clip_fast(P2 if ccw else P2[::-1])
            if not ccw:
                T = (n - 1) - T
                T = T[:, ::-1]
            # same orientation convention as the fan (ring order reversed for the start cap)
            T = T[:, ::-1] + base
            fan = T
        except ValueError:
            pass
    return fan[:, ::-1] if flip else fan


def loft(rings, start: str = "cap", end: str = "cap") -> Mesh:
    """Closed surface through closed rings ``(n_rings, n, 3)``. ``start`` / ``end``: "cap" (flat end: a fan to the
    centroid, or an ear-clipped triangulation for a concave ring) or "pole" (the end ring is joined to its centroid:
    a pointed / rounded tip, the centroid can be moved afterwards)."""
    R = np.asarray(rings, float)
    nr, n, _ = R.shape
    V = [R.reshape(-1, 3)]
    F = [_grid(nr, n)]
    off = nr * n
    for which, mode in ((0, start), (nr - 1, end)):
        V.append(R[which].mean(0)[None])
        if mode == "pole":
            j = np.arange(n)
            fan = np.stack([np.full(n, off), which * n + (j + 1) % n, which * n + j], -1)
            F.append(fan if which == 0 else fan[:, ::-1])
        else:
            F.append(_cap(R[which], which * n, off, flip=which != 0))
        off += 1
    V, F = np.vstack(V), np.vstack(F)
    V, F = orient_outward(V, F)
    return V, F


def revolve(x, r, n: int = 48, axis: str = "x", center=(0.0, 0.0, 0.0), phase: float = 0.0) -> Mesh:
    """Solid of revolution of the meridian ``r(x)`` about an axis through ``center``. End stations with r = 0
    close as poles, others are capped."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    keep = slice(1 if r[0] <= 1e-12 else 0, len(r) - (1 if r[-1] <= 1e-12 else 0))
    xs, rs = x[keep], np.maximum(r[keep], 0)
    ring = np.stack([np.repeat(xs[:, None], n, 1), rs[:, None] * np.cos(th), rs[:, None] * np.sin(th)], -1)
    V, F = loft(ring)
    # replace caps by poles when the meridian reaches the axis
    if r[0] <= 1e-12:
        V[len(xs) * n] = (x[0], 0, 0)
    if r[-1] <= 1e-12:
        V[len(xs) * n + 1] = (x[-1], 0, 0)
    perm = {"x": [0, 1, 2], "y": [2, 0, 1], "z": [1, 2, 0]}[axis]
    V = V[:, perm] + np.asarray(center, float)
    return orient_outward(V, F)


def superellipse(hw: float, hh: float, n: int, e: float = 2.0, zc: float = 0.0, yc: float = 0.0,
                 top_e: float | None = None) -> np.ndarray:
    """Ring of ``n`` (y, z) points: |y/hw|^e + |z/hh|^e = 1 (``top_e``: another exponent for the upper half)."""
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    ee = np.where(s > 0, top_e if top_e else e, e)
    y = hw * np.sign(c) * np.abs(c) ** (2 / ee)
    z = hh * np.sign(s) * np.abs(s) ** (2 / ee)
    return np.stack([y + yc, z + zc], 1)


def body_of_sections(xs, hw, zb, zt, n: int = 48, e=2.4, e_top=None) -> Mesh:
    """Fuselage-like body: at station x, a super-ellipse of half-width hw(x) between z = zb(x) and z = zt(x)."""
    rings = []
    for k, x in enumerate(xs):
        ee = e[k] if np.ndim(e) else e
        et = (e_top[k] if np.ndim(e_top) else e_top) if e_top is not None else None
        yz = superellipse(max(hw[k], 1e-6), max(0.5 * (zt[k] - zb[k]), 1e-6), n, ee, 0.5 * (zt[k] + zb[k]),
                          top_e=et)
        rings.append(np.c_[np.full(n, x), yz])
    return loft(rings)


# ============================================================================================ sections
def naca4(m: float = 0.0, p: float = 0.4, t: float = 0.12, n: int = 40, closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit outline, ``(2n - 2, 2)`` points, counter-clockwise from the trailing edge over the upper surface
    (x in [0, 1] from the leading edge, y up). ``closed_te`` uses the -0.1036 coefficient (sharp trailing edge)."""
    b = np.linspace(0, math.pi, n)
    x = 0.5 * (1 - np.cos(b))
    k4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + k4 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dy = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dy)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    up = np.stack([xu, yu], 1)[::-1]          # TE -> LE over the top
    lo = np.stack([xl, yl], 1)[1:-1]          # LE -> TE underneath (skip the shared LE and TE)
    return np.vstack([up, lo])


def thick_te(sec: np.ndarray, te: float) -> np.ndarray:
    """Open the trailing edge to a finite thickness ``te`` (fraction of chord), linearly from the leading edge."""
    s = sec.copy()
    n = len(s)
    half = (n + 2) // 2
    upper = np.arange(n) < half - 1
    s[:, 1] += np.where(upper, 0.5, -0.5) * te * s[:, 0]
    return s


def section_ring(sec2d: np.ndarray, chord: float, twist_deg: float, le, frame=None, pivot: float = 0.25) -> np.ndarray:
    """Place a unit-chord section in 3D: chord along ``frame[0]``, thickness along ``frame[1]``, twisted by
    ``twist_deg`` about ``pivot`` x chord (nose-up positive), leading edge at ``le``."""
    e1, e2 = (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])) if frame is None else (np.asarray(frame[0], float),
                                                                                   np.asarray(frame[1], float))
    u = (sec2d[:, 0] - pivot) * chord
    v = sec2d[:, 1] * chord
    a = math.radians(twist_deg)
    ur = u * math.cos(a) + v * math.sin(a)
    vr = -u * math.sin(a) + v * math.cos(a)
    return np.asarray(le, float) + (ur + pivot * chord)[:, None] * e1 + vr[:, None] * e2


def wing(stations: Sequence[dict], n: int = 36, frame=None, tip: str = "cap") -> Mesh:
    """Loft through wing / blade stations. Each station: ``le`` (3,), ``chord``, ``twist`` (deg), ``t`` (t/c),
    ``m`` / ``p`` (camber), optional ``section`` (a 2D outline overriding the NACA one). Closed at both ends."""
    rings = []
    for s in stations:
        sec = s.get("section")
        if sec is None:
            sec = naca4(s.get("m", 0.0), s.get("p", 0.4), s.get("t", 0.12), n)
        rings.append(section_ring(np.asarray(sec, float), s["chord"], s.get("twist", 0.0), s["le"], frame,
                                  s.get("pivot", 0.25)))
    return loft(rings, "cap", tip)


def blend_sections(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return (1 - w) * a + w * b


def round_section(n: int) -> np.ndarray:
    """A unit circle as a 'section' (x in [0, 1]) with the same point count as :func:`naca4` (``2n - 2``)."""
    m = 2 * n - 2
    th = np.linspace(0, 2 * math.pi, m, endpoint=False)
    return np.stack([0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th)], 1)


# ============================================================================================ solids
def box(size, center=(0, 0, 0), R=None) -> Mesh:
    sx, sy, sz = np.asarray(size, float) / 2
    V = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for a, b, c, d in q for t in ([a, b, c], [a, c, d])], np.int64)
    V, F = orient_outward(V, F)
    return tf((V, F), R, center)


def rounded_box(size, center=(0, 0, 0), radius: float = 0.05, n_side: int = 4, R=None) -> Mesh:
    """Box with every edge and corner rounded by ``radius`` (absolute): the Minkowski sum of a smaller box and a
    sphere, sampled octant by octant so the flat faces stay flat."""
    from .meshops import weld
    h = np.asarray(size, float) / 2
    r = float(min(radius, 0.999 * h.min()))
    core = h - r
    az, sx, sy = [], [], []
    for q, (qx, qy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        for a in np.linspace(90 * q, 90 * (q + 1), n_side + 1):
            az.append(math.radians(a))
            sx.append(qx)
            sy.append(qy)
    az, sx, sy = np.array(az), np.array(sx), np.array(sy)
    rings = []
    for hemi in (-1, 1):
        els = np.linspace(-90, 0, n_side + 1) if hemi < 0 else np.linspace(0, 90, n_side + 1)
        for e in els:
            el = math.radians(e)
            n = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full_like(az, math.sin(el))], 1)
            rings.append(r * n + np.stack([sx * core[0], sy * core[1], np.full_like(az, hemi * core[2])], 1))
    V, F = loft(rings)
    V[len(rings) * len(az)] = (0, 0, -h[2])
    V[len(rings) * len(az) + 1] = (0, 0, h[2])
    V, F = weld(V, F)
    return tf(orient_outward(V, F), R, center)


def cylinder(p0, p1, radius, n: int = 32, r1: float | None = None) -> Mesh:
    """Closed cylinder (or frustum with ``r1``) between two points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    ax = d / L
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ax, seed)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = [p + rr * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
             for p, rr in ((p0, radius), (p1, radius if r1 is None else r1))]
    return loft(rings)


def ellipsoid(center, radii, nu: int = 24, nv: int = 14, R=None) -> Mesh:
    th = np.linspace(0, math.pi, nv + 1)[1:-1]
    x = -np.cos(th)
    r = np.sin(th)
    rings = []
    ph = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    for xi, ri in zip(x, r, strict=False):
        rings.append(np.stack([np.full(nu, xi), ri * np.cos(ph), ri * np.sin(ph)], 1))
    V, F = loft(rings, "pole", "pole")
    V[-2] = (-1, 0, 0)
    V[-1] = (1, 0, 0)
    V = V * np.asarray(radii, float)
    return tf(orient_outward(V, F), R, center)


def tube(path, radius, n: int = 16, radii=None) -> Mesh:
    """Closed tube swept along a polyline with parallel-transport frames (capped ends)."""
    P = np.asarray(path, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    seed = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    nrm = np.cross(T[0], seed)
    nrm /= np.linalg.norm(nrm)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = []
    rr = np.full(len(P), radius, float) if radii is None else np.asarray(radii, float)
    for k, (p, t) in enumerate(zip(P, T, strict=False)):
        nrm = nrm - np.dot(nrm, t) * t
        nrm /= np.linalg.norm(nrm)
        b = np.cross(t, nrm)
        rings.append(p + rr[k] * (np.cos(th)[:, None] * nrm + np.sin(th)[:, None] * b))
    return loft(rings)


def tube_loop(path, radius, n: int = 16) -> Mesh:
    """Closed tube along a closed polyline (the last point joins the first; no caps): rings, hoops, halos."""
    P = np.asarray(path, float)
    m = len(P)
    T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    c = P.mean(0)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = []
    for p, t in zip(P, T, strict=False):
        u = p - c
        u = u - np.dot(u, t) * t
        u /= np.linalg.norm(u)
        w = np.cross(t, u)
        rings.append(p + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w))
    V = np.vstack(rings)
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    a = i * n + j
    b = i * n + (j + 1) % n
    a2 = ((i + 1) % m) * n + j
    b2 = ((i + 1) % m) * n + (j + 1) % n
    F = np.concatenate([np.stack([a, b, a2], -1).reshape(-1, 3), np.stack([b, b2, a2], -1).reshape(-1, 3)])
    return orient_outward(V, F)


def torus(center, R_major: float, r_minor: float, nu: int = 48, nv: int = 16, axis=(0, 0, 1)) -> Mesh:
    u = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    X = (R_major + r_minor * np.cos(Vv)) * np.cos(U)
    Y = (R_major + r_minor * np.cos(Vv)) * np.sin(U)
    Z = r_minor * np.sin(Vv)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i = np.arange(nu)[:, None]
    j = np.arange(nv)[None, :]
    a = i * nv + j
    b = ((i + 1) % nu) * nv + j
    c = ((i + 1) % nu) * nv + (j + 1) % nv
    d = i * nv + (j + 1) % nv
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    V, F = orient_outward(V, F)
    z = np.asarray(axis, float)
    z /= np.linalg.norm(z)
    if abs(z[2]) < 0.999:
        ax = np.cross([0, 0, 1.0], z)
        V = V @ rot(ax, math.degrees(math.acos(np.clip(z[2], -1, 1)))).T
    elif z[2] < 0:
        V = V * [1, 1, -1]
        F = F[:, ::-1]
    return V + np.asarray(center, float), F


def ear_clip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple polygon (counter-clockwise or not) by ear clipping; returns (k, 3) indices."""
    P = np.asarray(poly, float)
    n = len(P)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 10 * n * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-15:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = P[j]
                if cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    T = np.array(tris, np.int64)
    if area < 0:
        T = T[:, ::-1]
    return T


def extrude(poly2d, z0: float, z1: float, plane: str = "xy") -> Mesh:
    """Prism: a simple polygon in the ``plane`` extruded along the third axis from z0 to z1 (ear-clipped caps)."""
    P = np.asarray(poly2d, float)
    n = len(P)
    T = ear_clip(P)
    V = np.vstack([np.c_[P, np.full(n, z0)], np.c_[P, np.full(n, z1)]])
    j = np.arange(n)
    side = np.concatenate([np.stack([j, (j + 1) % n, j + n], 1), np.stack([(j + 1) % n, (j + 1) % n + n, j + n], 1)])
    F = np.vstack([T[:, ::-1], T + n, side])
    perm = {"xy": [0, 1, 2], "xz": [0, 2, 1], "yz": [2, 0, 1]}[plane]
    V = V[:, perm]
    return orient_outward(V, F)


def sweep_profile(profile2d, path, up=(0, 0, 1)) -> Mesh:
    """A closed 2D profile (u along the frame normal closest to ``up``, v along the binormal) swept along a path,
    capped (rails, hand-holds, ribs)."""
    P = np.asarray(path, float)
    S = np.asarray(profile2d, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.asarray(up, float)
    rings = []
    for p, t in zip(P, T, strict=False):
        nn = up - np.dot(up, t) * t
        if np.linalg.norm(nn) < 1e-9:
            nn = np.cross(t, [1.0, 0, 0])
        nn /= np.linalg.norm(nn)
        b = np.cross(t, nn)
        rings.append(p + S[:, 0, None] * nn + S[:, 1, None] * b)
    return loft(rings)


def is_closed_volume(mesh: Mesh) -> bool:
    return volume(*mesh) > 0


__all__ = ["rot", "tf", "mirror", "with_mirror", "merge", "loft", "revolve", "superellipse", "body_of_sections",
           "naca4", "thick_te", "section_ring", "wing", "round_section", "box", "rounded_box", "cylinder",
           "ellipsoid", "tube", "tube_loop", "torus", "ear_clip", "extrude", "sweep_profile", "blend_sections"]
