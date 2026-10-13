"""Mesh operations for the library: welding, orientation, watertightness, volume, self-intersection, union.

NumPy / SciPy only, except :func:`union` (``manifold3d``). Meshes are ``(V (n, 3) float, F (m, 3) int)``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]


def concat(meshes: Sequence[Mesh]) -> Mesh:
    vs, fs, off = [], [], 0
    for V, F in meshes:
        vs.append(np.asarray(V, float))
        fs.append(np.asarray(F, np.int64) + off)
        off += len(V)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(vs), np.vstack(fs)


def compact(V: np.ndarray, F: np.ndarray) -> Mesh:
    """Drop unreferenced vertices."""
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(F.shape)


def weld(V: np.ndarray, F: np.ndarray, tol: float = 1e-9) -> Mesh:
    """Merge coincident vertices (quantised to ``tol`` relative to the size), drop collapsed faces."""
    scale = max(float(np.ptp(V, axis=0).max()), 1e-12) if len(V) else 1.0
    q = np.round(V / (tol * scale)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[first], F2[ok]


def area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def volume(V: np.ndarray, F: np.ndarray) -> float:
    """Signed enclosed volume (positive for outward normals)."""
    t = V[F] - V.mean(0)
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def face_normals(V: np.ndarray, F: np.ndarray, unit: bool = True) -> np.ndarray:
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    if unit:
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    return n


def _edges(F: np.ndarray):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return e


def edge_stats(V: np.ndarray, F: np.ndarray) -> dict[str, int]:
    """Counts after welding: boundary edges (used once), non-manifold edges (> 2), edges whose two faces disagree
    on the winding."""
    Vw, Fw = weld(V, F)
    e = _edges(Fw)
    key = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    forward = (e[:, 0] < e[:, 1]).astype(np.int64)
    fwd_count = np.bincount(inv, weights=forward, minlength=len(uniq))
    two = cnt == 2
    bad_wind = int(np.sum(two & (fwd_count != 1)))
    return {"boundary_edges": int(np.sum(cnt == 1)), "nonmanifold_edges": int(np.sum(cnt > 2)),
            "misoriented_edges": bad_wind, "n_vertices": len(Vw), "n_faces": len(Fw)}


def components(F: np.ndarray, n_vertices: int) -> np.ndarray:
    """Connected-component label of every face (faces sharing a vertex)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = len(F)
    rows = np.repeat(np.arange(m), 3)
    A = coo_matrix((np.ones(3 * m), (rows, F.ravel())), shape=(m, n_vertices)).tocsr()
    G = A @ A.T
    _, lab = connected_components(G, directed=False)
    return lab


def orient_outward(V: np.ndarray, F: np.ndarray) -> Mesh:
    """Flip each closed component whose signed volume is negative (the winding inside a component is kept)."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64).copy()
    if len(F) == 0:
        return V, F
    Vw, Fw = weld(V, F)
    if len(Fw) != len(F):          # collapsed faces: orient on the original indices
        Fw = None
    lab = components(Fw if Fw is not None else F, len(Vw) if Fw is not None else len(V))
    for c in np.unique(lab):
        sel = lab == c
        if volume(V, F[sel]) < 0:
            F[sel] = F[sel][:, ::-1]
    return V, F


def quality(V: np.ndarray, F: np.ndarray, closed: bool = True) -> dict[str, float]:
    es = edge_stats(V, F)
    t = V[F]
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    scale = max(float(np.ptp(V, axis=0).max()), 1e-12)
    return {**es, "watertight": es["boundary_edges"] == 0 and es["nonmanifold_edges"] == 0
            and es["misoriented_edges"] == 0, "volume": volume(V, F) if closed else 0.0, "area": float(a.sum()),
            "degenerate_faces": int(np.sum(a < 1e-14 * scale * scale)), "finite": bool(np.isfinite(V).all())}


# ------------------------------------------------------------------------------------- self-intersection
def _tri_tri(p, q):
    """Vectorised Moller-style test: do triangles p[i] and q[i] (each (k, 3, 3)) intersect? Distances below 1e-5 of
    the pair's size count as touching, and pairs whose planes are within ~0.6 deg are not tested (coplanar or nearly
    parallel neighbours, where the intersection line is ill-conditioned)."""
    def plane(t):
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        d = -np.einsum("ij,ij->i", n, t[:, 0])
        return n, d

    c = p.mean(1, keepdims=True)                       # translation-invariant: work about the pair's centre
    p, q = p - c, q - c
    n1, d1 = plane(p)
    n2, d2 = plane(q)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-300
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-300
    d1 = -np.einsum("ij,ij->i", n1, p[:, 0])
    d2 = -np.einsum("ij,ij->i", n2, q[:, 0])
    dq = np.einsum("ikj,ij->ik", q, n1) + d1[:, None]    # signed distances [m]
    dp = np.einsum("ikj,ij->ik", p, n2) + d2[:, None]
    size = np.maximum(np.linalg.norm(p, axis=2).max(1), np.linalg.norm(q, axis=2).max(1))[:, None]
    tol = 1e-5 * size                                  # contacts / penetrations below 1e-5 of the size are touching
    dq = np.where(np.abs(dq) < tol, 0.0, dq)
    dp = np.where(np.abs(dp) < tol, 0.0, dp)
    sep = (np.all(dq > 0, 1) | np.all(dq < 0, 1) | np.all(dp > 0, 1) | np.all(dp < 0, 1))
    coplanar = np.all(dq == 0, 1)
    # a triangle that meets the other's plane only with one of its edges (two vertices on the plane, the third off
    # it) touches it along that edge: a side face meeting a flat cap at the boundary of a closed body
    edge_on = ((dq == 0).sum(1) == 2) | ((dp == 0).sum(1) == 2)
    cand = ~sep & ~coplanar & ~edge_on
    out = np.zeros(len(p), bool)
    if not cand.any():
        return out
    idx = np.where(cand)[0]
    D = np.cross(n1[idx], n2[idx])
    sin = np.linalg.norm(D, axis=1)
    par = sin < 1e-2                                   # planes within ~0.6 deg: neighbours on a flat or gently
                                                       # curved sheet whose intersection line is ill-conditioned
    idx, D, sin = idx[~par], D[~par], sin[~par]
    if len(idx) == 0:
        return out
    D = D / sin[:, None]                               # unit direction: intervals in metres

    def interval(t, d):
        proj = np.einsum("ikj,ij->ik", t, D)
        lo = np.full(len(t), np.inf)
        hi = np.full(len(t), -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = d[:, a], d[:, b]
            cross = (da * db) < 0
            tt = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
            x = proj[:, a] + tt * (proj[:, b] - proj[:, a])
            lo = np.where(cross, np.minimum(lo, x), lo)
            hi = np.where(cross, np.maximum(hi, x), hi)
            on = da == 0
            lo = np.where(on, np.minimum(lo, proj[:, a]), lo)
            hi = np.where(on, np.maximum(hi, proj[:, a]), hi)
        return lo, hi

    a0, a1 = interval(p[idx], dp[idx])
    b0, b1 = interval(q[idx], dq[idx])
    out[idx] = (np.minimum(a1, b1) - np.maximum(a0, b0)) > tol[idx, 0]
    return out


def self_intersections(V: np.ndarray, F: np.ndarray, max_pairs: int = 20_000_000, return_pairs: bool = False):
    """Number of intersecting pairs of non-adjacent triangles. Candidate pairs come from k-d trees on the triangle
    centres, bucketed by circumradius (powers of two), so a few large cap triangles do not blow up the search."""
    from scipy.spatial import cKDTree
    Vw, Fw = weld(V, F)
    t = Vw[Fw]
    c = t.mean(1)
    r = np.linalg.norm(t - c[:, None], axis=2).max(1)
    lvl = np.floor(np.log2(np.maximum(r, 1e-30) / max(float(r.max()), 1e-30))).astype(np.int64)
    lvl = np.maximum(lvl, -30)
    buckets = [np.where(lvl == k)[0] for k in np.unique(lvl)]
    trees = [(b, cKDTree(c[b]), float(r[b].max())) for b in buckets]
    I, J = [], []  # noqa: E741
    total = 0
    for a in range(len(trees)):
        for b in range(a, len(trees)):
            ia, ta, ra = trees[a]
            ib, tb, rb = trees[b]
            if a == b:
                p = ta.query_pairs(ra + rb, output_type="ndarray")
                if len(p):
                    I.append(ia[p[:, 0]])
                    J.append(ia[p[:, 1]])
            else:
                lst = ta.query_ball_tree(tb, ra + rb)
                ii = np.repeat(ia, [len(x) for x in lst])
                jj = ib[np.concatenate([np.asarray(x, np.int64) for x in lst])] if ii.size else np.zeros(0, np.int64)
                I.append(ii)
                J.append(jj)
            total += len(I[-1]) if I else 0
            if total > max_pairs:
                raise RuntimeError(f"more than {max_pairs} candidate pairs")
    if not I:
        return 0
    i, j = np.concatenate(I), np.concatenate(J)
    close = np.linalg.norm(c[i] - c[j], axis=1) <= r[i] + r[j]
    i, j = i[close], j[close]
    shared = (Fw[i][:, :, None] == Fw[j][:, None, :]).any((1, 2))      # adjacent faces share a vertex
    i, j = i[~shared], j[~shared]
    lo = np.minimum(t[i].min(1), t[j].min(1))
    ov = np.all(np.maximum(t[i].min(1), t[j].min(1)) <= np.minimum(t[i].max(1), t[j].max(1)), axis=1)
    i, j = i[ov], j[ov]
    del lo
    hit = np.concatenate([_tri_tri(t[i[k:k + 200_000]], t[j[k:k + 200_000]]) for k in range(0, len(i), 200_000)]) \
        if len(i) else np.zeros(0, bool)
    if return_pairs:
        return np.stack([i[hit], j[hit]], 1), Vw, Fw
    return int(hit.sum())


# ------------------------------------------------------------------------------------- boolean union
def union(meshes: Sequence[Mesh]) -> Mesh:
    """Boolean union of closed meshes with ``manifold3d`` (``pip install manifold3d``)."""
    try:
        import manifold3d as m3d
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError("Geometry.solid() needs manifold3d: pip install manifold3d") from e
    parts: list = []
    for V, F in meshes:
        Vw, Fw = weld(np.asarray(V, float), np.asarray(F, np.int64))
        mesh = m3d.Mesh(vert_properties=np.ascontiguousarray(Vw, np.float32),
                        tri_verts=np.ascontiguousarray(Fw, np.uint32))
        man = m3d.Manifold(mesh)
        if man.status() != m3d.Error.NoError:
            raise ValueError(f"a body is not a closed 2-manifold ({man.status()})")
        parts.append(man)
    out = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add) if len(parts) > 1 else parts[0]
    m = out.to_mesh()
    return np.asarray(m.vert_properties[:, :3], float), np.asarray(m.tri_verts, np.int64)


def crease_normals(V: np.ndarray, F: np.ndarray, angle_deg: float = 38.0) -> np.ndarray:
    """Per-corner normals ``(m, 3, 3)``: the smooth vertex normal where it is within ``angle_deg`` of the face
    normal, the face normal across creases (sharp box edges stay sharp)."""
    fn = face_normals(V, F, unit=False)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-300
    fu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-300)
    cn = vn[F]
    ok = np.einsum("ikj,ij->ik", cn, fu) > np.cos(np.radians(angle_deg))
    return np.where(ok[:, :, None], cn, fu[:, None, :])


def solidify(V: np.ndarray, F: np.ndarray, thickness: float):
    """Close an open sheet into a thin shell: the sheet, a copy offset by ``-thickness`` along the vertex normals
    with reversed winding, and side walls along the boundary edges. Returns ``(V, F, kept)`` where ``kept`` marks
    the original faces that survived welding."""
    Vw, Fw = weld(V, F)
    kept = np.ones(len(F), bool)
    if len(Fw) != len(F):                      # faces collapsed by welding are dropped
        q = np.round(V / (1e-9 * max(float(np.ptp(V, axis=0).max()), 1e-12))).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True)
        Fi = inv.ravel()[F]
        kept = (Fi[:, 0] != Fi[:, 1]) & (Fi[:, 1] != Fi[:, 2]) & (Fi[:, 0] != Fi[:, 2])
    fn = face_normals(Vw, Fw, unit=False)
    vn = np.zeros_like(Vw)
    for k in range(3):
        np.add.at(vn, Fw[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-300
    n = len(Vw)
    V2 = np.vstack([Vw, Vw - thickness * vn])
    e = _edges(Fw)
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = e[cnt[inv.ravel()] == 1]
    walls = np.concatenate([np.stack([bnd[:, 1], bnd[:, 0], bnd[:, 0] + n], 1),
                            np.stack([bnd[:, 1], bnd[:, 0] + n, bnd[:, 1] + n], 1)])
    F2 = np.vstack([Fw, Fw[:, ::-1] + n, walls])
    return V2, F2, kept


def fix_winding(V: np.ndarray, F: np.ndarray) -> Mesh:
    """Make the winding consistent across every edge-manifold component (breadth-first propagation over shared
    edges), then orient each component outward."""
    from collections import deque
    F = np.asarray(F, np.int64).copy()
    e = _edges(F)
    m = len(F)
    fid = np.tile(np.arange(m), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    nbr = [[] for _ in range(m)]
    for k in np.where(same)[0]:
        a, b = order[k], order[k + 1]
        if k + 2 < len(order) and same[k + 1]:
            continue                                    # non-manifold edge: leave it
        fa, fb = fid[a], fid[b]
        nbr[fa].append((fb, a, b))
        nbr[fb].append((fa, b, a))
    seen = np.zeros(m, bool)
    for s in range(m):
        if seen[s]:
            continue
        seen[s] = True
        dq = deque([s])
        while dq:
            f = dq.popleft()
            for g, ea, eb in nbr[f]:
                if seen[g]:
                    continue
                # the shared edge as stored for f (ea) and for g (eb): consistent if their directions are opposite
                da = e[ea] if fid[ea] == f else e[eb]
                db = e[eb] if fid[eb] == g else e[ea]
                if da[0] == db[0]:
                    F[g] = F[g][::-1]
                    rows = [g, g + m, g + 2 * m]
                    e[rows] = e[rows][:, ::-1]                 # same undirected edges, reversed
                seen[g] = True
                dq.append(g)
    return orient_outward(V, F)


def refan_concave_caps(V: np.ndarray, F: np.ndarray, materials=None, min_valence: int = 5):
    """Re-triangulate flat cap fans that fold over themselves: a vertex whose incident faces are coplanar and whose
    link loop is not star-shaped around it (a fan to the centroid of a cambered airfoil cap) is removed and its loop
    ear-clipped. Returns ``(V, F, materials)`` (unreferenced centre vertices are left in ``V``)."""
    from .kit import _plane_coords, ear_clip_fast
    F = np.asarray(F, np.int64)
    mats = None if materials is None or isinstance(materials, str) else np.asarray(materials, dtype=object)
    face_normals(V, F)
    order = np.argsort(F.ravel(), kind="stable")
    verts = F.ravel()[order]
    faces_of = order // 3
    starts = np.searchsorted(verts, np.arange(len(V)))
    ends = np.searchsorted(verts, np.arange(len(V)), side="right")
    remove = np.zeros(len(F), bool)
    new_f, new_m = [], []
    for v in np.where(ends - starts >= min_valence)[0]:
        fi = faces_of[starts[v]:ends[v]]
        if remove[fi].any():
            continue
        pts = V[np.unique(F[fi])]
        cc = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - cc, full_matrices=False)
        if sv[-1] > 1e-7 * max(sv[0], 1e-300):
            continue                                       # not a flat cap
        nxt = {}
        for f in fi:
            tri = F[f]
            k = int(np.where(tri == v)[0][0])
            a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            nxt[a] = b
        if len(nxt) != len(fi):
            continue
        loop = [next(iter(nxt))]
        while len(loop) <= len(nxt):
            n2 = nxt.get(loop[-1])
            if n2 is None or n2 == loop[0]:
                break
            loop.append(n2)
        if len(loop) != len(nxt) or nxt.get(loop[-1]) != loop[0]:
            continue
        P = V[loop]
        nm = np.cross(P - P.mean(0), np.roll(P, -1, axis=0) - P.mean(0)).sum(0)   # Newell normal, loop order
        nm /= np.linalg.norm(nm) + 1e-300
        P2, _ = _plane_coords(np.vstack([P, V[v][None]]))
        c2 = P2[-1]
        Q = P2[:-1] - c2
        cr = Q[:, 0] * np.roll(Q[:, 1], -1) - Q[:, 1] * np.roll(Q[:, 0], -1)
        if (cr > 0).all() or (cr < 0).all():
            continue                                       # the fan is fine
        L2 = P2[:-1]
        area = 0.5 * np.sum(L2[:, 0] * np.roll(L2[:, 1], -1) - np.roll(L2[:, 0], -1) * L2[:, 1])
        try:
            T = ear_clip_fast(L2 if area > 0 else L2[::-1])
        except ValueError:
            continue
        if area <= 0:
            T = (len(loop) - 1) - T
        T = np.asarray(loop)[T]
        tn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        flip = np.einsum("ij,j->i", tn, nm) < 0
        T[flip] = T[flip][:, ::-1]
        remove[fi] = True
        new_f.append(T)
        if mats is not None:
            from collections import Counter
            new_m.append(np.full(len(T), Counter(mats[fi].tolist()).most_common(1)[0][0], dtype=object))
    if not new_f:
        return V, F, materials
    F2 = np.vstack([F[~remove]] + new_f)
    m2 = materials if mats is None else np.concatenate([mats[~remove]] + new_m)
    return V, F2, m2
