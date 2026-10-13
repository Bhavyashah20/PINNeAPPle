"""Turn the part lists of the earlier models (aero ``Part``, vehicle ``CADPart``) into closed library bodies.

The earlier models split one closed skin into several parts by material (paint, windows, livery, ...). The adapter
pools the parts that share a group, welds coincident vertices and splits the pool into connected shells: each
shell becomes one :class:`Body` with one material per face, so the watertight checks see the whole skin.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np

from ..core import Body
from ..meshops import components, edge_stats

Item = tuple[str, np.ndarray, np.ndarray, str, str, bool]   # name, V, F, material, group, aero


def bodies_from_items(items: Iterable[Item], prefix: str = "") -> list[Body]:
    pools: OrderedDict[str, list] = OrderedDict()
    for it in items:
        pools.setdefault(it[4] or base_name(it[0], it[3]), []).append(it)
    out: list[Body] = []
    for group, its in pools.items():
        V = np.vstack([np.asarray(i[1], float) for i in its])
        offs = np.cumsum([0] + [len(i[1]) for i in its])[:-1]
        F = np.vstack([np.asarray(i[2], np.int64) + o for i, o in zip(its, offs, strict=False)])
        mats = np.concatenate([np.full(len(i[2]), i[3], dtype=object) for i in its])
        names = np.concatenate([np.full(len(i[2]), i[0], dtype=object) for i in its])
        aero = np.concatenate([np.full(len(i[2]), bool(i[5])) for i in its])
        Vw, Fw = _weld_keep(V, F)
        lab = components(Fw, len(Vw))
        # a welded shell that is not edge-manifold is several primitives touching at corners (box beams of a
        # frame): keep those primitives apart, using the connectivity of the original, unwelded faces
        lab_orig = components(F, len(V))
        for c in np.unique(lab):
            sel = lab == c
            if edge_stats(Vw, Fw[sel])["nonmanifold_edges"]:
                lab = np.where(sel, lab.max() + 1 + lab_orig, lab)
        for c in np.unique(lab):
            sel = lab == c
            used, inv = np.unique(Fw[sel], return_inverse=True)
            nm = Counter(names[sel].tolist()).most_common(1)[0][0]
            fm = mats[sel]
            out.append(Body(f"{prefix}{nm}", Vw[used], inv.reshape(-1, 3),
                            fm[0] if len(set(fm.tolist())) == 1 else fm, group=group, aero=bool(aero[sel].all())))
    # unique names
    seen = Counter()
    for b in out:
        seen[b.name] += 1
        if seen[b.name] > 1:
            b.name = f"{b.name}_{seen[b.name]}"
    return out


def base_name(name: str, material: str) -> str:
    """The skin a material-split part came from: ``fuselage_window`` -> ``fuselage``."""
    return name[: -len(material) - 1] if name.endswith("_" + material) else name


def _weld_keep(V, F):
    """Weld without dropping faces (collapsed faces of the earlier meshes are kept so that face materials stay
    aligned); the duplicates are removed afterwards by :func:`drop_degenerate`."""
    scale = max(float(np.ptp(V, axis=0).max()), 1e-12)
    q = np.round(V / (1e-9 * scale)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.ravel()[F]


def drop_degenerate(bodies: Sequence[Body]) -> list[Body]:
    out = []
    for b in bodies:
        F = b.faces
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        if not ok.all():
            fm = b.materials if isinstance(b.materials, str) else b.materials[ok]
            b = Body(b.name, b.vertices, F[ok], fm, b.group, b.aero, b.closed)
        out.append(b)
    return out


def close_sheets(bodies: Sequence[Body], thickness: float = 0.004) -> list[Body]:
    """Give the open sheets of the earlier models (rims, decals) a thickness so every body is a closed shell."""
    from ..meshops import edge_stats, fix_winding, refan_concave_caps, solidify
    out = []
    for b in bodies:
        V, F, m = refan_concave_caps(b.vertices, b.faces, b.materials)
        if len(F) != len(b.faces) or not np.array_equal(F, b.faces):
            used, inv = np.unique(F, return_inverse=True)
            b = Body(b.name, V[used], inv.reshape(-1, 3), m, b.group, b.aero, b.closed)
        es = edge_stats(b.vertices, b.faces)
        if es["misoriented_edges"] and not es["boundary_edges"]:
            b.vertices, b.faces = fix_winding(b.vertices, b.faces)
            es = edge_stats(b.vertices, b.faces)
        if es["boundary_edges"] or es["nonmanifold_edges"]:
            V, F, keep = solidify(b.vertices, b.faces, thickness)
            fm = b.materials if isinstance(b.materials, str) else np.concatenate([b.materials[keep]] * 2 + [
                np.full(len(F) - 2 * int(keep.sum()), b.materials[0], dtype=object)])
            b = Body(b.name, V, F, fm, b.group, b.aero, True)
        out.append(b)
    return out
