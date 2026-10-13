"""Global balances over the boundary of a box on a tensor-product grid (a stored grid, or a model evaluated on one):
net volume flux, net heat flow and resultant force, each with the total absolute flow used to normalise it."""
from __future__ import annotations

import numpy as np

__all__ = ["grid_mass", "grid_heat", "grid_force"]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _faces(coords: list[np.ndarray], names: list[str]):
    """(face name, axis, index along it, outward sign)."""
    for i, a in enumerate(names):
        yield f"{a}=min", i, 0, -1.0
        yield f"{a}=max", i, -1, 1.0


def _integrate(v: np.ndarray, coords: list[np.ndarray], skip: int) -> float:
    """Integral over all axes but ``skip`` of the face slice ``v`` (already without that axis)."""
    out = v
    rest = [c for k, c in enumerate(coords) if k != skip]
    for c in reversed(rest):
        out = _trapz(out, c, axis=-1)
    return float(out)


def _volume_integral(v: np.ndarray, coords: list[np.ndarray]) -> float:
    out = v
    for c in reversed(coords):
        out = _trapz(out, c, axis=-1)
    return float(out)


def grid_mass(coords: list[np.ndarray], names: list[str], vel: list[np.ndarray]) -> dict:
    per, tot = {}, 0.0
    for face, i, idx, s in _faces(coords, names):
        un = s * np.take(vel[i], idx, axis=i)
        per[face] = _integrate(un, coords, i)
        tot += _integrate(np.abs(un), coords, i)
    area = sum(_integrate(np.ones_like(np.take(vel[0], 0, axis=i)), coords, i) * 2 for i in range(len(coords)))
    uref = float(np.max(np.sqrt(np.sum([c ** 2 for c in vel], axis=0))))
    return {"net": float(sum(per.values())), "total": tot, "patches": per, "area": area, "uref": uref}


def grid_heat(coords: list[np.ndarray], names: list[str], T: np.ndarray, k: float, q: float) -> dict:
    """Heat flow INTO the domain: k dT/dn (outward normal) over the boundary, plus the source."""
    per, tot = {}, 0.0
    for face, i, idx, s in _faces(coords, names):
        g = np.gradient(T, coords[i], axis=i, edge_order=2)
        qn = s * k * np.take(g, idx, axis=i)
        per[face] = _integrate(qn, coords, i)
        tot += _integrate(np.abs(qn), coords, i)
    src = float(q) * _volume_integral(np.ones_like(T), coords)
    return {"net": float(sum(per.values())) + src, "total": tot + abs(src), "source": src, "patches": per}


def grid_force(coords: list[np.ndarray], names: list[str], disp: list[np.ndarray], E: float, nu: float,
               f: np.ndarray) -> dict:
    """Resultant of the boundary tractions sigma.n plus the body force (plane strain in 2-D)."""
    nd = len(coords)
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    G = [[np.gradient(disp[i], coords[j], axis=j, edge_order=2) for j in range(nd)] for i in range(nd)]
    div = sum(G[i][i] for i in range(nd))
    sig = [[lam * div * (i == j) + mu * (G[i][j] + G[j][i]) for j in range(nd)] for i in range(nd)]
    R = np.zeros(nd)
    tot = 0.0
    per = {}
    for face, j, idx, s in _faces(coords, names):
        t = [s * np.take(sig[i][j], idx, axis=j) for i in range(nd)]
        ti = np.array([_integrate(x, coords, j) for x in t])
        per[face] = float(np.linalg.norm(ti))
        R += ti
        tot += sum(_integrate(np.abs(x), coords, j) for x in t)
    vol = _volume_integral(np.ones_like(disp[0]), coords)
    fb = np.asarray(f, float) * vol
    R += fb
    return {"net": float(np.linalg.norm(R)), "resultant": R.tolist(), "total": tot + float(np.abs(fb).sum()),
            "patches": per}
