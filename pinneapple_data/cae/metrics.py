"""Per-cell flow metrics on a finite-volume mesh: Courant number and distance to the nearest wall.

The array functions take plain face/cell arrays (face area vectors, owner, neighbour, volumes: what
:func:`pinneapple_data.cae.geometry.face_geometry` / ``cell_geometry`` return, or what any FV code stores); the
``mesh_*`` wrappers compute those from a :class:`~pinneapple_data.cae.model.Mesh` (an OpenFOAM polyMesh read by
:func:`pinneapple_data.cae.foam.read_polymesh`, or an FE volume mesh converted to faces).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import Mesh

__all__ = ["courant_number", "wall_distance", "mesh_courant_number", "mesh_wall_distance"]


def courant_number(face_area_vectors, face_owner, face_neighbour, U, cell_volumes, dt: float,
                   where: str = "auto") -> np.ndarray:
    """Cell Courant number ``Co_P = 0.5 dt sum_f |U_f . S_f| / V_P`` over the faces f of cell P.

    The finite-volume form of ``|U| dt / dx`` for any polyhedron (Courant, Friedrichs & Lewy, Math. Ann. 100, 1928;
    Ferziger, Peric & Street, *Computational Methods for Fluid Dynamics*, 4th ed., 2020, §6.3): the outgoing plus
    incoming volume flux over the cell volume, halved. On an axis-aligned hexahedron of side dx in uniform flow it
    reduces to ``|U| dt / dx``.

    ``face_area_vectors`` (nf, 3); ``face_owner`` (nf,); ``face_neighbour`` (n_internal,) with the internal faces first
    (OpenFOAM ordering). ``U`` is either a face velocity (nf, 3) or a cell velocity (n_cells, 3), interpolated to the
    faces as the mean of owner and neighbour (owner value on boundary faces); ``where`` = ``"face"`` / ``"cell"``
    forces the choice when the counts coincide (``"auto"`` prefers faces). Units: dt [s], U [m/s], S [m^2], V [m^3].
    """
    sf = np.asarray(face_area_vectors, dtype=np.float64)
    own = np.asarray(face_owner, dtype=np.int64)
    nei = np.asarray(face_neighbour, dtype=np.int64)
    vol = np.asarray(cell_volumes, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    nf, nc, ni = len(sf), len(vol), len(nei)
    if where == "auto":
        where = "face" if len(U) == nf else "cell" if len(U) == nc else ""
    if where == "face":
        if len(U) != nf:
            raise ValueError(f"face velocity needs {nf} rows, got {len(U)}")
        uf = U
    elif where == "cell":
        if len(U) != nc:
            raise ValueError(f"cell velocity needs {nc} rows, got {len(U)}")
        uf = U[own].copy()
        uf[:ni] = 0.5 * (U[own[:ni]] + U[nei])
    else:
        raise ValueError(f"U has {len(U)} rows: neither the face count {nf} nor the cell count {nc}")
    phi = np.abs(np.einsum("ij,ij->i", uf, sf))
    total = np.bincount(own, phi, nc) + np.bincount(nei, phi[:ni], nc)
    return 0.5 * dt * total / vol


def wall_distance(cell_centres, wall_face_centres, wall_face_normals=None) -> np.ndarray:
    """Distance from each cell centre to the nearest wall face centre (``scipy.spatial.cKDTree`` query).

    With ``wall_face_normals`` (area vectors or unit normals of the same faces) the distance is projected on the
    normal of that nearest face, ``|(c - f) . n|``, which is exact for a flat wall and removes the in-plane offset to
    the face centre. Without normals the result over-estimates the true distance by at most half a wall face
    diagonal.
    """
    from scipy.spatial import cKDTree

    c = np.asarray(cell_centres, dtype=np.float64)
    f = np.asarray(wall_face_centres, dtype=np.float64)
    if len(f) == 0:
        raise ValueError("no wall faces")
    d, i = cKDTree(f).query(c)
    if wall_face_normals is None:
        return d
    n = np.asarray(wall_face_normals, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return np.abs(np.einsum("ij,ij->i", c - f[i], n[i]))


def _geometry(mesh: Mesh):
    from . import geometry as G

    fm = G.face_mesh(mesh)
    fg = G.face_geometry(fm)
    return fm, fg, G.cell_geometry(fm, fg)


def mesh_courant_number(mesh: Mesh, U, dt: float, where: str = "auto") -> np.ndarray:
    """:func:`courant_number` on a :class:`Mesh` (cell or face velocity ``U``, see there)."""
    fm, fg, cg = _geometry(mesh)
    return courant_number(fg["areas"], fm.owner, fm.neighbour, U, cg["volumes"], dt, where=where)


def mesh_wall_distance(mesh: Mesh, patches: Sequence[str] | None = None, project: bool = True) -> np.ndarray:
    """:func:`wall_distance` of every cell of a :class:`Mesh` to the faces of ``patches`` (default: every patch of
    type ``wall``). ``project`` uses the face normals (see :func:`wall_distance`)."""
    fm, fg, cg = _geometry(mesh)
    if patches is None:
        ids = [k for k, t in enumerate(fm.patch_types) if t == "wall"]
        if not ids:
            raise ValueError("no patch of type 'wall'; name the wall patches")
    else:
        unknown = set(patches) - set(fm.patch_names)
        if unknown:
            raise ValueError(f"unknown patches {sorted(unknown)}; have {fm.patch_names}")
        ids = [fm.patch_names.index(p) for p in patches]
    faces = fm.n_internal + np.nonzero(np.isin(fm.patch_of_face, ids))[0]
    normals = fg["areas"][faces] if project else None
    return wall_distance(cg["centres"], fg["centres"][faces], normals)
