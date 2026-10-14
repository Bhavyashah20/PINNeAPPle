"""The airfoil field dataset of the zoo (``airfoil_rans_cst_re4e6_fields``): every run of the 436-run OpenFOAM campaign
(:func:`.dataset.plan`, :func:`.dataset.run_job`) with its cell fields p, Ux, Uy, nut on the shared 20064-cell O-grid,
the cell centres, the CST weights, the angle of attack, the final-iteration coefficients and the run status, in
Hub-only ``<split>.npz`` files (``train``, ``test`` = the held-out shapes of the airfoil surrogates,
``naca_reference``).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .train import split as split_by_shape

__all__ = ["write_field_dataset"]


def _write_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    """Deflated npz with fixed zip timestamps, so the SHA-256 depends only on the data."""
    import io
    import zipfile
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in sorted(zin.infolist(), key=lambda i: i.filename):
            zi = zipfile.ZipInfo(item.filename, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, zin.read(item.filename))


def write_field_dataset(run_dir: str, out: str, name: str = "airfoil_rans_cst_re4e6_fields") -> dict:
    """Write the zoo field dataset from the campaign's run files: ``out/<name>/{train,test,naca_reference}.npz``
    (arrays ``<split>__<column>``) and ``meta.json``.

    Every run is kept with its final-iteration coefficients and a ``status`` column; the split is
    :func:`.train.split` applied to all LHS shapes (the held-out shapes are those of the shipped surrogates). Cell
    centres ``x``, ``y`` are stored per run (the grid moves with the shape)."""
    from .mesh import GridSpec, cell_centres, grid_points

    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "runs", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        r["npz"] = f[:-5] + ".npz"
        if os.path.exists(r["npz"]):
            recs.append(r)
    tr, te, ref = split_by_shape(recs)
    g = GridSpec()
    d = os.path.join(out, name)
    files, rows = [], {}
    for split, rs in (("train", tr), ("test", te), ("naca_reference", ref)):
        cols: dict[str, list] = {k: [] for k in ("run_id", "shape_id", "T0", "T1", "T2", "M0", "M1", "M2",
                                                 "alpha_deg", "Cl", "Cd", "Cm", "status", "x", "y", "p", "Ux",
                                                 "Uy", "nut")}
        for r in sorted(rs, key=lambda r: r["id"]):
            s = np.asarray(r["shape"])
            cc = cell_centres(grid_points(s, g)).reshape(-1, 2)
            z = np.load(r["npz"])
            for k, v in zip(("T0", "T1", "T2", "M0", "M1", "M2"), s, strict=True):
                cols[k].append(v)
            cols["run_id"].append(r["id"])
            cols["shape_id"].append(r["shape_id"])
            cols["alpha_deg"].append(r["alpha"])
            for k in ("Cl", "Cd", "Cm"):
                cols[k].append(r.get(k, np.nan))
            cols["status"].append(r.get("status", ""))
            cols["x"].append(cc[:, 0])
            cols["y"].append(cc[:, 1])
            for k in ("p", "Ux", "Uy", "nut"):
                cols[k].append(z[k])
        arrays = {}
        for k, v in cols.items():
            if k in ("run_id", "shape_id", "status"):
                arrays[f"{split}__{k}"] = np.array(v, dtype=str)
            elif k in ("x", "y", "p", "Ux", "Uy", "nut"):
                arrays[f"{split}__{k}"] = np.stack(v).astype(np.float32)
            else:
                arrays[f"{split}__{k}"] = np.asarray(v, dtype=np.float64)
        _write_npz(os.path.join(d, f"{split}.npz"), arrays)
        files.append(f"{split}.npz")
        rows[split] = len(rs)
    meta = {"name": name, "files": files, "rows": rows, "n_cells": int(g.ni * (g.nj)),
            "grid": {"ni": g.ni, "nj": g.nj}, "test_shapes": sorted({r["shape_id"] for r in te})}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
