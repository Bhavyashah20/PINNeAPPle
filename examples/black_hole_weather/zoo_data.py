"""Read the black-hole zoo dataset (``blackhole_riaf_density_256``) as whole trajectories.

    from zoo_data import load_trajectories
    trajs = load_trajectories("test")                 # {seed: {"t": (T,), "log10_rho": (T, 192, 96), "mdot": (T,)}}
    trajs = load_trajectories("train", path="dist/zoo-staging/blackhole_riaf_density_256/data.npz")
"""
from __future__ import annotations

import math

import numpy as np

NAME = "blackhole_riaf_density_256"
Q_LO, Q_HI = -8.0, 1.0


def _split(split: str, path: str | None) -> dict[str, np.ndarray]:
    if path is None:
        import pinneapple as pp
        return dict(pp.datasets.load(NAME, split=split))
    with np.load(path, allow_pickle=False) as z:
        return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(split + "__")}


def load_trajectories(split: str, path: str | None = None) -> dict[int, dict[str, np.ndarray]]:
    d = _split(split, path)
    out = {}
    for seed in np.unique(d["run"]):
        k = d["run"] == seed
        order = np.argsort(d["t"][k])
        out[int(seed)] = {"t": d["t"][k][order].astype(float), "mdot": d["mdot"][k][order],
                          "log10_rho": (Q_LO + d["log10_rho_q"][k][order].astype(np.float32) * ((Q_HI - Q_LO) / 65535.0)),
                          "alpha": float(d["alpha"][k][0])}
    return out


def window_volumes(path: str | None = None) -> np.ndarray:
    """2 pi r^2 dr d(cos theta) of the window cells (192, 96)."""
    gr, gt = _split("grid_r", path), _split("grid_theta", path)
    vr = (gr["r_face_hi"] ** 3 - gr["r_face_lo"] ** 3) / 3.0
    vt = np.cos(gt["theta_face_lo"]) - np.cos(gt["theta_face_hi"])
    return 2 * math.pi * vr[:, None] * vt[None, :]
