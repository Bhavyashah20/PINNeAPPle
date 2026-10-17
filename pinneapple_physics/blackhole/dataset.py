"""The black-hole weather dataset: density frames of whole simulated trajectories, packed for the model zoo.

    from pinneapple_physics.blackhole.dataset import build_arrays, write_npz, load_trajectories, window_volumes
    arrays = build_arrays("data/bh_runs", conv_report)            # from simulate_run folders (see SPLITS)
    write_npz(arrays, "dist/zoo-staging/blackhole_riaf_density_256/data.npz")
    test = load_trajectories("test")                             # pp.datasets.load under the hood, or path=...
    # {seed: {"t": (T,), "log10_rho": (T, 192, 96), "mdot": (T,), "alpha": float}}

One row per frame, 20 M apart, t in [2000, 10000] M, whole trajectories per split (no trajectory in two splits):
``train`` seeds 0-3, ``val`` seed 4, ``test`` seed 5 (alpha 0.1), ``ood_alpha0.3`` (alpha 0.3, seed 0). The density in
the forecast window (r < 126.5 M, theta in [pi/8, 7 pi/8]) is stored as uint16:
``log10 rho = Q_LO + q (Q_HI - Q_LO) / 65535`` (steps of 1.4e-4 dex).
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path

import numpy as np

from .convergence import FRAME_DT, window_mask
from .runs import load_run

__all__ = ["NAME", "Q_LO", "Q_HI", "SPLITS", "CONV_SCALARS", "encode", "decode", "build_arrays", "write_npz",
           "write_split_files",
           "deterministic_npz", "load_trajectories", "window_volumes"]

NAME = "blackhole_riaf_density_256"
Q_LO, Q_HI = -8.0, 1.0
T0, T1 = 2000.0, 10000.0
SPLITS = {"train": [("conv_n256_s0", 0.1, 0), ("conv_n256_s1", 0.1, 1), ("conv_n256_s2", 0.1, 2),
                    ("conv_n256_s3", 0.1, 3)],
          "val": [("conv_n256_s4", 0.1, 4)],
          "test": [("conv_n256_s5", 0.1, 5)],
          "ood_alpha0.3": [("ood_n256_a0.3_s0", 0.3, 0)]}
CONV_SCALARS = ["mdot_mean", "mdot_rms_over_mean", "mdot_detrended_rms", "mdot_psd_slope", "window_mass_mean", "persistence_mae_20M",
                "persistence_mae_100M", "persistence_mae_1000M", "rho_slope_r10_60", "ell_over_kepler_r10",
                "ell_over_kepler_r30"]


def encode(log10_rho: np.ndarray) -> np.ndarray:
    """log10 density -> uint16 (raises outside [Q_LO, Q_HI])."""
    q = np.round((np.asarray(log10_rho, float) - Q_LO) / (Q_HI - Q_LO) * 65535.0)
    if q.min() < 0 or q.max() > 65535:
        raise ValueError(f"log10 rho outside [{Q_LO}, {Q_HI}]")
    return q.astype(np.uint16)


def decode(q: np.ndarray) -> np.ndarray:
    """uint16 column ``log10_rho_q`` -> log10 density (float32)."""
    return (Q_LO + np.asarray(q, np.float64) * ((Q_HI - Q_LO) / 65535.0)).astype(np.float32)


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """npz with fixed zip timestamps and order, so the SHA-256 does not change between identical builds."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, buf.getvalue())
    return out.getvalue()


def build_arrays(runs: str | Path, conv: dict | None = None, *, splits: dict = SPLITS, t0: float = T0,
                 t1: float = T1) -> dict[str, np.ndarray]:
    """Arrays of the zoo dataset (``<split>__<column>``) from the simulate_run folders under ``runs`` and the
    resolution-study report ``conv`` (its scalars per grid and seed become the ``convergence`` split)."""
    runs = Path(runs)
    arrays: dict[str, np.ndarray] = {}
    grid = None
    for split, members in splits.items():
        cols = {"run": [], "alpha": [], "t": [], "mdot": [], "log10_rho_q": []}
        for folder, alpha, seed in members:
            run = load_run(str(runs / folder))
            t = run["t"]
            stride = int(round(FRAME_DT / float(np.diff(t).mean())))
            k = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
            idx = np.nonzero(k)[0][::stride]
            if len(idx) != int(round((t1 - t0) / FRAME_DT)) + 1:
                raise ValueError(f"{folder}: {len(idx)} frames in [{t0}, {t1}] (incomplete run?)")
            g = run["grid"]
            m = window_mask(g)
            ri, ti = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
            rho = run["frames"][idx, 0][:, ri[0]:ri[-1] + 1, ti[0]:ti[-1] + 1]
            cols["log10_rho_q"].append(encode(np.log10(np.clip(rho, 1e-30, None))))
            cols["t"].append(t[idx].astype(np.float32))
            cols["mdot"].append(run["mdot"][idx].astype(np.float32))
            cols["run"].append(np.full(len(idx), seed, np.int16))
            cols["alpha"].append(np.full(len(idx), alpha, np.float32))
            grid = (g, ri, ti)
        for c, v in cols.items():
            arrays[f"{split}__{c}"] = np.concatenate(v)
    g, ri, ti = grid
    arrays["grid_r__r"] = g["r"][ri].astype(np.float64)
    arrays["grid_r__r_face_lo"] = g["r_faces"][ri].astype(np.float64)
    arrays["grid_r__r_face_hi"] = g["r_faces"][ri + 1].astype(np.float64)
    arrays["grid_theta__theta"] = g["theta"][ti].astype(np.float64)
    arrays["grid_theta__theta_face_lo"] = g["theta_faces"][ti].astype(np.float64)
    arrays["grid_theta__theta_face_hi"] = g["theta_faces"][ti + 1].astype(np.float64)
    if conv is None:
        return arrays
    # resolution-study scalars, one row per run (grid, seed); seeds parsed from the run names "..._s<seed>"
    rows = {"nr": [], "ntheta": [], "seed": [], **{q: [] for q in CONV_SCALARS}}
    for nr, nth in conv["levels"]:
        for i, run_name in enumerate(conv["runs"][str(nr)] if str(nr) in conv["runs"] else conv["runs"][nr]):
            rows["nr"].append(nr)
            rows["ntheta"].append(nth)
            rows["seed"].append(int(re.search(r"_s(\d+)$", run_name).group(1)))
            for q in CONV_SCALARS:
                bl = conv["scalars"][q]["by_level"]
                rows[q].append((bl[str(nr)] if str(nr) in bl else bl[nr])["per_seed"][i])
    for c, v in rows.items():
        arrays[f"convergence__{c}"] = np.asarray(v, np.int32 if c in ("nr", "ntheta", "seed") else np.float64)
    return arrays


def write_npz(arrays: dict[str, np.ndarray], path: str | Path) -> Path:
    """All splits in one file (``<split>__<column>`` arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_npz(arrays))
    return path


def write_split_files(arrays: dict[str, np.ndarray], folder: str | Path) -> dict[str, Path]:
    """One ``<split>.npz`` per split in ``folder`` (the zoo layout for large datasets: a user downloads only the split
    they load, and every file stays far below the 100 MB a git hosting allows). Returns {file name: path}."""
    by_split: dict[str, dict[str, np.ndarray]] = {}
    for k, v in arrays.items():
        by_split.setdefault(k.split("__", 1)[0], {})[k] = v
    return {f"{s}.npz": write_npz(a, Path(folder) / f"{s}.npz") for s, a in sorted(by_split.items())}


def _split(split: str, path: str | Path | None) -> dict[str, np.ndarray]:
    """One split as {column: array}: from the zoo (path None), a folder of ``<split>.npz`` files or one npz."""
    if path is None:
        from pinneapple_hub.zoo import datasets
        return dict(datasets.load(NAME, split=split))
    path = Path(path)
    if path.is_dir():
        path = path / f"{split}.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(split + "__")}


def load_trajectories(split: str, path: str | Path | None = None) -> dict[int, dict]:
    """Whole trajectories of a split, by seed, frames in time order. ``path``: a local data.npz (default: the zoo)."""
    d = _split(split, path)
    out = {}
    for seed in np.unique(d["run"]):
        k = d["run"] == seed
        order = np.argsort(d["t"][k])
        out[int(seed)] = {"t": d["t"][k][order].astype(float), "mdot": d["mdot"][k][order],
                          "log10_rho": decode(d["log10_rho_q"][k][order]), "alpha": float(d["alpha"][k][0])}
    return out


def window_volumes(path: str | Path | None = None) -> np.ndarray:
    """2 pi r^2 dr d(cos theta) of the window cells."""
    gr, gt = _split("grid_r", path), _split("grid_theta", path)
    vr = (gr["r_face_hi"] ** 3 - gr["r_face_lo"] ** 3) / 3.0
    vt = np.cos(gt["theta_face_lo"]) - np.cos(gt["theta_face_hi"])
    return 2 * math.pi * vr[:, None] * vt[None, :]
