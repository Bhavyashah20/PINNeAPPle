"""Resolution study of the black-hole accretion simulations: is the forecaster's training data converged?

    from pinneapple_physics.blackhole.convergence import resolution_study
    report = resolution_study({128: ["runs/conv_n128_s0", ...], 181: [...], 256: [...], 362: [...]},
                              t0=2000.0, t1=10000.0)

The runs of each grid share the physics and the seeded initial perturbation (``RIAFConfig(perturbation_kind="modes")``
samples one continuous field on every grid). Compared on the coarsest grid after a conservative remap (the grids
need not nest):

* the accretion rate at the inner edge: time mean, rms variability, slope of its power spectrum;
* the time-mean radial profiles of density and specific angular momentum (shell averages), the density slope and
  l / l_Kepler, and the advective angular-momentum flux through each shell;
* the statistics of the field the forecaster is scored on (normalised log density in the forecast window): its
  time mean and temporal standard deviation, and the persistence error at 20, 100 and 1000 M.

Each quantity gets the mean over seeds with a 95 % interval (batch means within a run combined with the
seed-to-seed spread), the observed order and GCI of the three finest grids
(``pinneapple_analysis.verification.richardson_extrapolate``, Roache 1994) when they converge monotonically, and a
**statistical convergence** verdict: the two finest grids agree within their combined interval.
"""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from pinneapple_analysis.verification.convergence import richardson_extrapolate
from pinneapple_analysis.verification.forecast_skill import batch_means_ci

from .runs import load_run

__all__ = ["R_WINDOW", "TH_WINDOW", "FRAME_DT", "window_mask", "cell_volumes", "remap", "remap_r", "psd_slope",
           "run_statistics", "resolution_study", "LOG_RHO_RANGE"]

FRAME_DT = 20.0                  # forecaster frame spacing (M)
R_WINDOW = 4.0 * 100.0 ** 0.75   # forecast window: r < 126.5 M (the inner 96 of 128 radial cells) ...
TH_WINDOW = math.pi / 8          # ... and theta in [pi/8, 7 pi/8] (8 of 64 cells dropped at each pole)
LOG_RHO_RANGE = (-6.0, 0.0)      # fixed normalisation of log10 rho for comparing grids


def window_mask(grid: Mapping[str, np.ndarray]) -> np.ndarray:
    """Cells of any grid whose centre lies in the physical forecast window."""
    r, th = grid["r"], grid["theta"]
    return (r[:, None] < R_WINDOW) & (th[None, :] > TH_WINDOW) & (th[None, :] < math.pi - TH_WINDOW)


def cell_volumes(grid: Mapping[str, np.ndarray]) -> np.ndarray:
    """2 pi r^2 dr d(cos theta) of every cell."""
    rf, tf = grid["r_faces"], grid["theta_faces"]
    return 2 * math.pi * ((rf[1:] ** 3 - rf[:-1] ** 3) / 3.0)[:, None] * (np.cos(tf[:-1]) - np.cos(tf[1:]))[None, :]


def _overlap(fine: np.ndarray, coarse: np.ndarray, prim) -> np.ndarray:
    """O[i, j] = integral of the weight over fine cell i intersected with coarse cell j (``prim``: antiderivative)."""
    lo = np.maximum(fine[:-1, None], coarse[None, :-1])
    hi = np.minimum(fine[1:, None], coarse[None, 1:])
    return np.where(hi > lo, prim(hi) - prim(lo), 0.0)


def remap(a: np.ndarray, faces_f: tuple[np.ndarray, np.ndarray], faces_c: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Conservative (volume-weighted) remap of cell averages (..., nr, nth) between two spherical grids covering the
    same domain; the grids need not nest. The volume r^2 dr sin(theta) dtheta is separable, so the remap is one
    overlap matrix per axis."""
    o_r = _overlap(faces_f[0], faces_c[0], lambda r: r ** 3 / 3.0)
    o_t = _overlap(faces_f[1], faces_c[1], lambda t: -np.cos(t))
    num = np.einsum("...ik,ij,kl->...jl", a, o_r, o_t)
    den = np.einsum("ij,kl->jl", o_r, o_t)
    return num / den


def remap_r(a: np.ndarray, rf_f: np.ndarray, rf_c: np.ndarray) -> np.ndarray:
    """Remap of a radial profile (shell averages) by overlap length in log r."""
    ov = _overlap(np.log(rf_f), np.log(rf_c), lambda x: x)
    return (a @ ov) / ov.sum(0)


def psd_slope(x: np.ndarray, dt: float, f_lo: float, f_hi: float) -> float:
    """Slope of the Welch power spectrum of x in log-log between f_lo and f_hi."""
    from scipy import signal
    f, p = signal.welch(x - x.mean(), fs=1.0 / dt, nperseg=min(len(x), 256))
    k = (f >= f_lo) & (f <= f_hi) & (p > 0)
    return float(np.polyfit(np.log(f[k]), np.log(p[k]), 1)[0])


def run_statistics(folder: str, t0: float, t1: float, log_rho_range: tuple[float, float] = LOG_RHO_RANGE) -> dict:
    """Time-window statistics of one run on its own grid: scalars as time series, profiles and fields as time means."""
    run = load_run(folder)
    t, fr, mdot = run["t"], run["frames"], run["mdot"]
    k = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
    fr, tk, md = fr[k], t[k], mdot[k]
    g = run["grid"]
    vol = cell_volumes(g)
    rho, vr, vph = fr[:, 0].astype(float), fr[:, 1].astype(float), fr[:, 3].astype(float)
    r, th = g["r"], g["theta"]
    ell = (r[:, None] * np.sin(th)[None, :])[None] * vph
    area = (r[:, None] ** 2) * (np.cos(g["theta_faces"][:-1]) - np.cos(g["theta_faces"][1:]))[None]
    mask = window_mask(g)
    every = float(np.diff(tk).mean())
    lo, hi = log_rho_range
    x = ((np.log10(np.clip(rho, 1e-30, None)) - lo) / (hi - lo))[::max(int(round(FRAME_DT / every)), 1)]
    xw = x[:, mask]
    out = {"t": tk, "mdot": md, "every": every, "grid": g, "vol": vol, "mask": mask,
           "faces": (g["r_faces"], g["theta_faces"]),
           "rho_shell": (rho * vol).sum(-1).mean(0) / vol.sum(-1),
           "ell_shell": (rho * ell * vol).sum(-1).mean(0) / (rho * vol).sum(-1).mean(0),
           "jflux": 2 * math.pi * (rho * vr * ell * area).sum(-1).mean(0),
           "mass_window": (rho * vol * mask).sum((1, 2)),
           "x_mean": x.mean(0), "x_std": x.std(0)}
    for lead_m in (20, 100, 1000):
        L = int(round(lead_m / FRAME_DT))
        out[f"persistence_{lead_m}"] = np.abs(xw[L:] - xw[:-L]).mean(1)       # error time series
    return out


def _richardson(f_coarse, f_medium, f_fine, ratio) -> dict:
    """Observed order / GCI of the three finest grids, or why they are not available (non-monotone or diverging)."""
    fc, fm, ff = (np.asarray(v, float) for v in (f_coarse, f_medium, f_fine))
    if fc.ndim == 0 and (ff - fm) * (fm - fc) <= 0:
        return {"monotone": False, "observed_order": None, "gci_fine": None, "extrapolated": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = richardson_extrapolate(fc, fm, ff, ratio)
    except ValueError as e:
        return {"monotone": False, "observed_order": None, "gci_fine": None, "extrapolated": None, "note": str(e)}
    ok = res.observed_order > 0
    return {"monotone": True, "observed_order": res.observed_order, "gci_fine": res.gci_fine if ok else None,
            "extrapolated": (float(res.extrapolated_value) if fc.ndim == 0 else None) if ok else None,
            "asymptotic_ratio": res.asymptotic_ratio, "is_asymptotic": bool(res.is_asymptotic and ok)}


def _t95(n: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


SCALARS = {
    "mdot_mean": lambda st: batch_means_ci(st["mdot"]),
    "mdot_rms_over_mean": lambda st: (float(st["mdot"].std() / st["mdot"].mean()), 0.0),
    "mdot_detrended_rms": lambda st: _rms_ci(_detrended(st["mdot"], st["every"])),
    "mdot_psd_slope": lambda st: (psd_slope(st["mdot"], st["every"], 1 / 2000, 1 / 50), 0.0),
    "window_mass_mean": lambda st: batch_means_ci(st["mass_window"]),
    "persistence_mae_20M": lambda st: batch_means_ci(st["persistence_20"]),
    "persistence_mae_100M": lambda st: batch_means_ci(st["persistence_100"]),
    "persistence_mae_1000M": lambda st: batch_means_ci(st["persistence_1000"]),
    "field_temporal_std_mean": lambda st: (float(st["x_std"][st["mask"]].mean()), 0.0),
}


def _detrended(x: np.ndarray, every: float, window_M: float = 500.0) -> np.ndarray:
    """x / (running mean over ``window_M``) - 1: the fluctuations without the secular drain of the torus."""
    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(np.asarray(x, float), max(int(round(window_M / every)), 1), mode="nearest")
    return x / trend - 1.0


def _rms_ci(x: np.ndarray) -> tuple[float, float]:
    """rms of x and its 95 % half-width (batch means of x^2, propagated through the square root)."""
    m2, h2 = batch_means_ci(np.square(x))
    rms = math.sqrt(m2)
    return rms, h2 / (2 * rms) if rms > 0 else 0.0


def _rho_slope(st, lo=10.0, hi=60.0):
    r = st["grid"]["r"]
    k = (r >= lo) & (r <= hi)
    return float(np.polyfit(np.log(r[k]), np.log(st["rho_shell"][k]), 1)[0]), 0.0


def _ell_over_kepler(st, r0):
    r = st["grid"]["r"]
    i = int(np.argmin(np.abs(r - r0)))
    return float(st["ell_shell"][i] / (r[i] ** 1.5 / (r[i] - 2.0))), 0.0


SCALARS.update({"rho_slope_r10_60": _rho_slope,
                "ell_over_kepler_r10": lambda st: _ell_over_kepler(st, 10.0),
                "ell_over_kepler_r30": lambda st: _ell_over_kepler(st, 30.0)})

PROFILES = {"rho_shell": "rho(r), shell average", "ell_shell": "l(r), mass weighted",
            "jflux": "advective angular-momentum flux through r", "x_mean": "time-mean window field",
            "x_std": "temporal std of the window field"}


def resolution_study(runs: Mapping[int, Sequence[str]], *, t0: float = 2000.0, t1: float = 10000.0,
                     log=print) -> dict:
    """``runs``: {nr: [run folders, one per seed]} with ntheta = nr / 2 and a constant refinement ratio."""
    levels = sorted(runs)
    if len(levels) < 3:
        raise ValueError("a resolution study needs at least three grids")
    ratio = (levels[-1] / levels[0]) ** (1 / (len(levels) - 1))
    if any(abs(levels[i + 1] / levels[i] / ratio - 1) > 0.01 for i in range(len(levels) - 1)):
        raise ValueError(f"grids {levels} are not in a constant refinement ratio")
    stats: dict[int, list[dict]] = {}
    for n in levels:
        stats[n] = []
        for folder in runs[n]:
            st = run_statistics(folder, t0, t1)
            stats[n].append(st)
            log(f"loaded {folder}: {len(st['t'])} frames, t {st['t'][0]:.0f}-{st['t'][-1]:.0f}")
    coarse = levels[0]
    ref = stats[coarse][0]
    report: dict = {"levels": [[n, n // 2] for n in levels], "ratio": ratio,
                    "n_seeds": {n: len(stats[n]) for n in levels}, "window_M": [t0, t1],
                    "runs": {n: [os.path.basename(os.path.normpath(f)) for f in runs[n]] for n in levels},
                    "scalars": {}, "profiles": {}, "fields": {}}

    for name, fn in SCALARS.items():
        rows = {}
        for n in levels:
            vals, cis = zip(*[fn(st) for st in stats[n]], strict=True)
            m = float(np.mean(vals))
            seed_ci = _t95(len(vals)) * float(np.std(vals, ddof=1)) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            within = float(np.sqrt(np.mean(np.square(cis))) / math.sqrt(len(vals)))
            rows[n] = {"mean": m, "ci95": float(math.hypot(seed_ci, within)), "per_seed": list(map(float, vals))}
        f3, f2, f1 = (rows[n]["mean"] for n in levels[-3:])
        c1, c2 = rows[levels[-1]]["ci95"], rows[levels[-2]]["ci95"]
        report["scalars"][name] = {
            "by_level": rows, "richardson": _richardson(f3, f2, f1, ratio),
            "finest_two_rel_diff": abs(f1 - f2) / abs(f1) if f1 else None,
            "statistically_converged": bool(abs(f1 - f2) <= math.hypot(c1, c2))}

    def on_coarse(st, key, n):
        a = st[key]
        if n == coarse:
            return a
        if a.ndim == 1:
            return remap_r(a, st["grid"]["r_faces"], ref["grid"]["r_faces"])
        return remap(a, st["faces"], ref["faces"])

    w_field = ref["mask"].astype(float)                      # pixel mean over the window, as the forecaster's MAE
    for key, label in PROFILES.items():
        prof = {n: np.mean([on_coarse(st, key, n) for st in stats[n]], 0) for n in levels}
        w = np.ones_like(prof[coarse]) if prof[coarse].ndim == 1 else w_field

        def l1(a, b, w=w):
            return float((np.abs(a - b) * w).sum() / w.sum())

        scale = l1(prof[levels[-1]], 0 * prof[levels[-1]])
        d_abs = [l1(prof[levels[i + 1]], prof[levels[i]]) for i in range(len(levels) - 1)]
        seed_spread = None
        if len(stats[levels[-1]]) > 1:
            arr = np.array([on_coarse(st, key, levels[-1]) for st in stats[levels[-1]]])
            seed_spread = float((arr.std(0, ddof=1) * w).sum() / w.sum())
        sel = (w > 0) if prof[coarse].ndim == 2 else slice(None)
        entry = {"label": label, "abs_L1_diff_consecutive": d_abs, "rel_L1_diff_consecutive": [d / scale for d in d_abs],
                 "seed_spread_abs": seed_spread,
                 "richardson": _richardson(*(prof[n][sel] for n in levels[-3:]), ratio),
                 # statistically converged: the finest change is within the seed-to-seed spread of a single run
                 "within_seed_spread": bool(seed_spread is not None and d_abs[-1] <= seed_spread)}
        if prof[coarse].ndim == 1:
            entry["r"] = ref["grid"]["r"].tolist()
            entry["by_level"] = {n: prof[n].tolist() for n in levels}
            report["profiles"][key] = entry
        else:
            report["fields"][key] = entry
    return report
