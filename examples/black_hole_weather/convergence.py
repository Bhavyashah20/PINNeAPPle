"""Resolution study of the black-hole weather reference simulation (issue #399): is the training data converged?

Runs ``simulate.py`` at three or more grids in a constant refinement ratio (same physics, same seeded initial
perturbation via ``--perturbation-kind modes``) and compares, on the coarsest grid:

* the accretion rate at the inner edge: time mean, rms variability, power-spectrum slope;
* the time-mean radial profiles of density and specific angular momentum (shell averages, mass weighted);
* the time-mean angular-momentum flux (advective) through each shell;
* the statistics of the field the forecaster is scored on (normalised log density in the forecast window):
  time-mean field, temporal standard deviation, the mean change per 20 M and the persistence error at
  100 M and 1000 M.

Turbulent (time-dependent) quantities are compared through time averages with confidence intervals (batch means
over independent blocks, and the spread over seeds when several seeds are given). For each quantity the observed
order of convergence p and the grid convergence index (GCI, Roache 1994; safety factor 1.25 for three grids) are
reported when the three grids are in the asymptotic range (monotone convergence, differences shrinking); otherwise
the spread between the two finest grids is reported as the uncertainty band.

    python examples/black_hole_weather/convergence.py --runs data/bh_conv --levels 64 128 256 --seeds 0 1 2 \\
        --t0 2000 --t1 10000 --out examples/black_hole_weather/results/convergence.json

Run names are ``conv_n{nr}_s{seed}``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import load_run  # noqa: E402

FRAME_DT = 20.0             # forecaster frame spacing (M)


R_WINDOW = 4.0 * 100.0 ** 0.75   # forecast window: r < 126.5 M (the inner 96 of 128 radial cells) ...
TH_WINDOW = math.pi / 8          # ... and theta in [pi/8, 7 pi/8] (8 of 64 cells dropped at each pole)


def window_mask(grid) -> np.ndarray:
    """Cells of any grid whose centre lies in the physical forecast window."""
    r, th = grid["r"], grid["theta"]
    return (r[:, None] < R_WINDOW) & (th[None, :] > TH_WINDOW) & (th[None, :] < math.pi - TH_WINDOW)


def _overlap(fine: np.ndarray, coarse: np.ndarray, prim) -> np.ndarray:
    """O[i, j] = integral of the weight over fine cell i intersected with coarse cell j (``prim``: antiderivative)."""
    lo = np.maximum(fine[:-1, None], coarse[None, :-1])
    hi = np.minimum(fine[1:, None], coarse[None, 1:])
    return np.where(hi > lo, prim(hi) - prim(lo), 0.0)


def remap(a: np.ndarray, faces_f: tuple[np.ndarray, np.ndarray], faces_c: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Conservative (volume-weighted) remap of cell averages (..., nr, nth) from a fine to a coarse spherical grid;
    the grids need not nest. Volume = r^2 dr x sin(theta) dtheta, separable, so the remap is one matrix per axis.
    The coarse cells must lie inside the fine grid's extent (true for the forecast window)."""
    Or = _overlap(faces_f[0], faces_c[0], lambda r: r ** 3 / 3.0)
    Ot = _overlap(faces_f[1], faces_c[1], lambda t: -np.cos(t))
    num = np.einsum("...ik,ij,kl->...jl", a, Or, Ot)
    den = np.einsum("ij,kl->jl", Or, Ot)
    return num / den


def remap_r(a: np.ndarray, rf_f: np.ndarray, rf_c: np.ndarray) -> np.ndarray:
    """Remap of a radial profile (shell averages) by overlap length in log r."""
    ov = _overlap(np.log(rf_f), np.log(rf_c), lambda x: x)
    return (a @ ov) / ov.sum(0)


def volumes(grid) -> np.ndarray:
    rf, tf = grid["r_faces"], grid["theta_faces"]
    return 2 * math.pi * ((rf[1:] ** 3 - rf[:-1] ** 3) / 3.0)[:, None] * (np.cos(tf[:-1]) - np.cos(tf[1:]))[None, :]


def batch_mean_ci(x: np.ndarray, n_batches: int = 8) -> tuple[float, float]:
    """Mean and 95 % half-width from batch means (blocks long enough to be nearly independent)."""
    x = np.asarray(x, float)
    b = np.array_split(x, n_batches)
    m = np.array([bb.mean() for bb in b])
    from scipy import stats
    return float(x.mean()), float(stats.t.ppf(0.975, n_batches - 1) * m.std(ddof=1) / math.sqrt(n_batches))


def psd_slope(x: np.ndarray, dt: float, f_lo: float, f_hi: float) -> float:
    """Slope of the Welch power spectrum of x in log-log between f_lo and f_hi (1/M)."""
    from scipy import signal
    f, p = signal.welch(x - x.mean(), fs=1.0 / dt, nperseg=min(len(x), 256))
    k = (f >= f_lo) & (f <= f_hi) & (p > 0)
    return float(np.polyfit(np.log(f[k]), np.log(p[k]), 1)[0])


def run_stats(folder: str, t0: float, t1: float, codec_range: tuple[float, float]) -> dict:
    """Per-run statistics on the run's own grid (fields) and scalars."""
    run = load_run(folder)
    t, fr, mdot = run["t"], run["frames"], run["mdot"]
    k = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
    fr, tk, md = fr[k], t[k], mdot[k]
    g = run["grid"]
    vol = volumes(g)
    rho, vph = fr[:, 0].astype(float), fr[:, 3].astype(float)
    vr = fr[:, 1].astype(float)
    r, th = g["r"], g["theta"]
    R = r[:, None] * np.sin(th)[None, :]
    ell = R[None] * vph
    out = {"t": tk, "mdot": md, "vol": vol, "grid": g,
           # shell (theta-integrated) quantities, time means
           "rho_shell": (rho * vol).sum(-1).mean(0) / vol.sum(-1),
           "ell_shell": (rho * ell * vol).sum(-1).mean(0) / (rho * vol).sum(-1).mean(0),
           # advective angular-momentum flux through each shell (theta-integrated rho v_r l dA, time mean)
           "jflux": (rho * vr * ell * (r[:, None] ** 2) * (np.cos(g["theta_faces"][:-1]) - np.cos(g["theta_faces"][1:]))[None]
                     ).sum(-1).mean(0) * 2 * math.pi,
           "mass_window": None}
    lo, hi = codec_range
    out["x"] = ((np.log10(np.clip(rho, 1e-30, None)) - lo) / (hi - lo)).astype(np.float32)   # full grid
    out["mask"] = window_mask(g)
    out["faces"] = (g["r_faces"], g["theta_faces"])
    out["mass_window"] = (rho * vol * out["mask"]).sum((1, 2))
    return out


def field_scores(x: np.ndarray, mask: np.ndarray, every: float) -> dict:
    """Statistics of the normalised log-density field (frames ``every`` M apart, resampled to 20 M): time mean and
    temporal std (full grid, compared after remapping) and the persistence error in the window (pixel mean, as the
    forecaster is scored)."""
    s = max(int(round(FRAME_DT / every)), 1)
    x = x[::s]
    out = {"x_mean": x.mean(0), "x_std": x.std(0)}
    xw = x[:, mask]
    for lead_m in (20, 100, 1000):
        L = int(round(lead_m / FRAME_DT))
        out[f"persistence_{lead_m}"] = np.abs(xw[L:] - xw[:-L]).mean(1)      # time series of the error
    return out


def order_gci(f1: float, f2: float, f3: float, r: float, fs: float = 1.25) -> dict:
    """f1 finest. Observed order, Richardson extrapolation and GCI_fine (Roache); flags non-asymptotic cases."""
    e21, e32 = f1 - f2, f2 - f3
    res = {"fine": f1, "medium": f2, "coarse": f3, "e21": e21, "e32": e32}
    if e21 == 0 or e32 == 0 or e21 * e32 < 0:
        res.update(monotone=False, p=None, gci=None, extrapolated=None)
        return res
    p = math.log(abs(e32 / e21)) / math.log(r)
    res["monotone"] = True
    res["p"] = p
    if p <= 0:                 # differences grow under refinement: not converging
        res.update(gci=None, extrapolated=None)
        return res
    ext = f1 + e21 / (r ** p - 1)
    res["extrapolated"] = ext
    res["gci"] = fs * abs(e21 / f1) / (r ** p - 1) if f1 != 0 else None
    return res


def norm_diff(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float((np.abs(a - b) * w).sum() / w.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--levels", type=int, nargs="+", required=True, help="nr of each grid (ntheta = nr / 2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--t0", type=float, default=2000.0)
    ap.add_argument("--t1", type=float, default=10000.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    levels = sorted(a.levels)
    ratio = (levels[-1] / levels[0]) ** (1 / (len(levels) - 1))       # e.g. 128, 181, 256, 362: sqrt(2) to 0.03 %
    if any(abs(levels[i + 1] / levels[i] / ratio - 1) > 0.01 for i in range(len(levels) - 1)):
        raise SystemExit("levels must be in a constant refinement ratio")

    # one normalisation for every grid (the codec of the chosen production grid is fitted on its own training runs;
    # here a fixed range keeps the field statistics comparable): log10 rho in [-6, 0]
    codec_range = (-6.0, 0.0)
    stats: dict[int, list[dict]] = {}
    for n in levels:
        stats[n] = []
        for s in a.seeds:
            folder = os.path.join(a.runs, f"conv_n{n}_s{s}")
            if not os.path.exists(os.path.join(folder, "times.npy")):
                print(f"missing {folder}", flush=True)
                continue
            st = run_stats(folder, a.t0, a.t1, codec_range)
            every = float(np.diff(st["t"]).mean())
            st.update(field_scores(st.pop("x"), st["mask"], every))
            st["every"] = every
            stats[n].append(st)
            print(f"loaded {folder}: {len(st['t'])} frames, t {st['t'][0]:.0f}-{st['t'][-1]:.0f}", flush=True)

    coarse = levels[0]
    report: dict = {"levels": [[n, n // 2] for n in levels], "ratio": ratio, "seeds": a.seeds,
                    "window_M": [a.t0, a.t1], "scalars": {}, "profiles": {}, "fields": {}}

    # ------------------------------------------------------------ scalars, per level: mean over seeds +- CI
    def scalar_table(name, fn):
        rows = {}
        for n in levels:
            vals, cis = [], []
            for st in stats[n]:
                v, ci = fn(st)
                vals.append(v)
                cis.append(ci)
            if not vals:
                continue
            m = float(np.mean(vals))
            # combine the within-run batch-mean CI with the seed-to-seed spread
            seed_ci = (float(np.std(vals, ddof=1) / math.sqrt(len(vals)) * 4.30 if len(vals) == 3 else
                             np.std(vals, ddof=1) / math.sqrt(len(vals)) * 2.0) if len(vals) > 1 else 0.0)
            within = float(np.sqrt(np.mean(np.square(cis))) / math.sqrt(len(vals)))
            rows[n] = {"mean": m, "ci95": float(math.hypot(seed_ci, within)), "per_seed": vals}
        res = {"by_level": rows}
        if len(rows) >= 3:
            f3, f2, f1 = (rows[n]["mean"] for n in levels[-3:])
            res["richardson"] = order_gci(f1, f2, f3, ratio)
            res["finest_two_rel_diff"] = abs(f1 - f2) / abs(f1) if f1 else None
            # statistically converged if the two finest grids agree within their combined CI
            c1, c2 = rows[levels[-1]]["ci95"], rows[levels[-2]]["ci95"]
            res["statistically_converged"] = bool(abs(f1 - f2) <= math.hypot(c1, c2))
        report["scalars"][name] = res

    scalar_table("mdot_mean", lambda st: batch_mean_ci(st["mdot"]))
    scalar_table("mdot_rms_over_mean", lambda st: (float(st["mdot"].std() / st["mdot"].mean()), 0.0))
    scalar_table("mdot_psd_slope", lambda st: (psd_slope(st["mdot"], st["every"], 1 / 2000, 1 / 50), 0.0))
    scalar_table("window_mass_mean", lambda st: batch_mean_ci(st["mass_window"]))
    for lead in (20, 100, 1000):
        scalar_table(f"persistence_mae_{lead}M", lambda st, lead=lead: batch_mean_ci(st[f"persistence_{lead}"]))
    def rho_slope(st, lo=10.0, hi=60.0):
        r = st["grid"]["r"]
        k = (r >= lo) & (r <= hi)
        return float(np.polyfit(np.log(r[k]), np.log(st["rho_shell"][k]), 1)[0]), 0.0

    def ell_over_kepler(st, r0):
        r = st["grid"]["r"]
        i = int(np.argmin(np.abs(r - r0)))
        return float(st["ell_shell"][i] / (r[i] ** 1.5 / (r[i] - 2.0))), 0.0

    scalar_table("rho_slope_r10_60", rho_slope)
    scalar_table("ell_over_kepler_r10", lambda st: ell_over_kepler(st, 10.0))
    scalar_table("ell_over_kepler_r30", lambda st: ell_over_kepler(st, 30.0))
    scalar_table("field_temporal_std_mean", lambda st: (float(st["x_std"].mean()), 0.0))

    # ------------------------------------------------------------ profiles and fields on the coarse grid
    ref = stats[coarse][0]

    def restricted(st, key, n):
        a_ = st[key]
        if n == coarse:
            return a_
        if a_.ndim == 1:                                     # shell profiles
            return remap_r(a_, st["grid"]["r_faces"], ref["grid"]["r_faces"])
        return remap(a_, st["faces"], ref["faces"])

    if True:
        for key, label in (("rho_shell", "rho(r)"), ("ell_shell", "l(r)"), ("jflux", "angular-momentum flux"),
                           ("x_mean", "time-mean window field"), ("x_std", "temporal std of the window field")):
            prof = {n: np.mean([restricted(st, key, n) for st in stats[n]], 0) for n in levels if stats[n]}
            if len(prof) < 3:
                continue
            w = np.ones_like(prof[coarse]) if prof[coarse].ndim == 1 else ref["mask"].astype(float)   # pixel mean
            # differences between consecutive levels (L1, volume weighted for fields), relative to the finest
            scale = float((np.abs(prof[levels[-1]]) * w).sum() / w.sum())
            d_abs = [norm_diff(prof[levels[i + 1]], prof[levels[i]], w) for i in range(len(levels) - 1)]
            d = [v / scale for v in d_abs]
            seed_spread = None
            if len(stats[levels[-1]]) > 1:
                arr = np.array([restricted(st, key, levels[-1]) for st in stats[levels[-1]]])
                seed_spread = float((arr.std(0, ddof=1) * w).sum() / w.sum()) / scale
            p = math.log(d[-2] / d[-1]) / math.log(ratio) if len(d) >= 2 and d[-1] > 0 and d[-2] > 0 else None
            entry = {"label": label, "rel_L1_diff_consecutive": d, "abs_L1_diff_consecutive": d_abs, "observed_order": p,
                     "seed_spread_rel": seed_spread}
            if p is not None and p > 0:
                entry["gci_fine"] = 1.25 * d[-1] / (ratio ** p - 1)
            target = "fields" if prof[coarse].ndim == 2 else "profiles"
            report[target][key] = entry
            if prof[coarse].ndim == 1:
                entry["r"] = ref["grid"]["r"].tolist()
                entry["by_level"] = {n: prof[n].tolist() for n in prof}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: {kk: (vv.get("richardson") or vv) for kk, vv in v.items()} if k in ("scalars",) else v
                      for k, v in report.items() if k in ("scalars",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
