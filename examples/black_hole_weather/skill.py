"""Skill of the black-hole weather forecaster against persistence, climatology and DMD, with confidence intervals.

    python examples/black_hole_weather/skill.py --dataset blackhole_riaf_density_256 --models runs/bh256_res runs/bh256_roll \\
        --split test --out runs/bh256_skill

* Rollouts of ``--blocks`` blocks (5 frames = 100 M each) from start times every ``--start-every`` frames of every
  trajectory of the split (whole trajectories never seen in training).
* Baselines on the same starts: **persistence** (last input frame), **climatology** (time mean of the training
  trajectories) and **DMD** (exact dynamic mode decomposition of the training trajectories, affine, rank chosen on
  the validation trajectory), iterated from the last input frame.
* Error: MAE of the normalised log density in the window (the forecaster's scale). For each lead, the mean
  difference model - persistence and the skill score 1 - MAE_model / MAE_persistence get 95 % intervals from a
  moving-block bootstrap over start times (blocks of 1000 M, since nearby starts are correlated). The **skill
  horizon** is the last lead up to which the upper bound of (model - persistence) stays below zero.
* Physics checks without ground truth (trust report): mass in the window changes no faster than in training
  (``MassEnvelope``), positivity and range of the forecast density, and the inward mass transport implied by the
  forecast (the net mass flow through shells, which outward angular-momentum transport drives) against the truth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoo_data import load_trajectories, window_volumes  # noqa: E402

from pinneapple_physics.blackhole.forecast import MassEnvelope, load_forecaster, rollout  # noqa: E402

K = 5


# ---------------------------------------------------------------------------------------------- baselines
class DMD:
    """Affine exact DMD on frames: x_{t+1} - mu = U A U^T (x_t - mu), rank r."""

    def __init__(self, trajs: list[np.ndarray], rank: int):
        flat = [t.reshape(len(t), -1).astype(np.float64) for t in trajs]
        self.mu = np.concatenate(flat).mean(0)
        X0 = np.concatenate([f[:-1] - self.mu for f in flat]).T
        X1 = np.concatenate([f[1:] - self.mu for f in flat]).T
        U, s, Vt = np.linalg.svd(X0, full_matrices=False)
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank]
        self.U = U
        self.A = U.T @ X1 @ Vt.T / s[None, :]
        self.shape = trajs[0].shape[1:]
        self.rank = rank

    def forecast(self, last: np.ndarray, n: int) -> np.ndarray:
        z = self.U.T @ (last.reshape(-1) - self.mu)
        res = np.empty((n, *self.shape), np.float32)
        x_perp = last.reshape(-1) - self.mu - self.U @ z        # the part DMD does not model is kept frozen
        for i in range(n):
            z = self.A @ z
            res[i] = (self.mu + self.U @ z + x_perp).reshape(self.shape)
        return res


# ---------------------------------------------------------------------------------------------- statistics
def block_bootstrap(d: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of d (starts x ...) and its 95 % interval from a moving-block bootstrap over the first axis."""
    rng = np.random.default_rng(seed)
    n = len(d)
    block = max(1, min(block, n))
    nb = math.ceil(n / block)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, nb)])[:n]
        means[b] = d[idx].mean()
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ratio_bootstrap(a: np.ndarray, b: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0):
    """Skill score 1 - mean(a)/mean(b) and its 95 % interval (paired moving-block bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    block = max(1, min(block, n))
    nb = math.ceil(n / block)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, nb)])[:n]
        vals[i] = 1 - a[idx].mean() / b[idx].mean()
    return float(1 - a.mean() / b.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------------------------- physics
def window_mass(x: np.ndarray, codec, vol: np.ndarray) -> np.ndarray:
    return (10.0 ** (x.astype(np.float64) * (codec.log_max - codec.log_min) + codec.log_min) * vol).sum((-2, -1))


def implied_inflow(x: np.ndarray, codec, vol: np.ndarray, dt: float, radii_idx: list[int]) -> np.ndarray:
    """Net inward mass flow through the shells at ``radii_idx`` implied by the change of the enclosed mass,
    -(dM(<r)/dt), per frame step (frames, len(radii_idx)); ignores what crosses the polar edges of the window."""
    rho = 10.0 ** (x.astype(np.float64) * (codec.log_max - codec.log_min) + codec.log_min)
    m_r = np.cumsum((rho * vol).sum(-1), -1)[:, radii_idx]          # enclosed mass M(< r_i) per frame
    return -np.diff(m_r, axis=0) / dt


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="blackhole_riaf_density_256", help="zoo name or path to data.npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--start-every", type=int, default=5)
    ap.add_argument("--dmd-ranks", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    path = a.dataset if a.dataset.endswith(".npz") else None

    models = {os.path.basename(os.path.normpath(m)): load_forecaster(os.path.join(m, "forecaster.pt")) for m in a.models}
    codec = next(iter(models.values()))[1]
    for name, (_, c, _) in models.items():
        if (c.log_min, c.log_max) != (codec.log_min, codec.log_max):
            raise SystemExit(f"{name}: different normalisation than {next(iter(models))}")

    def enc(v):
        return ((v["log10_rho"] - codec.log_min) / (codec.log_max - codec.log_min)).astype(np.float32)

    train = [enc(v) for v in load_trajectories("train", path).values()]
    val = [enc(v) for v in load_trajectories("val", path).values()]
    test_raw = load_trajectories(a.split, path)
    test = {s: enc(v) for s, v in test_raw.items()}
    dt = float(np.diff(next(iter(test_raw.values()))["t"]).mean())
    vol = window_volumes(path)
    clim = np.concatenate(train).mean(0)
    n_lead = a.blocks * K

    # DMD rank on the validation trajectory (mean MAE over the rollout)
    def starts_of(x):
        return list(range(K - 1, len(x) - n_lead, a.start_every))

    dmd_scores = {}
    for r in a.dmd_ranks:
        d = DMD(train, r)
        errs = [np.abs(d.forecast(x[s], n_lead) - x[s + 1:s + 1 + n_lead]).mean() for x in val for s in starts_of(x)]
        dmd_scores[r] = float(np.mean(errs))
    best_rank = min(dmd_scores, key=dmd_scores.get)
    dmd = DMD(train, best_rank)
    print(f"DMD validation MAE by rank {dmd_scores} -> rank {best_rank}", flush=True)

    env = MassEnvelope.fit([window_mass(x, codec, vol) for x in train], dt)
    shells = [vol.shape[0] // 4, vol.shape[0] // 2]                 # 256 x 128 grid: r = 9.5 and 22.6 M

    errors = {n: [] for n in ["persistence", "climatology", "dmd", *models]}
    mass_err = {n: [] for n in ["persistence", "dmd", *models]}
    phys = {n: {"envelope_violations": 0, "frames": 0, "min_x": 1e9, "max_x": -1e9, "nonfinite": 0,
                "inflow_pred": [], "inflow_true": []} for n in models}
    start_index = []
    for seed, x in test.items():
        for s in starts_of(x):
            truth = x[s + 1:s + 1 + n_lead]
            last = x[s]
            start_index.append((seed, s))
            fc = {"persistence": np.repeat(last[None], n_lead, 0), "climatology": np.repeat(clim[None], n_lead, 0),
                  "dmd": dmd.forecast(last, n_lead)}
            for name, (model, _, _) in models.items():
                fc[name] = rollout(model, x[s - K + 1:s + 1], a.blocks)
            m_true = window_mass(truth, codec, vol)
            for n, f in fc.items():
                errors[n].append(np.abs(f - truth).mean((1, 2)))
                if n in mass_err:
                    mass_err[n].append(np.abs(np.log(window_mass(f, codec, vol) / m_true)))
            m0 = float(window_mass(last[None], codec, vol)[0])
            infl_true = implied_inflow(np.concatenate([last[None], truth]), codec, vol, dt, shells).mean(0)
            for name in models:
                f = fc[name]
                p = phys[name]
                p["nonfinite"] += int((~np.isfinite(f)).sum())
                p["min_x"], p["max_x"] = min(p["min_x"], float(f.min())), max(p["max_x"], float(f.max()))
                v = env.violations(window_mass(f, codec, vol), dt, m0)
                p["envelope_violations"] += int(v.sum())
                p["frames"] += len(v)
                p["inflow_pred"].append(implied_inflow(np.concatenate([last[None], f]), codec, vol, dt, shells).mean(0))
                p["inflow_true"].append(infl_true)
    errors = {n: np.array(v) for n, v in errors.items()}           # (starts, leads)
    mass_err = {n: np.array(v) for n, v in mass_err.items()}
    block = max(1, int(round(1000.0 / (a.start_every * dt))))
    leads = dt * (1 + np.arange(n_lead))

    report = {"split": a.split, "n_starts": len(start_index), "trajectories": sorted(test), "frame_dt": dt,
              "leads_M": leads.tolist(), "bootstrap_block_starts": block, "dmd_rank": best_rank,
              "dmd_val_mae_by_rank": dmd_scores, "mean_mae": {n: e.mean(0).tolist() for n, e in errors.items()},
              "models": {}}
    for name in models:
        diff = errors[name] - errors["persistence"]
        ci = [block_bootstrap(diff[:, j], block, seed=j) for j in range(n_lead)]
        ss = [ratio_bootstrap(errors[name][:, j], errors["persistence"][:, j], block, seed=j) for j in range(n_lead)]
        better = [c[2] < 0 for c in ci]
        horizon = float(leads[np.argmin(better)] - dt) if not all(better) else float(leads[-1])
        horizon_block = None                                      # judged on 100 M block means, as the README did
        nbk = n_lead // K
        db = diff.reshape(len(diff), nbk, K).mean(2)
        ci_b = [block_bootstrap(db[:, j], block, seed=100 + j) for j in range(nbk)]
        ok_b = [c[2] < 0 for c in ci_b]
        horizon_block = float(K * dt * (np.argmin(ok_b) if not all(ok_b) else nbk))
        comp = {}
        for base in ("climatology", "dmd"):
            dd = errors[name] - errors[base]
            cc = [block_bootstrap(dd[:, j], block, seed=200 + j) for j in range(n_lead)]
            okc = [c[2] < 0 for c in cc]
            comp[base] = {"diff_ci": cc, "horizon_M": float(leads[np.argmin(okc)] - dt) if not all(okc) else float(leads[-1])}
        p = phys[name]
        ip, it = np.array(p["inflow_pred"]), np.array(p["inflow_true"])
        report["models"][name] = {
            "diff_vs_persistence_ci": ci, "skill_score_ci": ss, "skill_horizon_M": horizon,
            "skill_horizon_blocks_M": horizon_block, "vs": comp,
            "mass_log_error_mean": mass_err[name].mean(0).tolist(),
            "mass_log_error_persistence": mass_err["persistence"].mean(0).tolist(),
            "physics": {
                "mass_envelope_violation_fraction": p["envelope_violations"] / max(p["frames"], 1),
                "envelope_rate_max": env.rate_max,
                "nonfinite_values": p["nonfinite"],
                "x_range_forecast": [p["min_x"], p["max_x"]],
                "density_positive": True,                         # 10**x > 0 by construction; range checked above
                "inflow_shell_radius_index": shells,
                "inflow_sign_agreement": float((np.sign(ip) == np.sign(it)).mean()),
                "inflow_corr": [float(np.corrcoef(ip[:, i], it[:, i])[0, 1]) for i in range(ip.shape[1])],
                "inflow_mean_pred": ip.mean(0).tolist(), "inflow_mean_true": it.mean(0).tolist()},
        }
        print(f"{name}: skill horizon {horizon:.0f} M (frames), {horizon_block:.0f} M (100 M blocks); "
              f"SS at 100 M {ss[K - 1][0]:+.3f} [{ss[K - 1][1]:+.3f}, {ss[K - 1][2]:+.3f}], "
              f"at {leads[-1]:.0f} M {ss[-1][0]:+.3f} [{ss[-1][1]:+.3f}, {ss[-1][2]:+.3f}]", flush=True)
    for n, e in errors.items():
        print(f"  MAE {n:12s} 100 M {e[:, K - 1].mean():.4f}  1000 M {e[:, min(50, n_lead) - 1].mean():.4f}  "
              f"{leads[-1]:.0f} M {e[:, -1].mean():.4f}", flush=True)
    with open(os.path.join(a.out, f"skill_{a.split}.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    main()
