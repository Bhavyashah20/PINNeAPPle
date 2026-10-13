"""Latent surrogate with a trust card: error, speed and calibration against the full solver.

Full model: steady 2-D advection-diffusion-reaction of a pollutant released by a Gaussian source,

    -kappa * lap(u) + v . grad(u) + k u = f(x; xs),   u = 0 on the boundary of [0, 1]^2,

finite differences (upwind convection) on a 96 x 96 grid, sparse direct solve. Four design parameters: wind speed V,
wind direction theta, decay rate k, source position xs (kappa = 0.1, Peclet up to 40). The surrogate is a POD latent
model of the normalised fields with Gaussian-process latent regression (``ParametricPOD``) with
the checks of ``TrustedLatentSurrogate``: conformal error bound, design envelope, physics residual of the decoded
field (the same discrete operator, applied without solving).

The training envelope has a hole (a band of wind speeds and directions never simulated) and the test set has three
groups: inside the envelope, inside the hole (inside the parameter box: a box check passes them) and beyond the box.
Every surrogate answer is compared with a full solve.

    python examples/latent/trusted_latent_surrogate.py [--out results.json] [--n-train 150]

Runs in about a minute on a laptop CPU (NumPy/SciPy only).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr

from pinneapple_neural.architectures.rom import ParametricPOD, TrustedLatentSurrogate
from pinneapple_security.cps import roc_auc

N = 96
KAPPA = 0.1
LO = np.array([0.5, 0.0, 0.0, 0.35])                  # V, theta, k, xs
HI = np.array([4.0, np.pi / 2, 8.0, 0.65])
TOL = 0.02                                            # acceptable relative L2 error of the field


def _grid():
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return h, X, Y


H, XG, YG = _grid()
_I = sp.identity(N, format="csr")
_D2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N), format="csr") / H ** 2
_DB = sp.diags([-1.0, 1.0], [-1, 0], shape=(N, N), format="csr") / H        # backward difference
_DF = sp.diags([-1.0, 1.0], [0, 1], shape=(N, N), format="csr") / H         # forward difference
_LAP = sp.kron(_D2, _I) + sp.kron(_I, _D2)


def operator(p):
    """Discrete operator and right-hand side for parameters p = (V, theta, k, xs)."""
    V, th, k, xs = p
    vx, vy = V * np.cos(th), V * np.sin(th)
    dx = _DB if vx >= 0 else _DF                                         # upwind
    dy = _DB if vy >= 0 else _DF
    L = -KAPPA * _LAP + vx * sp.kron(dx, _I) + vy * sp.kron(_I, dy) + k * sp.identity(N * N)
    f = 100.0 * np.exp(-((XG - xs) ** 2 + (YG - 0.5) ** 2) / (2 * 0.05 ** 2))
    return L.tocsr(), f.ravel()


def solve(p):
    L, f = operator(p)
    return spsolve(L.tocsc(), f)


def residual(p, u):
    """Relative residual of the discrete equations on a field (no solve: one sparse mat-vec)."""
    L, f = operator(p)
    return float(np.linalg.norm(L @ u - f) / np.linalg.norm(f))


def in_hole(Ps):
    s = (Ps - LO) / (HI - LO)
    return (s[:, 0] - 0.55) ** 2 + (s[:, 1] - 0.5) ** 2 < 0.18 ** 2


def sample_envelope(n, rng):
    out = []
    while len(out) < n:
        P = LO + (HI - LO) * rng.random((4 * n, 4))
        out.extend(P[~in_hole(P)])
    return np.array(out[:n])


def sample_hole(n, rng):
    out = []
    while len(out) < n:
        P = LO + (HI - LO) * rng.random((40 * n, 4))
        out.extend(P[in_hole(P)])
    return np.array(out[:n])


def sample_beyond(n, rng):
    """One parameter 5-30 % of its range beyond the box (wind speed, decay or source position)."""
    P = LO + (HI - LO) * rng.random((n, 4))
    j = rng.choice([0, 2, 3], size=n)
    over = (HI - LO)[j] * rng.uniform(0.05, 0.3, n)
    up = rng.random(n) < 0.5
    up[j == 0] = True                                                     # keep V > 0
    up[j == 2] = True                                                     # keep k >= 0
    P[np.arange(n), j] = np.where(up, HI[j] + over, LO[j] - over)
    return P


def run(n_train=150, n_cal=100, n_test=150, seed=0, alpha=0.1):
    rng = np.random.default_rng(seed)
    P_tr, P_cal, P_in = (sample_envelope(m, rng) for m in (n_train, n_cal, n_test))
    P_hole, P_out = sample_hole(n_test, rng), sample_beyond(n_test, rng)
    t0 = time.perf_counter()
    X_tr = np.array([solve(p) for p in P_tr])
    t_solve = (time.perf_counter() - t0) / n_train
    X_cal = np.array([solve(p) for p in P_cal])
    groups = {"envelope": P_in, "hole": P_hole, "beyond": P_out}
    truth = {g: np.array([solve(p) for p in P]) for g, P in groups.items()}

    sur = TrustedLatentSurrogate(ParametricPOD(r=64, energy=0.99999, regressor="gpr", normalize=True),
                                 alpha=alpha, residual=residual, seed=seed).fit(P_tr, X_tr, P_cal, X_cal)
    nearest = ParametricPOD(r=64, energy=0.99999, regressor="nearest", normalize=True).fit(P_tr, X_tr)

    t0 = time.perf_counter()
    for _ in range(5):
        sur.predict(P_in)
    t_pred_batch = (time.perf_counter() - t0) / (5 * len(P_in))
    t0 = time.perf_counter()
    for p in P_in[:50]:
        sur.predict(p[None])
    t_pred_one = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    sur.assess(P_in[:50], TOL)
    t_assess = (time.perf_counter() - t0) / 50

    rows, allv = {}, {k: [] for k in ("err", "bound", "p", "res", "dec", "box", "group")}
    for g, P in groups.items():
        a = sur.assess(P, TOL)
        X = truth[g]
        err = np.linalg.norm(a["X"] - X, axis=1) / np.linalg.norm(X, axis=1)
        err_nn = nearest.error(P, X)
        dec = a["decision"]
        rows[g] = {
            "n": len(P),
            "error_median": float(np.median(err)), "error_p95": float(np.quantile(err, 0.95)),
            "nearest_error_median": float(np.median(err_nn)),
            "fraction_error_above_tol": float(np.mean(err > TOL)),
            "bound_coverage": float(np.mean(err <= a["error_bound"])),
            "approved": float(np.mean(dec == "APPROVED")), "review": float(np.mean(dec == "REVIEW")),
            "rejected": float(np.mean(dec == "REJECT")),
            "approved_but_wrong": float(np.mean((dec == "APPROVED") & (err > TOL))),
        }
        inside_box = np.all((P >= LO) & (P <= HI), axis=1)
        for k, v in (("err", err), ("bound", a["error_bound"]), ("p", a["envelope_p"]), ("res", a["residual"]),
                     ("dec", dec), ("box", inside_box), ("group", np.full(len(P), g))):
            allv[k].append(v)
    v = {k: np.concatenate(x) for k, x in allv.items()}
    wrong = v["err"] > TOL
    ood = v["group"] != "envelope"
    gate_ok = v["dec"] == "APPROVED"
    box_ok = v["box"]
    summary = {
        "full_solve_s": t_solve, "surrogate_batch_s": t_pred_batch, "surrogate_single_s": t_pred_one,
        "surrogate_with_checks_s": t_assess,
        "speedup_batch": t_solve / t_pred_batch, "speedup_single": t_solve / t_pred_one,
        "speedup_with_checks": t_solve / t_assess,
        "rank": sur.rom.rank, "conformal_q": sur.calibration_["q"],
        "auc_ood": {"box_check": roc_auc(~v["box"], ood), "envelope_pvalue": roc_auc(-v["p"], ood),
                    "physics_residual": roc_auc(v["res"], ood), "error_bound": roc_auc(v["bound"], ood)},
        "auc_error_above_tol": {"box_check": roc_auc(~v["box"], wrong), "envelope_pvalue": roc_auc(-v["p"], wrong),
                                "physics_residual": roc_auc(v["res"], wrong),
                                "error_bound": roc_auc(v["bound"], wrong)},
        "spearman_bound_error": float(spearmanr(v["bound"], v["err"]).statistic),
        "spearman_residual_error": float(spearmanr(v["res"], v["err"]).statistic),
        "gate": {"approved": int(gate_ok.sum()), "approved_but_wrong": int((gate_ok & wrong).sum()),
                 "wrong_total": int(wrong.sum()), "max_error_approved": float(v["err"][gate_ok].max())
                 if gate_ok.any() else None},
        "box_gate": {"approved": int(box_ok.sum()), "approved_but_wrong": int((box_ok & wrong).sum()),
                     "max_error_approved": float(v["err"][box_ok].max())},
    }
    return {"config": {"grid": N, "n_train": n_train, "n_cal": n_cal, "n_test_per_group": n_test, "seed": seed,
                       "kappa": KAPPA, "alpha": alpha, "tol": TOL}, "groups": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = run(n_train=a.n_train, seed=a.seed)
    s = r["summary"]
    print(f"POD rank {s['rank']}; full solve {s['full_solve_s'] * 1e3:.1f} ms; surrogate {s['surrogate_single_s'] * 1e3:.2f}"
          f" ms ({s['speedup_single']:.0f}x), with checks {s['surrogate_with_checks_s'] * 1e3:.2f} ms "
          f"({s['speedup_with_checks']:.0f}x)")
    print(f"{'group':10s} {'median err':>10s} {'p95 err':>8s} {'nearest':>8s} {'>tol':>6s} {'cover':>6s} "
          f"{'APPR':>6s} {'REVIEW':>6s} {'REJECT':>6s} {'appr&wrong':>10s}")
    for g, row in r["groups"].items():
        print(f"{g:10s} {row['error_median']:10.4f} {row['error_p95']:8.4f} {row['nearest_error_median']:8.4f} "
              f"{row['fraction_error_above_tol']:6.2f} {row['bound_coverage']:6.2f} {row['approved']:6.2f} "
              f"{row['review']:6.2f} {row['rejected']:6.2f} {row['approved_but_wrong']:10.3f}")
    print("AUC out-of-envelope:", {k: round(v, 3) for k, v in s["auc_ood"].items()})
    print("AUC error > tol:   ", {k: round(v, 3) for k, v in s["auc_error_above_tol"].items()})
    print("trust gate:", s["gate"], " box check:", s["box_gate"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
