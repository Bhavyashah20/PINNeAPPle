"""Heat-sink surrogate trained with monotonicity constraints (zoo model ``heatsink_natural_mlp``).

The app's unconstrained natural-convection surrogate fits the physics engine to 0.7 % but gets the sign of the
conductivity effect wrong on 5 % of held-out designs: raising k by 5 % lowers the engine's resistance by only 0.13 %
(median over 400 designs), below the fit error, so a plain regression does not resolve it. :func:`train_monotone`
trains the same architecture (``bench_mlp`` 4 x 128 SiLU, same features, same split) with a hinge penalty
(:func:`pinneapple_neural.trainer.monotonic.monotonicity_penalty`) on d log R / d log k and d log R / d log(source
width), both required to be <= -``MARGIN``, evaluated on the training designs and on random in-range designs; Adam
first, then a full-batch L-BFGS polish of the same constrained objective.

Power is not constrained: the engine's own resistance rises with power on 0.5 % of designs (up to +0.7 % for a +5 %
step), so "more power, lower resistance" is not a law of the reference.
"""
from __future__ import annotations

import gzip
import json
import math
import time

import numpy as np
import torch

from .surrogate import (
    ARCH,
    COMMON_RANGES,
    FEATURES,
    MODE_RANGES,
    _build,
    _dataset_path,
    features,
    n_fins_for,
    rows_to_xy,
)

__all__ = ["MONOTONE", "MARGIN", "LAMBDA", "random_designs", "split_indices", "train_monotone"]

MONOTONE = {"natural": {"log_k": -1, "log_fw": -1}}   # required sign of d log R / d feature
# required |d log R / d log x|; the engine's smallest effects over 400 probed designs are 1.6e-3 (k) and 9e-4 (source
# width) per unit log
MARGIN = 5e-4
LAMBDA = 100.0        # weight of the hinge (physical log units) against the standardised MSE
WARMUP = 50           # Adam epochs of plain regression before the penalty is switched on


def random_designs(mode: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Feature rows of random in-range designs, sampled log-uniformly like the dataset."""
    r = {**COMMON_RANGES, **MODE_RANGES[mode]}
    lu = lambda k: np.exp(rng.uniform(*np.log(r[k]), n))  # noqa: E731
    W, D, tb, t, H, k = (lu(c) for c in ("base_width_mm", "base_depth_mm", "base_thickness_mm",
                                          "fin_thickness_mm", "fin_height_mm", "k_w_mk"))
    N = n_fins_for(W, lu("fin_gap_mm"), t)
    fw, fd = lu("source_frac_w"), lu("source_frac_d")
    Ta = rng.uniform(*r["t_ambient_c"], n)
    Q = lu("power_w")
    V = lu("air_velocity_m_s") if mode == "forced" else None
    X = features(mode, W, D, tb, N, t, H, k, fw * W, fd * D, Ta, Q, V)
    return X[(W - N * t) / np.maximum(N - 1, 1) > 0]


def split_indices(n: int) -> dict[str, np.ndarray]:
    """The app's split: numpy default_rng(0).permutation(n): test 15 %, cal 10 %, val 10 %, train the rest."""
    idx = np.random.default_rng(0).permutation(n)
    n_te, n_cal, n_val = int(0.15 * n), int(0.10 * n), int(0.10 * n)
    return {"test": idx[:n_te], "cal": idx[n_te:n_te + n_cal], "val": idx[n_te + n_cal:n_te + n_cal + n_val],
            "train": idx[n_te + n_cal + n_val:]}


def train_monotone(mode: str = "natural", *, seed: int = 0, epochs: int = 600, lbfgs_rounds: int = 4,
                   lbfgs_steps: int = 200, n_random: int = 20000, lam: float = LAMBDA, margin: float = MARGIN,
                   log=print) -> dict:
    """Train the constrained surrogate on the app's dataset; returns the bundle that ``torch.save`` stores (state,
    normalisation, split-conformal quantiles of log R, history, summary metrics)."""
    from pinneapple_neural.trainer.lbfgs_finetune import multi_round_lbfgs
    from pinneapple_neural.trainer.monotonic import monotonicity_penalty

    torch.manual_seed(seed)
    rows = json.load(gzip.open(_dataset_path(mode)))
    X, Y = rows_to_xy(mode, rows)
    Y = Y[:, :1]                                                   # log R only
    sp = split_indices(len(X))
    tr, va, cal, te = sp["train"], sp["val"], sp["cal"], sp["test"]
    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
    y_mean, y_std = Y[tr].mean(0), Y[tr].std(0) + 1e-8
    zx = lambda a: torch.as_tensor((a - x_mean) / x_std, dtype=torch.float32)  # noqa: E731
    zy = lambda a: torch.as_tensor((a - y_mean) / y_std, dtype=torch.float32)  # noqa: E731
    mono = MONOTONE[mode]
    cols = [FEATURES[mode].index(c) for c in mono]
    signs = [mono[c] for c in mono]
    scale = (y_std[0] / x_std[cols]).astype(np.float32)          # d log R / d raw feature from standardised units

    model = _build(X.shape[1], 1)
    out_of = lambda z: (lambda y: y.y if hasattr(y, "y") else y)(model(z))  # noqa: E731
    pen = lambda z, cg=True: monotonicity_penalty(model, z, cols, signs, scale=scale, margin=margin,  # noqa: E731
                                                  create_graph=cg)

    rng = np.random.default_rng(seed + 1)
    xtr, ytr, xva, yva = zx(X[tr]), zy(Y[tr]), zx(X[va]), zy(Y[va])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-5)
    best, best_state, hist = math.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        w = lam if ep >= WARMUP else 0.0
        model.train()
        perm = torch.randperm(len(xtr))
        for b in range(0, len(perm), 128):
            j = perm[b:b + 128]
            loss = torch.mean((out_of(xtr[j]) - ytr[j]) ** 2)
            if w:
                loss = loss + w * (pen(xtr[j]) + pen(zx(random_designs(mode, 128, rng))))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(torch.mean((out_of(xva) - yva) ** 2))
        p = float(pen(xva, False))
        hist.append({"epoch": ep, "val_mse": v, "val_mono_penalty": p})
        if w and v + lam * p < best:
            best, best_state = v + lam * p, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 50 == 0:
            log(f"epoch {ep} val_mse {v:.3e} mono {p:.3e} {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)

    # full-batch L-BFGS polish of the constrained objective; fresh random designs every round
    def closure_builder():
        zr = zx(random_designs(mode, n_random, rng))

        def closure():
            return torch.mean((out_of(xtr) - ytr) ** 2) + lam * (pen(xtr) + pen(zr))
        return closure

    lb = multi_round_lbfgs(list(model.parameters()), closure_builder, n_rounds=lbfgs_rounds,
                           n_steps_per_round=lbfgs_steps)
    model.eval()
    with torch.no_grad():
        v = float(torch.mean((out_of(xva) - yva) ** 2))
    log(f"after L-BFGS: val_mse {v:.3e} mono {float(pen(xva, False)):.3e} rounds {list(lb.round_losses)}")

    def pred_log(a):
        with torch.no_grad():
            return out_of(zx(a)).numpy()[:, 0] * y_std[0] + y_mean[0]

    res = np.abs(pred_log(X[cal]) - Y[cal, 0])
    m = len(res)
    q = {str(c): float(np.quantile(res, min(1.0, math.ceil((m + 1) * c / 100) / m))) for c in (90, 95)}
    e = 100 * np.abs(np.exp(pred_log(X[te]) - Y[te, 0]) - 1)
    summary = {"test_R_mean_abs_pct": float(e.mean()), "test_R_p95_abs_pct": float(np.percentile(e, 95)),
               "train_seconds": time.time() - t0, "epochs": epochs, "lbfgs_round_losses": list(lb.round_losses),
               "margin": margin, "lambda": lam, "warmup": WARMUP}
    log(json.dumps(summary))
    return {"mode": mode, "state": model.state_dict(), "arch": ARCH, "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std, "conformal_log_q": q, "history": hist, "summary": summary,
            "monotone": mono, "split": {k: len(v) for k, v in sp.items()}}
