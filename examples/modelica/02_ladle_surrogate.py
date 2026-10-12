"""Surrogate of a Modelica model: hot-metal ladle cooling, from an FMU sweep to a neural network.

`models/LadleCooling.mo` (melt, refractory lining and steel shell as three lumped masses; conduction through the
lining, radiation from the slag-covered melt surface, convection and radiation from the shell) is compiled by
OpenModelica into an FMI 2.0 Co-Simulation FMU. 80 Latin-hypercube runs over five uncertain inputs (lining
conductivity and thickness, surface emissivity, tapping temperature, shell convection) give 4-hour trajectories;
64 train an MLP T(params, t), 16 are held out as unseen parameter combinations.

Checks on the held-out runs:
* error of the melt and shell temperatures (K) and of the time the melt takes to fall to 1350 °C;
* the network against a lookup baseline (the nearest training run in normalised parameter space);
* physics: the melt never warms up (dT_melt/dt <= 0) in the network's predictions.

Needs OpenModelica (omc, or Docker with openmodelica/openmodelica) and fmpy. A few minutes on 4 cores (the 80 FMU runs are cached in _out/).
Writes docs/assets/modelica/ladle_surrogate.png and examples/modelica/_out/ladle_metrics.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinneapple_simulation.external_solvers.modelica.omc import build_fmu
from pinneapple_simulation.external_solvers.modelica.sweep import FMUDataset, fmu_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "modelica")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(2)

RANGES = {"k_lin": (1.5, 4.0), "d_lin": (0.15, 0.35), "eps_top": (0.2, 0.7), "T0": (1723.15, 1823.15),
          "h_out": (5.0, 20.0)}
OUTPUTS = ["T_melt", "T_shell"]
T_LIMIT = 1623.15                                   # 1350 °C
data = os.path.join(OUT, "ladle_sweep.npz")
if os.path.exists(data):
    ds = FMUDataset.load(data)
else:
    fmu = build_fmu(os.path.join(HERE, "models", "LadleCooling.mo"), "LadleCooling", OUT)
    ds = fmu_sweep(fmu, RANGES, n=80, stop_time=4 * 3600, output_interval=120, outputs=OUTPUTS, seed=1,
                   step_size=10.0)
    ds.save(data)
train, test = ds.split(16, seed=0)
lo = np.array([ds.bounds[n][0] for n in ds.names])
hi = np.array([ds.bounds[n][1] for n in ds.names])
t_n = ds.time / ds.time[-1]
y_mu = np.array([train.outputs[k].mean() for k in OUTPUTS])
y_sd = np.array([train.outputs[k].std() for k in OUTPUTS])


def features(params):
    p = (params - lo) / (hi - lo)
    P = np.repeat(p, len(t_n), axis=0)
    T = np.tile(t_n, len(p))[:, None]
    return torch.tensor(np.hstack([P, T]), dtype=torch.float32)


def targets(d):
    return torch.tensor((np.stack([d.outputs[k].ravel() for k in OUTPUTS], 1) - y_mu) / y_sd, dtype=torch.float32)


net = torch.nn.Sequential(torch.nn.Linear(len(ds.names) + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                          torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2))
X, Y = features(train.params), targets(train)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
for _it in range(3000):
    opt.zero_grad()
    loss = torch.mean((net(X) - Y) ** 2)
    loss.backward()
    opt.step()
    sched.step()
print(f"train loss {loss.item():.2e}")


def predict(params):
    with torch.no_grad():
        y = net(features(params)).numpy() * y_sd + y_mu
    return {k: y[:, i].reshape(len(params), -1) for i, k in enumerate(OUTPUTS)}


def time_to(T, limit):
    out = []
    for row in T:
        below = np.nonzero(row <= limit)[0]
        if len(below) == 0:
            out.append(np.nan)
            continue
        j = below[0]
        out.append(np.interp(limit, [row[j], row[j - 1]], [ds.time[j], ds.time[j - 1]]) if j else ds.time[0])
    return np.array(out)


pred = predict(test.params)
pn = lambda p: (p - lo) / (hi - lo)                                                     # noqa: E731
nn_idx = np.argmin(((pn(test.params)[:, None] - pn(train.params)[None]) ** 2).sum(-1), axis=1)
look = {k: train.outputs[k][nn_idx] for k in OUTPUTS}
m = {"runs": {"train": len(train.params), "test": len(test.params)}, "time_points": len(ds.time)}
for k in OUTPUTS:
    e, b = pred[k] - test.outputs[k], look[k] - test.outputs[k]
    m[k] = {"rmse_K": float(np.sqrt((e ** 2).mean())), "max_abs_K": float(np.abs(e).max()),
            "baseline_rmse_K": float(np.sqrt((b ** 2).mean()))}
tt, tp, tb = time_to(test.outputs["T_melt"], T_LIMIT), time_to(pred["T_melt"], T_LIMIT), time_to(
    look["T_melt"], T_LIMIT)
ok = np.isfinite(tt) & np.isfinite(tp)
m["time_to_1350C"] = {"runs_reaching": int(np.isfinite(tt).sum()),
                      "mae_min": float(np.nanmean(np.abs(tp - tt)[ok]) / 60) if ok.any() else None,
                      "baseline_mae_min": float(np.nanmean(np.abs(tb - tt)[ok & np.isfinite(tb)]) / 60)
                      if (ok & np.isfinite(tb)).any() else None}
dT = np.diff(pred["T_melt"], axis=1)
m["physics"] = {"melt_warming_steps": int((dT > 0.05).sum()), "steps_checked": int(dT.size)}
print(json.dumps(m, indent=1))
json.dump(m, open(os.path.join(OUT, "ladle_metrics.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
h = ds.time / 3600
for i in range(4):
    c = f"C{i}"
    for a, k in zip(ax, OUTPUTS, strict=True):
        a.plot(h, test.outputs[k][i] - 273.15, color=c, lw=2, label="FMU (OpenModelica)" if i == 0 else None)
        a.plot(h, pred[k][i] - 273.15, "--", color=c, lw=1.5, label="neural surrogate" if i == 0 else None)
ax[0].axhline(T_LIMIT - 273.15, color="0.5", lw=0.8, ls=":")
ax[0].set_title(f"Melt temperature, 4 unseen runs (RMSE {m['T_melt']['rmse_K']:.1f} K)")
ax[1].set_title(f"Shell temperature (RMSE {m['T_shell']['rmse_K']:.1f} K)")
for a in ax:
    a.set_xlabel("time [h]")
    a.set_ylabel("°C")
    a.grid(alpha=0.3)
ax[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "ladle_surrogate.png"), dpi=110)
print("wrote", os.path.join(ASSETS, "ladle_surrogate.png"))
