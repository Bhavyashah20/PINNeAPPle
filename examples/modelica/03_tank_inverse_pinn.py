"""Inverse problem on a Modelica model: identify a tank's discharge coefficient with a physics-informed network.

`models/DrainingTank.mo` (A dh/dt = q_in - Cd a sqrt(2 g h)) is compiled to an FMU and simulated with the true
Cd = 0.62. The "measurements" are 21 level readings over the first 300 s with 5 mm noise. A PINN learns h(t) and
Cd together: it fits the readings and the tank's mass balance at 200 collocation points over 0-900 s, so the
balance also carries the solution past the last reading.

Compared with:
* finite differences: Cd from (q_in - A dh/dt) / (a sqrt(2 g h)) on the noisy readings, averaged;
* a data-only network (no physics), which cannot give Cd and is checked on the extrapolation to 900 s.

Needs OpenModelica (omc, or Docker with openmodelica/openmodelica) and fmpy. About a minute.
Writes docs/assets/modelica/tank_inverse_pinn.png and examples/modelica/_out/tank_metrics.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinneapple_simulation.external_solvers.modelica import FMUSimConfig, simulate_fmu
from pinneapple_simulation.external_solvers.modelica.omc import build_fmu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "modelica")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)
torch.manual_seed(0)
torch.set_num_threads(1)
rng = np.random.default_rng(0)

A, a, qin, g, CD_TRUE = 2.0, 0.01, 0.01, 9.81, 0.62
T_DATA, T_END, NOISE = 300.0, 900.0, 0.005
fmu = build_fmu(os.path.join(HERE, "models", "DrainingTank.mo"), "DrainingTank", OUT)
ref = simulate_fmu(fmu, FMUSimConfig(stop_time=T_END, output_interval=5.0, output_variables=["level"],
                                     parameters={"Cd": CD_TRUE, "A": A, "a": a, "qin": qin, "h0": 1.0}))
t_ref, h_ref = ref["time"], ref["level"]
t_obs = np.linspace(0, T_DATA, 21)
h_obs = np.interp(t_obs, t_ref, h_ref) + NOISE * rng.standard_normal(len(t_obs))

# finite-difference estimate from the readings
dhdt = np.gradient(h_obs, t_obs)
cd_fd = (qin - A * dhdt) / (a * np.sqrt(2 * g * np.clip(h_obs, 1e-6, None)))


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 1))


ts = lambda t: torch.tensor(np.asarray(t, float)[:, None] / T_END, dtype=torch.float32)   # noqa: E731
To, Ho = ts(t_obs), torch.tensor(h_obs[:, None], dtype=torch.float32)
Tc = ts(np.linspace(0, T_END, 200)).requires_grad_(True)


def train(physics: bool, iters: int = 6000):
    net = mlp()
    raw_cd = torch.nn.Parameter(torch.tensor(0.0))                  # Cd = 0.3 + 0.6 sigmoid(raw), starts at 0.6
    params = list(net.parameters()) + ([raw_cd] if physics else [])
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.mean((net(To) - Ho) ** 2) / NOISE ** 2
        if physics:
            cd = 0.3 + 0.6 * torch.sigmoid(raw_cd)
            h = net(Tc)
            dh = torch.autograd.grad(h.sum(), Tc, create_graph=True)[0] / T_END
            res = A * dh - (qin - cd * a * torch.sqrt(2 * g * torch.clamp(h, min=1e-6)))
            loss = loss + torch.mean(res ** 2) / (0.01 * qin) ** 2
        loss.backward()
        opt.step()
    cd = float(0.3 + 0.6 * torch.sigmoid(raw_cd.detach())) if physics else None
    return net, cd


pinn, cd_pinn = train(True)
data_net, _ = train(False)
with torch.no_grad():
    h_pinn = pinn(ts(t_ref)).numpy().ravel()
    h_data = data_net(ts(t_ref)).numpy().ravel()
ext = t_ref > T_DATA
m = {"Cd_true": CD_TRUE, "Cd_pinn": cd_pinn, "Cd_pinn_error_pct": 100 * abs(cd_pinn - CD_TRUE) / CD_TRUE,
     "Cd_finite_differences_mean": float(cd_fd.mean()), "Cd_finite_differences_std": float(cd_fd.std()),
     "noise_m": NOISE, "readings": len(t_obs),
     "extrapolation_rmse_mm": {"pinn": float(1e3 * np.sqrt(np.mean((h_pinn[ext] - h_ref[ext]) ** 2))),
                               "data_only": float(1e3 * np.sqrt(np.mean((h_data[ext] - h_ref[ext]) ** 2)))},
     "steady_level_m": {"true": (qin / (CD_TRUE * a)) ** 2 / (2 * g),
                        "from_pinn_Cd": (qin / (cd_pinn * a)) ** 2 / (2 * g)}}
print(json.dumps(m, indent=1))
json.dump(m, open(os.path.join(OUT, "tank_metrics.json"), "w"), indent=1)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(t_ref, h_ref, "k", lw=2, label="FMU (OpenModelica), Cd = 0.62")
ax.plot(t_obs, h_obs, "o", color="C1", label="readings (5 mm noise)")
ax.plot(t_ref, h_pinn, "--", color="C0", lw=2, label=f"PINN, identified Cd = {cd_pinn:.3f}")
ax.plot(t_ref, h_data, ":", color="C3", lw=2, label="data-only network")
ax.axvspan(T_DATA, T_END, color="0.92", zorder=0)
ax.text(T_DATA + 15, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.9, "no readings", color="0.4")
ax.set_xlabel("time [s]")
ax.set_ylabel("level [m]")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
ax.set_title("Draining tank: discharge coefficient from 21 noisy readings")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "tank_inverse_pinn.png"), dpi=110)
print("wrote", os.path.join(ASSETS, "tank_inverse_pinn.png"))
