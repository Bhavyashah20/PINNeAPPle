"""3-D lattice Boltzmann wind tunnel: drag on a sphere against Stokes' law and the Schiller-Naumann correlation, in
32-bit and 16-bit storage of the distributions.

1. Stokes flow (Re ~ 0.06): a sphere of radius 5 cells in a box of 8 radii, the open sides fixed to the analytic
   Stokes far field (as in the usual LBM validation of this case). Drag against F = 6 pi rho nu R U.
2. Re = 20: sphere of diameter 8 cells in a 96 x 48 x 48 tunnel (inlet, outlet, open sides), drag coefficient against
   Schiller & Naumann (1933), with FP32 and with FP16 storage of the shifted distributions (half the memory).

The solver is pinneapple_simulation.numerical_solvers.lbm_tunnel (PyTorch, CPU or GPU). About 35 minutes on 4 CPU
cores. Writes docs/assets/lbm/sphere_tunnel.png and examples/lbm/_out/sphere_tunnel.json.
"""
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinneapple_simulation.numerical_solvers.lbm_tunnel import (
    WindTunnelLBM,
    drag_sphere_schiller_naumann,
    sphere,
    stokes_sphere_field,
)

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "lbm")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)
res = {}

# 1. Stokes drag
R, nu, u = 5.0, 1 / 6, 0.002
L = int(8 * R)
c = ((L - 1) / 2,) * 3
lbm = WindTunnelLBM((L, L, L), nu=nu, u_in=(u, 0, 0), faces={k: "open" for k in WindTunnelLBM.FACES})
lbm.add_solid(sphere(lbm.shape, c, R))
lbm.initialize(stokes_sphere_field(lbm.shape, c, R, (u, 0, 0)))
lbm.run(1200)
F = lbm.force_on_solid()[0]
res["stokes"] = {"radius_cells": R, "Re": 2 * R * u / nu, "F_lbm": float(F), "F_stokes": 6 * np.pi * nu * R * u,
                 "error": float(F / (6 * np.pi * nu * R * u) - 1)}
print("Stokes", res["stokes"], flush=True)

# 2. Re = 20, FP32 and FP16
D, U, Re = 8.0, 0.05, 20.0
nu = U * D / Re
shape = (96, 48, 48)
center = (24, 23.5, 23.5)
fields = {}
for storage in ("fp32", "fp16s"):
    t0 = time.time()
    lbm = WindTunnelLBM(shape, nu=nu, u_in=(U, 0, 0), storage=storage)
    lbm.add_solid(sphere(shape, center, D / 2))
    hist = []
    lbm.run(3600, every=200, callback=lambda m, h=hist: h.append((m.t, float(m.force_on_solid()[0]))))
    Fx = np.mean([f for t, f in hist[-4:]])
    cd = Fx / (0.5 * U ** 2 * np.pi * D ** 2 / 4)
    res[storage] = {"Cd": float(cd), "Cd_schiller_naumann": drag_sphere_schiller_naumann(Re),
                    "error": float(cd / drag_sphere_schiller_naumann(Re) - 1), "bytes_per_cell": lbm.bytes_per_cell(),
                    "history": hist, "wall_s": round(time.time() - t0)}
    fields[storage] = lbm.fields()
    print(storage, {k: v for k, v in res[storage].items() if k != "history"}, flush=True)
res["fp16_vs_fp32"] = {"Cd_rel_diff": res["fp16s"]["Cd"] / res["fp32"]["Cd"] - 1,
                       "u_max_abs_diff": float(np.abs(fields["fp16s"]["u"] - fields["fp32"]["u"]).max())}
json.dump(res, open(os.path.join(OUT, "sphere_tunnel.json"), "w"), indent=1)

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, width_ratios=(1.7, 1))
for k, storage in enumerate(("fp32", "fp16s")):
    ax = fig.add_subplot(gs[k, 0])
    uxz = fields[storage]["u"][0, :, 24, :].T
    im = ax.imshow(uxz / U, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=1.3)
    ax.set_title(f"u_x / U at mid-plane, Re = 20, {storage.upper()} storage ({res[storage]['bytes_per_cell']} B/cell "
                 f"for the distributions): Cd = {res[storage]['Cd']:.2f}", fontsize=9)
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("z [cells]")
    plt.colorbar(im, ax=ax, fraction=0.025)
ax = fig.add_subplot(gs[:, 1])
for storage, col in (("fp32", "C0"), ("fp16s", "C1")):
    h = np.array(res[storage]["history"])
    ax.plot(h[:, 0], h[:, 1] / (0.5 * U ** 2 * np.pi * D ** 2 / 4), color=col, label=storage.upper())
ax.axhline(drag_sphere_schiller_naumann(Re), color="k", ls="--", label="Schiller & Naumann (1933)")
ax.set_ylim(0, 6)
ax.set_xlabel("time step")
ax.set_ylabel("Cd")
ax.set_title(f"Sphere drag, Re = 20 (D = {D:g} cells); Stokes check: {100 * res['stokes']['error']:+.1f} %",
             fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sphere_tunnel.png"), dpi=110)
print(json.dumps({k: v for k, v in res.items() if k not in ("fp32", "fp16s")}, indent=1))
