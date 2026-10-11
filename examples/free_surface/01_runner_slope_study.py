"""Melt runner: how the floor slope changes the jet's attack on the floor (VOF, k-omega SST, interFoam).

A tapping jet leaves a 8 cm slot at 1 m/s forward and 2.5 m/s down and reaches the bath at about 5 m/s; it falls into
a runner whose bath is held by a weir (skimmer) 0.5 m high at
the downstream end. The floor pivots about the weir end: +2° (floor descending toward the weir) leaves 0.26 m of
metal under the jet, 0° 0.5 m, -2° 0.74 m. Same weir, same bath level, same jet; only the depth at impact changes.
The idea comes from a design discussion of runner refractory wear: a deep pool at impact should damp the jet
before it reaches the floor.

Each case is a 2-D longitudinal section (pinneapple_design.geometry.trough.TroughSection), run with
pp.cfd.FreeSurfaceFlow for 12 s. k is sampled 5 cm above the floor every 0.1 s and averaged over 4-12 s. Reported:
* time-mean k and the peak velocity 5 cm above the floor in the impact zone (0.5-3 m), relative to the 0° case;
* the liquid volume over time (a plateau means inflow and outflow balance: the flow has settled).

A 2-D section with RANS is a screening tool: it ranks the variants, it does not give the 3-D wear rate.
Needs OpenFOAM (v1912+). About 10 minutes on 1 core. Writes docs/assets/free_surface/runner_slope.png and
examples/free_surface/_out/slope_metrics.json.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinneapple_design.geometry.trough import TroughSection
from pinneapple_simulation.numerical_solvers.free_surface import (
    FreeSurfaceFlow,
    Phase,
    SampleLine,
    _scalar,
    _times,
    cell_centres,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
ASSETS = os.path.join(HERE, "..", "..", "docs", "assets", "free_surface")
os.makedirs(ASSETS, exist_ok=True)
JET = (1.0, 0.0, -2.5)                       # world frame: x downstream, z up
T_END, T_AVG = 12.0, 4.0
ZONE = (0.5, 3.0)
SLOPES = (2.0, 0.0, -2.0)


def run(slope):
    sec = TroughSection(slope_deg=slope, inlet_width=0.08)
    case = os.path.join(OUT, f"slope_{slope:+.0f}")
    flow = FreeSurfaceFlow(
        mesh=sec.write_mesh(os.path.join(case, "mesh")), two_d=True,
        roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
        inlet_velocity={"inlet": sec.to_mesh(JET)}, level=sec.level(), gravity=sec.gravity(9.8),
        liquid=Phase("metal", 7200.0, 8.472e-6), gas=Phase("air", 1.2, 1.5e-5), sigma=1.8,
        end_time=T_END, sample_every=0.1, keep_every=T_AVG,
        lines=[SampleLine("floor", (0.0, 0.025, 0.05), (sec.length, 0.025, 0.05), 161)])
    run_dir = os.path.join(case, "run")
    if not os.path.exists(os.path.join(run_dir, f"{T_END:g}")):
        flow.write(run_dir, procs=1)
        flow.run(run_dir, procs=1)
    return sec, flow, run_dir


res = {}
for s in SLOPES:
    sec, flow, case = run(s)
    prof = flow.profiles(case, t_start=T_AVG)["floor"]
    x = prof["s"]
    zone = (x >= ZONE[0]) & (x <= ZONE[1])
    log = open(os.path.join(case, "log.interFoam")).read()
    steps = [(float(b.split("\n", 1)[0]), re.findall(r"Phase-1 volume fraction = (\S+)", b))
             for b in log.split("\nTime = ")[1:]]
    t = np.array([a for a, f in steps if f])
    frac = np.array([float(f[-1]) for _, f in steps if f])
    cc = cell_centres(os.path.join(case, "constant", "polyMesh"))
    tl = _times(case)[-1][1]
    res[s] = {"sec": sec, "x": x, "k": prof["mean_k"], "k_zone": float(prof["mean_k"][zone].mean()),
              "k_peak": float(prof["mean_k"][zone].max()), "t": t, "frac": frac, "cc": cc,
              "alpha": _scalar(os.path.join(case, tl, "alpha.metal"), len(cc)),
              "depth_at_jet": sec.depth_at(sec.inlet_x + 0.5 * sec.inlet_width)}
k0 = res[0.0]["k_zone"]
metrics = {f"{s:+.0f} deg": {"depth_at_jet_m": round(r["depth_at_jet"], 3), "k_zone_mean": r["k_zone"],
                             "k_zone_rel_to_0deg": r["k_zone"] / k0, "k_peak": r["k_peak"],
                             "liquid_fraction_last_4s_change_pct":
                                 float(100 * np.ptp(r["frac"][r["t"] > T_END - 4]) / r["frac"][-1])}
           for s, r in res.items()}
os.makedirs(OUT, exist_ok=True)
json.dump(metrics, open(os.path.join(OUT, "slope_metrics.json"), "w"), indent=1)
print(json.dumps(metrics, indent=1))

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(4, 2, width_ratios=(1.35, 1), hspace=0.55)
for i, s in enumerate(SLOPES):
    r = res[s]
    ax = fig.add_subplot(gs[i, 0])
    sec = r["sec"]
    th = sec.theta
    X = r["cc"][:, 0] * np.cos(th) + r["cc"][:, 2] * np.sin(th)        # mesh -> world (pivot at the weir end)
    Z = -(r["cc"][:, 0] - sec.length) * np.sin(th) + r["cc"][:, 2] * np.cos(th)
    ax.tricontourf(X, Z, r["alpha"], levels=[0, 0.5, 1], colors=["#dfe7ef", "#e0662b"])
    ax.plot([0, sec.length], [sec.floor(0), 0], color="k", lw=1.5)
    ax.set_xlim(0, sec.length)
    ax.set_ylim(min(sec.floor(0), 0) - 0.1, 1.7)
    ax.set_aspect("equal")
    ax.set_title(f"{s:+.0f}°: {r['depth_at_jet']:.2f} m of metal under the jet, t = {T_END:g} s", fontsize=10)
    ax.tick_params(labelsize=8)
ax = fig.add_subplot(gs[:3, 1])
for s, c in zip(SLOPES, ("C3", "C0", "C2"), strict=True):
    r = res[s]
    ax.plot(r["x"], r["k"], color=c, lw=2, label=f"{s:+.0f}° (zone mean ×{r['k_zone'] / k0:.2f})")
ax.axvspan(*ZONE, color="0.93", zorder=0)
ax.set_xlabel("distance from the upstream wall [m]")
ax.set_ylabel(f"k 5 cm above the floor, mean {T_AVG:g}-{T_END:g} s [m²/s²]")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[3, 1])
for s, c in zip(SLOPES, ("C3", "C0", "C2"), strict=True):
    ax2.plot(res[s]["t"], res[s]["frac"] / res[s]["frac"][0], color=c, lw=1.2)
ax2.set_xlabel("time [s]")
ax2.set_ylabel("liquid volume / initial", fontsize=8)
ax2.grid(alpha=0.3)
fig.savefig(os.path.join(ASSETS, "runner_slope.png"), dpi=110, bbox_inches="tight")
print("wrote", os.path.join(ASSETS, "runner_slope.png"))
