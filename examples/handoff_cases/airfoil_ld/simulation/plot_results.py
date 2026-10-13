"""Figures and tables for results/: campaign overview, optimised shapes, the earlier optimiser picks in CFD.

    python plot_results.py RESULTS_DIR [--verification ../../../../apps/aero_optimizer/model/verification.json]

Reads RESULTS_DIR/campaign/campaign_runs.csv and RESULTS_DIR/optimisation/optimised.json; writes
campaign/campaign_overview.png, optimisation/optimised_shapes.png and
optimisation/aircraft_pareto_picks_cfd.csv (the 6 designs of the existing aircraft-level search, verified with
OpenFOAM polars: their Cl, Cd, Cm and L/D at 4 deg, read from the CFD polar).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from pinneapple_design.aero.geometry import PARAMS, properties, surfaces  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--verification", default=os.path.join(ROOT, "apps", "aero_optimizer", "model",
                                                           "verification.json"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(a.results, "campaign", "campaign_runs.csv"))))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=140)
    colors = {"converged": "#1f3b73", "steady-ish": "#2e8b57", "unsteady": "#c2491d"}
    for st, c in colors.items():
        r = [x for x in rows if x["status"] == st]
        al = [float(x["alpha_deg"]) for x in r]
        ax[0].scatter(al, [float(x["Cl"]) for x in r], s=7, color=c, label=f"{st} ({len(r)})")
        ax[1].scatter([float(x["Cd"]) for x in r], [float(x["Cl"]) for x in r], s=7, color=c)
    ld4 = [(float(x["L_over_D"]), float(x["thickness"])) for x in rows if 2 <= float(x["alpha_deg"]) <= 6]
    ax[2].scatter([t for _, t in ld4], [ld for ld, _ in ld4], s=8, color="#1f3b73")
    ax[0].set_xlabel("alpha (deg)")
    ax[0].set_ylabel("Cl")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Cd")
    ax[1].set_ylabel("Cl")
    ax[2].set_xlabel("thickness t/c")
    ax[2].set_ylabel("L/D (runs with 2 <= alpha <= 6 deg)")
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle("436 OpenFOAM runs, Re 4 million (100 Latin-hypercube airfoils x 4 angles + 4 NACA polars x 9)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "campaign", "campaign_overview.png"))
    plt.close(fig)

    opt = json.load(open(os.path.join(a.results, "optimisation", "optimised.json")))
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=140)
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
    for (name, d), c in zip(opt["designs"].items(), ["0.4", "#c2491d", "#2e8b57", "#7a4fa3"], strict=False):
        s = np.array([d[p] for p in PARAMS])
        yu, yl = surfaces(s, x)
        cfd = d.get("openfoam", {})
        ax.plot(np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]], color=c, lw=1.3,
                label=f"{name}: t/c {properties(s)['thickness']:.3f}, L/D {cfd.get('L_over_D', float('nan')):.1f} "
                      f"(CFD), Cm {cfd.get('Cm', float('nan')):.3f}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_xlabel("x / c")
    ax.set_title(f"Max L/D at alpha {opt['alpha_deg']:g} deg, Re 4 million: surrogate optimum checked in OpenFOAM",
                 fontsize=9)
    ax.set_ylim(-0.12, 0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "optimisation", "optimised_shapes.png"))
    plt.close(fig)

    ver = json.load(open(a.verification))
    with open(os.path.join(a.results, "optimisation", "aircraft_pareto_picks_cfd.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design_id", *PARAMS, "thickness", "wing_area_m2", "cfd_Cl_a4", "cfd_Cd_a4", "cfd_Cm_a4",
                    "cfd_L_over_D_a4", "cfd_max_L_over_D", "alpha_at_max_L_over_D", "surrogate_L_over_D_a4",
                    "aircraft_vmax_kt_cfd", "aircraft_vmax_kt_surrogate", "feasible_cfd"])
        for rd in ver["rounds"]:
            for d in rd["designs"]:
                P = d["openfoam"]["polar"]
                al, cl, cd, cm = (np.array(P[k]) for k in ("alpha", "cl", "cd", "cm"))
                i4 = int(np.argmin(abs(al - 4)))
                ld = cl / cd
                S = d["surrogate_polar"]
                j4 = S["alpha"].index(4)
                w.writerow([d["id"], *[f"{v:.6f}" for v in d["shape"]], f"{d['props']['thickness']:.4f}",
                            f"{d['wing_area']:.2f}", f"{cl[i4]:.5f}", f"{cd[i4]:.6f}", f"{cm[i4]:.5f}",
                            f"{ld[i4]:.2f}", f"{ld.max():.2f}", f"{al[int(np.argmax(ld))]:g}",
                            f"{S['cl'][j4] / S['cd'][j4]:.2f}", f"{d['openfoam']['vmax_kt']:.1f}",
                            f"{d['surrogate']['vmax_kt']:.1f}", d["openfoam"]["feasible"]])
    print("ok")


if __name__ == "__main__":
    main()
