"""Maximise L/D = Cl/Cd at one angle of attack with the trained surrogate, then check the optimum in OpenFOAM.

    python optimise_ld.py OUT_DIR [--alpha 4] [--model ../../../../apps/aero_optimizer/model/surrogate.pt]
                                  [--cfd] [--grid medium]

Search: scipy differential evolution over the 6 CST weights inside the campaign bounds (the surrogate is not
trusted outside them). Objective: Cl/Cd of the MLP-ensemble mean at (alpha, Re 4e6). A design is feasible when
  thickness >= t_min, |Cm c/4| <= cm_max, the shape is valid (no crossing / wavy surfaces), and the
  ensemble is confident (std of Cl < 0.01 and std of Cd < 2 % of Cd).
Constraint violations are added as large penalties. Three constraint sets are searched (the brief's set and
two tighter ones), each design is then run in OpenFOAM (--cfd) on the campaign grid, and OUT_DIR/optimised.json
holds surrogate vs CFD.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if os.path.isdir(os.path.join(ROOT, "pinneapple_design")):
    sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pinneapple_design.aero.geometry import BOUNDS, PARAMS, naca4, properties, valid  # noqa: E402
from pinneapple_design.aero.optimize import Engine  # noqa: E402

CASES = {"opt_A_brief": {"t_min": 0.12, "cm_max": 0.10},
         "opt_B_low_moment": {"t_min": 0.12, "cm_max": 0.06},
         "opt_C_thick": {"t_min": 0.15, "cm_max": 0.10}}


def evaluate(engine: Engine, X: np.ndarray, alpha: float) -> dict:
    r = engine.mlp_coefficients(np.atleast_2d(X), np.array([alpha]))
    return {k: v[:, 0] for k, v in r.items()}


def search(engine: Engine, alpha: float, t_min: float, cm_max: float, seed: int = 1) -> np.ndarray:
    def f(x):
        if not valid(x)[0]:
            return 1e3
        p = properties(x)
        r = evaluate(engine, x, alpha)
        ld = float(r["cl"][0] / r["cd"][0])
        pen = 0.0
        pen += 2e3 * max(0.0, t_min - p["thickness"])
        pen += 2e3 * max(0.0, abs(float(r["cm"][0])) - cm_max)
        pen += 1e3 * max(0.0, float(r["cl_std"][0]) - 0.01)
        pen += 1e3 * max(0.0, float(r["cd_std"][0] / r["cd"][0]) - 0.02)
        return -ld + pen
    res = differential_evolution(f, bounds=[tuple(b) for b in BOUNDS], seed=seed, popsize=30, maxiter=300,
                                 tol=1e-8, polish=False, init="latinhypercube")
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--model", default=os.path.join(ROOT, "apps", "aero_optimizer", "model", "surrogate.pt"))
    ap.add_argument("--cfd", action="store_true")
    ap.add_argument("--grid", default="medium")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    engine = Engine.load(a.model)
    designs = {"baseline_naca2412": {"constraints": None, "shape": naca4("2412")}}
    for name, c in CASES.items():
        designs[name] = {"constraints": c, "shape": search(engine, a.alpha, **c)}
    out = {"alpha_deg": a.alpha, "reynolds": 4e6, "bounds": {p: list(map(float, b)) for p, b in zip(PARAMS, BOUNDS, strict=False)},
           "designs": {}}
    for name, d in designs.items():
        s = d["shape"]
        r = evaluate(engine, s, a.alpha)
        rec = {"constraints": d["constraints"], **{p: float(v) for p, v in zip(PARAMS, s, strict=False)},
               "properties": properties(s),
               "surrogate": {"Cl": float(r["cl"][0]), "Cd": float(r["cd"][0]), "Cm": float(r["cm"][0]),
                             "L_over_D": float(r["cl"][0] / r["cd"][0]), "Cl_std": float(r["cl_std"][0]),
                             "Cd_std": float(r["cd_std"][0])}}
        if a.cfd:
            from airfoil_case import run, write
            case = os.path.join(a.out, "cases", name)
            write(case, s, a.alpha, 4e6, a.grid)
            c = run(case, s, a.alpha, 4e6, a.grid)
            rec["openfoam"] = {k: c.get(k) for k in ("Cl", "Cd", "Cm", "L_over_D", "Cl_osc", "Cd_osc", "iterations",
                                                     "y_plus_max", "seconds")}
            rec["openfoam"]["grid"] = a.grid
            rec["openfoam"]["converged"] = bool(c.get("Cl_osc", 1) < 2e-3 and c.get("Cd_osc", 1) < 5e-5)
            rec["openfoam"]["L_over_D_error_surrogate_pct"] = 100 * (rec["surrogate"]["L_over_D"] / c["L_over_D"] - 1)
        out["designs"][name] = rec
        print(name, json.dumps({"t": round(rec["properties"]["thickness"], 4), "sur": rec["surrogate"],
                                "cfd": rec.get("openfoam")}))
    with open(os.path.join(a.out, "optimised.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
