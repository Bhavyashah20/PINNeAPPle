"""Write and run one 2-D airfoil case (OpenFOAM simpleFoam, k-omega SST, y+ < 1) from 6 CST weights.

This is a thin layer over the PINNeAPPle library (``pinneapple_design.aero``), the same code that produced the
436-run campaign. Cl / Cd / Cm come from the library's integration of the written wall fields (pressure and
viscous shear of the wall-adjacent cells, moment about the quarter chord), at every write, so the iteration
history is also a convergence check. No OpenFOAM function objects are used: in the OpenFOAM v1912 build used here
(Ubuntu 24.04 package) any function object in controlDict aborts with "error in IOstream sha1". With a newer
OpenFOAM you may add a ``forceCoeffs`` function object as an independent check (see README.md).

    python airfoil_case.py CASE_DIR --shape naca2412 --alpha 4 --re 4e6 [--grid medium] [--run]
    python airfoil_case.py CASE_DIR --shape 0.17,0.15,0.14,0.02,0.05,0.04 --alpha 4 --run

Non-dimensional set-up: chord c = 1, |U_inf| = 1, nu = 1/Re, rho = 1 (kinematic pressure), span 0.05 (one cell).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if os.path.isdir(os.path.join(ROOT, "pinneapple_design")):
    sys.path.insert(0, ROOT)
from pinneapple_design.aero.case import (  # noqa: E402
    FOAM_BASHRC,
    read_coefficients,
    wall_geometry,
    write_case,
)
from pinneapple_design.aero.geometry import naca4, properties  # noqa: E402
from pinneapple_design.aero.mesh import GridSpec, grid_points  # noqa: E402

# Three nested grids of the same O-grid topology. "medium" is the grid of the 436-run campaign.
GRIDS = {
    "coarse": GridSpec(n_surf=56, n_te=2, nj=60, first=4e-6, radius=200.0),
    "medium": GridSpec(),                                                     # 228 x 88 = 20 064 cells
    "fine": GridSpec(n_surf=168, n_te=6, nj=132, first=4e-6, radius=200.0),
}
ITERATIONS = {"coarse": 1500, "medium": 2000, "fine": 4000}


def parse_shape(s: str) -> np.ndarray:
    if s.lower().startswith("naca"):
        return naca4(s[4:])
    v = np.array([float(x) for x in s.split(",")])
    if v.shape != (6,):
        raise SystemExit("--shape needs 6 comma-separated CST weights T0,T1,T2,M0,M1,M2 or nacaXXXX")
    return v


def write(case: str, shape: np.ndarray, alpha: float, reynolds: float, grid: str, iterations: int | None = None):
    g = GRIDS[grid]
    it = iterations or ITERATIONS[grid]
    write_case(case, shape, alpha, reynolds, it, grid=g, write_every=100)
    meta = {"shape_T0_T1_T2_M0_M1_M2": [float(x) for x in shape], "alpha_deg": alpha, "reynolds": reynolds,
            "grid": grid, "ni": g.ni, "nj": g.nj, "cells": g.ni * g.nj, "first_cell": g.first,
            "farfield_radius_chords": g.radius, "iterations_max": it, "properties": properties(shape)}
    with open(os.path.join(case, "case_info.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def y_plus(P2: np.ndarray, cf: np.ndarray, reynolds: float) -> np.ndarray:
    """y+ of the wall-adjacent cell centres from the wall shear (u_tau = sqrt(|Cf| / 2) with |U_inf| = 1)."""
    d = wall_geometry(P2)["d"]
    return np.sqrt(np.abs(cf) / 2.0) * d * reynolds


def residuals(case: str) -> dict:
    """Initial residuals per iteration from log.simpleFoam."""
    out: dict = {}
    txt = open(os.path.join(case, "log.simpleFoam")).read()
    for f, v in re.findall(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)", txt):
        out.setdefault(f, []).append(float(v))
    m = re.search(r"SIMPLE solution converged in (\d+) iterations", txt)
    out["converged_in"] = int(m.group(1)) if m else None
    return out


def run(case: str, shape: np.ndarray, alpha: float, reynolds: float, grid: str, procs: int = 1) -> dict:
    t0 = time.time()
    cmd = "simpleFoam"
    if procs > 1:
        with open(os.path.join(case, "system", "decomposeParDict"), "w") as f:
            f.write("FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
                    f"numberOfSubdomains {procs};\nmethod scotch;\n")
        cmd = f"decomposePar -force > log.decomposePar 2>&1 && mpirun --allow-run-as-root -np {procs} simpleFoam -parallel"
    subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && checkMesh > log.checkMesh '
                                  f'2>&1; {cmd} > log.simpleFoam 2>&1'], check=False)
    if procs > 1:
        subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && '
                                      f'reconstructPar > log.reconstructPar 2>&1'], check=False)
    g = GRIDS[grid]
    P2 = grid_points(shape, g)
    lib = read_coefficients(case, P2, 1.0 / reynolds, alpha, g.ni)
    res = {"seconds": round(time.time() - t0, 1), "residuals": residuals(case), "history": lib.get("history"),
           **{k: lib.get(k) for k in ("iterations", "Cl", "Cd", "Cm", "Cd_pressure", "Cl_osc", "Cd_osc", "Cm_osc")}}
    if lib.get("wall"):
        res["wall"] = lib["wall"]
        yp = y_plus(P2, np.array(lib["wall"]["Cf"]), reynolds)
        res["y_plus_max"], res["y_plus_mean"] = float(yp.max()), float(yp.mean())
    res["L_over_D"] = res["Cl"] / res["Cd"] if res.get("Cd") else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--shape", default="naca2412")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=4e6)
    ap.add_argument("--grid", choices=list(GRIDS), default="medium")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    shape = parse_shape(a.shape)
    meta = write(a.case, shape, a.alpha, a.re, a.grid, a.iterations)
    print(json.dumps({k: meta[k] for k in ("grid", "cells", "alpha_deg", "reynolds")}))
    if a.run:
        r = run(a.case, shape, a.alpha, a.re, a.grid, a.procs)
        with open(os.path.join(a.case, "result.json"), "w") as f:
            json.dump(r, f)
        print(json.dumps({k: r.get(k) for k in ("iterations", "Cl", "Cd", "Cm", "L_over_D", "Cl_osc", "Cd_osc",
                                                "y_plus_max", "seconds")} | {"converged_in": r["residuals"]["converged_in"]}))


if __name__ == "__main__":
    main()
