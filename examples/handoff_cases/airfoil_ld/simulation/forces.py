"""Cl, Cd, Cm of a finished airfoil case (written by airfoil_case.py), at every kept write.

    python forces.py CASE_DIR

Reads CASE_DIR/case_info.json (shape, alpha, Re, grid) and the written fields; prints the coefficient history and
writes CASE_DIR/forces.json. Pressure and viscous shear of the wall-adjacent cells, integrated over the airfoil,
moment about the quarter chord (nose-up positive), reference length 1, dynamic pressure 1/2.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from airfoil_case import GRIDS, y_plus  # noqa: E402

from pinneapple_design.aero.case import read_coefficients  # noqa: E402
from pinneapple_design.aero.mesh import grid_points  # noqa: E402


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "."
    info = json.load(open(os.path.join(case, "case_info.json")))
    shape = np.array(info["shape_T0_T1_T2_M0_M1_M2"])
    g = GRIDS[info["grid"]]
    P2 = grid_points(shape, g)
    r = read_coefficients(case, P2, 1.0 / info["reynolds"], info["alpha_deg"], g.ni)
    for it, cl, cd, cm in r["history"]:
        print(f"iteration {int(it):6d}  Cl {cl:.5f}  Cd {cd:.6f}  Cm {cm:.5f}  L/D {cl / cd:.2f}")
    yp = y_plus(P2, np.array(r["wall"]["Cf"]), info["reynolds"])
    out = {k: r[k] for k in ("iterations", "Cl", "Cd", "Cm", "Cd_pressure", "Cl_osc", "Cd_osc", "Cm_osc", "history")}
    out.update({"L_over_D": r["Cl"] / r["Cd"], "y_plus_max": float(yp.max()), "wall": r["wall"]})
    with open(os.path.join(case, "forces.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
