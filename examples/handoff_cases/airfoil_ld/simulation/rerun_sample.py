"""Re-run campaign runs by id, export their full fields to VTK, and compare with the stored coefficients.

    python rerun_sample.py OUT_DIR s012_a1 s047_a2 NACA4412_a+08 [--csv ../results/campaign/campaign_runs.csv]

The original campaign kept only the coefficients and wall data, not the OpenFOAM fields. These runs repeat
selected campaign runs exactly (same shape, alpha, grid, solver settings, 1200 iterations) and keep the fields:
OUT_DIR/<id>/VTK (foamToVTK, last iteration) and OUT_DIR/rerun_check.json (stored vs re-run Cl, Cd).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from airfoil_case import FOAM_BASHRC, run, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "campaign", "campaign_runs.csv"))
    a = ap.parse_args()
    rows = {r["run_id"]: r for r in csv.DictReader(open(a.csv))}
    out = {}
    for rid in a.ids:
        r = rows[rid]
        shape = np.array([float(r[k]) for k in ("T0", "T1", "T2", "M0", "M1", "M2")])
        alpha = float(r["alpha_deg"])
        case = os.path.join(a.out, rid)
        write(case, shape, alpha, float(r["reynolds"]), "medium", iterations=1200)
        res = run(case, shape, alpha, float(r["reynolds"]), "medium")
        subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && '
                                      f'foamToVTK -latestTime -ascii -fields "(p U k omega nut)" -noPointValues > log.foamToVTK 2>&1'], check=False)
        out[rid] = {"alpha_deg": alpha, "stored": {"Cl": float(r["Cl"]), "Cd": float(r["Cd"]), "status": r["status"]},
                    "rerun": {"Cl": res["Cl"], "Cd": res["Cd"], "Cm": res["Cm"], "iterations": res["iterations"]},
                    "diff_Cl": res["Cl"] - float(r["Cl"]), "diff_Cd_pct": 100 * (res["Cd"] / float(r["Cd"]) - 1)}
        print(rid, json.dumps(out[rid]))
    with open(os.path.join(a.out, "rerun_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
