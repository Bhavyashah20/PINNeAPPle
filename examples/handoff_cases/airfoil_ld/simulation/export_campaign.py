"""Export the 436-run OpenFOAM campaign of apps/aero_optimizer as one CSV (shape parameters, alpha, Cl, Cd, status).

    python export_campaign.py OUT.csv [--dataset ../../../../apps/aero_optimizer/model/dataset.json]

The stored campaign summary (model/dataset.json) keeps, per run, its id, alpha, Cl, Cd, convergence status and
thickness. The 6 CST weights of each run are regenerated from the campaign plan (pinneapple_design.aero.dataset.plan, seeded
Latin hypercube, seed 7); the script checks that the regenerated angle and thickness match the stored ones.
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
from pinneapple_design.aero.dataset import RE, plan  # noqa: E402
from pinneapple_design.aero.geometry import PARAMS, properties  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "apps", "aero_optimizer", "model", "dataset.json"))
    a = ap.parse_args()
    jobs = {j["id"]: j for j in plan(100, 4)}
    pts = json.load(open(a.dataset))["points"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "set", "shape_id", *PARAMS, "thickness", "alpha_deg", "reynolds", "Cl", "Cd",
                    "L_over_D", "status"])
        for p in pts:
            j = jobs[p["id"]]
            s = np.array(j["shape"])
            t = properties(s)["thickness"]
            assert abs(j["alpha"] - p["alpha"]) < 1e-9 and abs(t - p["t"]) < 1e-9, p["id"]
            w.writerow([p["id"], j["set"], j["shape_id"], *[f"{v:.6f}" for v in s], f"{t:.5f}", f"{p['alpha']:.4f}",
                        f"{RE:.0f}", f"{p['cl']:.6f}", f"{p['cd']:.7f}", f"{p['cl'] / p['cd']:.3f}", p["status"]])
    print(len(pts), "runs ->", a.out)


if __name__ == "__main__":
    main()
