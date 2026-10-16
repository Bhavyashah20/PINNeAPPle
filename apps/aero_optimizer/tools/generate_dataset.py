"""Run the OpenFOAM training set: LHS airfoil shapes x angles of attack, plus NACA reference polars.

    python apps/aero_optimizer/tools/generate_dataset.py OUT_DIR [--shapes 100] [--alphas 4] [--workers 4]

Each run leaves OUT_DIR/runs/<id>.json (inputs, coefficients, convergence, wall Cp/Cf) and <id>.npz (cell fields
p, Ux, Uy, nut on the shared O-grid). Re-running skips finished runs. The plan and the run itself are
``pinneapple_design.aero.dataset`` (``plan``, ``run_job``).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.dataset import ALPHA_BINS, RE, plan, run_job  # noqa: E402,F401

one = run_job                                                         # the former name of run_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shapes", type=int, default=100)
    ap.add_argument("--alphas", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["lhs", "reference"], default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    jobs = [j for j in plan(a.shapes, a.alphas) if (a.only is None or j["set"] == a.only)
            and not os.path.exists(os.path.join(a.out, "runs", j["id"] + ".json"))]
    jobs.sort(key=lambda j: j["set"] != "reference")                 # references first: validation early
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, j, a.out) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                print(i + 1, *f.result(), flush=True)
            except Exception as e:                                    # noqa: BLE001
                print(i + 1, "error", e, flush=True)


if __name__ == "__main__":
    main()
