"""Check the optimizer's picks with OpenFOAM: full polars for designs along the Pareto front, compared with the surrogate.

    python apps/aero_optimizer/tools/verify.py MODEL.pt OUT_DIR [--designs 6] [--workers 4] [--round 1]

OUT_DIR/runs/ gets the CFD runs in the training-set format (set "verify"), so they can be added to the next training
(active learning: --extra OUT_DIR). OUT_DIR/verification_round<N>.json holds, per design, the surrogate prediction and
the OpenFOAM result for top speed, CO2 per 100 km, stall speed and the polar. The logic is
``pinneapple_design.aero.verify``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.aircraft import Aircraft, Requirements  # noqa: E402
from pinneapple_design.aero.dataset import run_job  # noqa: E402
from pinneapple_design.aero.optimize import Engine  # noqa: E402
from pinneapple_design.aero.verify import (  # noqa: E402
    CFD_ALPHAS,
    compare_design,
    pick,
    verification_jobs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--designs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    eng = Engine.load(a.model)
    ac, req = Aircraft(), Requirements()
    res = eng.search(ac, req, population=96, generations=60, seed=a.seed)
    D = res["designs"]
    chosen = pick(res["pareto"], a.designs)
    print(f"{res['evaluations']} designs evaluated, {len(res['pareto'])} on the front, verifying {len(chosen)}", flush=True)
    jobs, meta = verification_jobs(D, chosen, a.round)
    jobs = [j for j in jobs if not os.path.exists(os.path.join(a.out, "runs", j["id"] + ".json"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_job, jobs, [a.out] * len(jobs)):
            print(*r, flush=True)
    out = []
    for did, d in meta:
        runs = []
        for al in CFD_ALPHAS:
            with open(os.path.join(a.out, "runs", f"{did}_a{al:+03d}.json")) as f:
                runs.append(json.load(f))
        out.append(compare_design(eng, did, d, [r for r in runs if "Cl" in r], ac, req))
        s, c = out[-1]["surrogate"], out[-1]["openfoam"]
        print(f"{did}: vmax {s['vmax_kt']:.1f} vs CFD {c['vmax_kt']:.1f} kt | CO2 {s['co2_100km']:.2f} vs {c['co2_100km']:.2f} | "
              f"stall {s['v_stall_kt']:.1f} vs {c['v_stall_kt']:.1f} kt | feasible {s['feasible']} vs {c['feasible']}", flush=True)
    with open(os.path.join(a.out, f"verification_round{a.round}.json"), "w") as f:
        json.dump({"round": a.round, "aircraft": dataclasses.asdict(ac), "requirements": req.as_dict(), "designs": out,
                   "evaluations": res["evaluations"], "pareto_size": len(res["pareto"])}, f, indent=1)


if __name__ == "__main__":
    main()
