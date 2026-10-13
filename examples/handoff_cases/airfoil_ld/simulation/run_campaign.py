"""Run a campaign of airfoil variants in OpenFOAM (in parallel) and collect Cl, Cd, Cm, L/D in one CSV.

    # the variants of geometry/cst_parameters.json at 3 angles, campaign grid, 4 runs at a time
    python run_campaign.py OUT_DIR --variants ../geometry/cst_parameters.json --alphas 3 4 5 --workers 4

    # n new Latin-hypercube variants inside the CST bounds (valid shapes only), at alpha 4
    python run_campaign.py OUT_DIR --lhs 40 --seed 11 --alphas 4

    # the original 436-run training campaign (same plan, seed 7), as in apps/aero_optimizer
    python run_campaign.py OUT_DIR --original

Each run is a case directory OUT_DIR/cases/<id> (fields kept, use --clean to delete them) and a line of
OUT_DIR/campaign.csv. Finished runs (result.json present) are skipped on a re-run. Convergence status as in the
original campaign: "converged" (residual criteria met, or Cl / Cd steady over the last 5 writes), "steady-ish",
"unsteady" (oscillating, usually separated flow near stall), "failed".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from airfoil_case import run, write  # noqa: E402

from pinneapple_design.aero.geometry import (  # noqa: E402
    BOUNDS,
    PARAMS,
    latin_hypercube,
    properties,
    valid,
)


def status(r: dict) -> str:
    if r.get("Cl") is None:
        return "failed"
    if r["residuals"].get("converged_in") or (r["Cl_osc"] < 2e-3 and r["Cd_osc"] < 5e-5):
        return "converged"
    if r["Cl_osc"] < 0.02 and r["Cd_osc"] < 1e-3:
        return "steady-ish"
    return "unsteady"


def one(job: dict, out: str, grid: str, reynolds: float, clean: bool) -> dict:
    case = os.path.join(out, "cases", job["id"])
    done = os.path.join(case, "result.json")
    if os.path.exists(done):
        r = json.load(open(done))
    else:
        s = np.array(job["shape"])
        write(case, s, job["alpha"], reynolds, grid)
        r = run(case, s, job["alpha"], reynolds, grid)
        r.pop("wall", None)
        with open(done, "w") as f:
            json.dump(r, f)
    rec = {"run_id": job["id"], "variant": job["variant"], **dict(zip(PARAMS, job["shape"], strict=False)),
           "thickness": properties(np.array(job["shape"]))["thickness"], "alpha_deg": job["alpha"],
           "reynolds": reynolds, "grid": grid, **{k: r.get(k) for k in ("Cl", "Cd", "Cm", "L_over_D", "Cl_osc",
                                                                      "Cd_osc", "iterations", "y_plus_max")},
           "status": status(r)}
    if clean:
        for d in os.listdir(case):
            if d.replace(".", "").isdigit() and d != "0":
                shutil.rmtree(os.path.join(case, d))
    return rec


def jobs_from(a) -> list[dict]:
    jobs = []
    if a.original:
        from pinneapple_design.aero.dataset import plan
        return [{"id": j["id"], "variant": j["shape_id"], "shape": j["shape"], "alpha": j["alpha"]} for j in plan(100, 4)]
    shapes = {}
    if a.variants:
        d = json.load(open(a.variants))
        for name, v in d["shapes"].items():
            shapes[name] = [v[p] for p in PARAMS]
    if a.lhs:
        rng = np.random.default_rng(a.seed)
        k = 0
        while k < a.lhs:
            for s in latin_hypercube(a.lhs, BOUNDS, rng):
                if valid(s)[0] and k < a.lhs:
                    shapes[f"lhs{k:03d}"] = s.tolist()
                    k += 1
    for name, s in shapes.items():
        for al in a.alphas:
            jobs.append({"id": f"{name}_a{al:+05.1f}".replace(".", "p"), "variant": name, "shape": list(map(float, s)),
                         "alpha": float(al)})
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--variants", help="JSON with a 'shapes' dict of CST weights (geometry/cst_parameters.json)")
    ap.add_argument("--lhs", type=int, default=0, help="number of Latin-hypercube variants")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--original", action="store_true", help="the 436-run training campaign")
    ap.add_argument("--alphas", type=float, nargs="+", default=[4.0])
    ap.add_argument("--re", type=float, default=4e6)
    ap.add_argument("--grid", default="medium", choices=["coarse", "medium", "fine"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", action="store_true", help="delete the field directories after each run")
    a = ap.parse_args()
    jobs = jobs_from(a)
    if not jobs:
        raise SystemExit("nothing to run: give --variants, --lhs or --original")
    os.makedirs(os.path.join(a.out, "cases"), exist_ok=True)
    print(f"{len(jobs)} runs", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, j, a.out, a.grid, a.re, a.clean): j["id"] for j in jobs}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                rows.append(r)
                print(i + 1, r["run_id"], r["status"], r["Cl"], r["Cd"], flush=True)
            except Exception as e:                                         # noqa: BLE001
                print(i + 1, futs[f], "error", e, flush=True)
    rows.sort(key=lambda r: r["run_id"])
    with open(os.path.join(a.out, "campaign.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
