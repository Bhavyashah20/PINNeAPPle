"""Resolution study of the black-hole weather reference simulation (issue #399): is the training data converged?

Run ``simulate.py`` on three or more grids in a constant refinement ratio with the same physics and seeded
perturbation (``--perturbation-kind modes``), then:

    python examples/black_hole_weather/convergence.py --runs data/bh_conv --levels 128 181 256 362 --seeds 0 1 2 \\
        --t0 2000 --t1 10000 --out examples/black_hole_weather/results/convergence.json

Run folders are ``conv_n{nr}_s{seed}``. The analysis is ``pinneapple_physics.blackhole.convergence.resolution_study``
(quantities, conservative remap, observed order and GCI, statistical convergence).
"""
from __future__ import annotations

import argparse
import json
import os

from pinneapple_physics.blackhole.convergence import resolution_study


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--levels", type=int, nargs="+", required=True, help="nr of each grid (ntheta = nr / 2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--t0", type=float, default=2000.0)
    ap.add_argument("--t1", type=float, default=10000.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = {n: [os.path.join(a.runs, f"conv_n{n}_s{s}") for s in a.seeds
                if os.path.exists(os.path.join(a.runs, f"conv_n{n}_s{s}", "times.npy"))] for n in a.levels}
    report = resolution_study(runs, t0=a.t0, t1=a.t1, log=lambda m: print(m, flush=True))
    report["seeds"] = a.seeds
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for name, v in report["scalars"].items():
        row = "  ".join(f"{n}: {d['mean']:.4g} +- {d['ci95']:.2g}" for n, d in v["by_level"].items())
        r = v["richardson"]
        if r.get("gci_fine"):
            extra = f"p {r['observed_order']:.2f} GCI {r['gci_fine']:.2%}"
        elif r.get("monotone"):
            extra = f"diverging (p {r['observed_order']:.2f})"
        else:
            extra = "not monotone"
        print(f"{name:26s} {row}  | {extra} | stat. converged {v['statistically_converged']}")
    for kind in ("profiles", "fields"):
        for name, v in report[kind].items():
            print(f"{name:12s} L1 change between grids {', '.join(f'{d:.3g}' for d in v['abs_L1_diff_consecutive'])}"
                  f"  seed spread {v['seed_spread_abs']}")


if __name__ == "__main__":
    main()
