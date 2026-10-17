"""Skill of the black-hole weather forecaster against persistence, climatology and DMD, with confidence intervals.

    python examples/black_hole_weather/skill.py --dataset blackhole_riaf_density_256 --models runs/bh256_res runs/bh256_roll \\
        --split test --out runs/bh256_skill

The evaluation is ``pinneapple_physics.blackhole.skill.evaluate_skill`` (rollouts from many starts, baselines,
moving-block bootstrap intervals, skill horizon, physics checks); this script loads the data and models and writes
``<out>/skill_<split>.json``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from pinneapple_physics.blackhole.dataset import load_trajectories, window_volumes
from pinneapple_physics.blackhole.forecast import load_forecaster
from pinneapple_physics.blackhole.skill import evaluate_skill


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="blackhole_riaf_density_256", help="zoo name, a data.npz or a folder of <split>.npz files")
    ap.add_argument("--models", nargs="+", required=True, help="training output folders (forecaster.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--start-every", type=int, default=5)
    ap.add_argument("--dmd-ranks", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    path = a.dataset if os.path.exists(a.dataset) else None   # a data.npz, a folder of <split>.npz, or the zoo name
    models = {}
    for m in a.models:
        net, codec, _ = load_forecaster(os.path.join(m, "forecaster.pt"))
        models[os.path.basename(os.path.normpath(m))] = (net, codec)
    test = load_trajectories(a.split, path)
    report = evaluate_skill(models, [v["log10_rho"] for v in load_trajectories("train", path).values()],
                            [v["log10_rho"] for v in load_trajectories("val", path).values()],
                            {s: v["log10_rho"] for s, v in test.items()},
                            frame_dt=float(np.diff(next(iter(test.values()))["t"]).mean()),
                            cell_volume=window_volumes(path), blocks=a.blocks, start_every=a.start_every,
                            dmd_ranks=a.dmd_ranks, log=lambda s: print(s, flush=True))
    report["split"] = a.split
    for n, e in report["mean_mae"].items():
        print(f"  MAE {n:14s} +100 M {e[4]:.4f}  +1000 M {e[min(49, len(e) - 1)]:.4f}  "
              f"+{report['leads_M'][-1]:.0f} M {e[-1]:.4f}")
    with open(os.path.join(a.out, f"skill_{a.split}.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
