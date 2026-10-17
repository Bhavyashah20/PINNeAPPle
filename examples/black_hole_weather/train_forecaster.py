"""Train the black-hole weather forecaster (U-Net of Duarte, Nemmen & Navarro 2022) on simulations from simulate.py.

Two set-ups, as in the paper:

* one-sim: a single run, split in time 70 / 10 / 20 (train / validation / test);
* multi-sim: several runs for training, a withheld run with different physics for testing
  (the paper withholds PL0SS3, alpha = 0.3; here PL0SS0.3).

    python examples/black_hole_weather/train_forecaster.py --runs data/bh --train PL0SS0.1 --out runs/bh_one
    python examples/black_hole_weather/train_forecaster.py --runs data/bh --train PL0SS0.1 PL0ST0.01 PL0.2SS0.1 \\
        --out runs/bh_multi

From the zoo dataset (whole trajectories: train seeds 0-3, validation seed 4; frames are already 20 M apart and
cropped to the forecast window), with the PINNeAPPle additions (rollout loss, input noise, mass penalty):

    python examples/black_hole_weather/train_forecaster.py --dataset blackhole_riaf_density_256 --out runs/bh256 \\
        --filters 16 --residual --rollout-blocks 3 --noise 0.002 --mass-weight 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import load_run  # noqa: E402

from pinneapple_physics.blackhole.forecast import (  # noqa: E402
    DensityCodec,
    TrainConfig,
    make_blocks,
    train_forecaster,
)


def split_one(n: int):
    a, b = int(0.7 * n), int(0.8 * n)
    return slice(0, a), slice(a, b), slice(b, n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--train", nargs="+")
    ap.add_argument("--dataset", help="zoo dataset name or path to its data.npz (instead of --runs/--train)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2, help="use every n-th saved frame (10 M -> 20 M)")
    ap.add_argument("--r-cells", type=int, default=96)
    ap.add_argument("--theta-trim", type=int, default=8)
    ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--residual", action="store_true", help="predict the change from the last frame")
    ap.add_argument("--padding", default="zeros", choices=["zeros", "replicate"], help="convolution padding")
    ap.add_argument("--loss", default="multi", choices=["multi", "regional"],
                    help="paper multi-sim loss, or the one-sim regional loss (Table 1 weights)")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient clipping (0: off, as in the paper)")
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rollout-blocks", type=int, default=1, help="unroll this many blocks in the loss")
    ap.add_argument("--val-blocks", type=int, default=1, help="select the checkpoint on this many unrolled blocks")
    ap.add_argument("--noise", type=float, default=0.0, help="input noise std (normalised units)")
    ap.add_argument("--mass-weight", type=float, default=0.0, help="weight of the window-mass consistency penalty")
    ap.add_argument("--mass-rate-weight", type=float, default=0.0,
                    help="weight of the penalty on frame-to-frame window-mass changes against the truth")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (fast on CPUs with AMX)")
    ap.add_argument("--init", help="start from this forecaster.pt (fine-tuning)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="random subset of blocks per epoch (0: all)")
    ap.add_argument("--val-stride", type=int, default=1, help="use every n-th validation block")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.dataset:
        return main_dataset(a)

    runs = {name: load_run(os.path.join(a.runs, name)) for name in a.train}
    rho = {k: v["frames"][::a.stride, 0] for k, v in runs.items()}
    one = len(a.train) == 1
    if one:
        tr, va, _ = split_one(len(rho[a.train[0]]))
        train_frames = {a.train[0]: rho[a.train[0]][tr]}
        val_frames = {a.train[0]: rho[a.train[0]][va]}
    else:                                    # last 10 % of each run for validation
        train_frames = {k: v[:int(0.9 * len(v))] for k, v in rho.items()}
        val_frames = {k: v[int(0.9 * len(v)):] for k, v in rho.items()}
    codec = DensityCodec.fit(np.concatenate(list(train_frames.values())), a.r_cells, a.theta_trim)
    X, Y = zip(*[make_blocks(codec.encode(v)) for v in train_frames.values()], strict=True)
    Xv, Yv = zip(*[make_blocks(codec.encode(v)) for v in val_frames.values()], strict=True)
    X, Y, Xv, Yv = (np.concatenate(z) for z in (X, Y, Xv, Yv))
    print(f"train {X.shape}  val {Xv.shape}  codec {codec}", flush=True)
    cfg = TrainConfig(filters=a.filters, epochs=a.epochs, batch_size=a.batch_size, alpha=a.alpha, lr=a.lr, residual=a.residual, loss=a.loss, clip=a.clip,
                      max_minutes=a.max_minutes)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "setup.json"), "w") as f:
        json.dump({"train": a.train, "stride": a.stride, "frame_dt": float(np.diff(runs[a.train[0]]["t"][::a.stride]).mean()),
                   "one_sim": one, "n_train": int(len(X)), "n_val": int(len(Xv))}, f, indent=1)
    train_forecaster(X, Y, Xv, Yv, cfg, codec, a.out, log=lambda s: print(s, flush=True))


def main_dataset(a):
    from pinneapple_physics.blackhole.dataset import load_trajectories, window_volumes
    from pinneapple_physics.blackhole.forecast import train_on_trajectories
    path = a.dataset if os.path.exists(a.dataset) else None   # a data.npz, a folder of <split>.npz, or the zoo name
    tr, va = load_trajectories("train", path), load_trajectories("val", path)
    cfg = TrainConfig(filters=a.filters, epochs=a.epochs, batch_size=a.batch_size, alpha=a.alpha, lr=a.lr,
                      residual=a.residual, loss=a.loss, clip=a.clip, max_minutes=a.max_minutes, seed=a.seed,
                      rollout_blocks=a.rollout_blocks, noise=a.noise, mass_weight=a.mass_weight, bf16=a.bf16,
                      mass_rate_weight=a.mass_rate_weight, samples_per_epoch=a.samples_per_epoch, padding=a.padding)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "setup.json"), "w") as f:
        json.dump({"dataset": a.dataset, "train_seeds": sorted(tr), "val_seeds": sorted(va),
                   "frame_dt": float(np.diff(next(iter(tr.values()))["t"]).mean()), "stride": 1, "one_sim": False,
                   "args": vars(a)}, f, indent=1)
    train_on_trajectories([v["log10_rho"] for v in tr.values()], [v["log10_rho"] for v in va.values()], cfg, a.out,
                          cell_volume=window_volumes(path), val_blocks=a.val_blocks, val_stride=a.val_stride,
                          init=a.init,
                          log=lambda s: print(s, flush=True))

if __name__ == "__main__":
    main()
