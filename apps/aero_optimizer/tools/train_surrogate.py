"""Train the surrogates on the OpenFOAM runs and measure them on designs they never saw.

    python apps/aero_optimizer/tools/train_surrogate.py DATA_DIR OUT.pt [--gnn-minutes 60] [--extra DIR ...]

Split by shape (not by run): 15 % of the LHS shapes are held out with all their angles; the four NACA airfoils are
a second, independent test. Writes OUT.pt (both models + normalisation) and OUT.metrics.json. The training itself is
``pinneapple_design.aero.train``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.train import load_runs, train_surrogates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--gnn-minutes", type=float, default=60)
    ap.add_argument("--gnn-from", default=None, help="reuse the graph network (and its metrics) of an earlier bundle")
    a = ap.parse_args()
    recs = load_runs([a.data] + a.extra)
    bundle, res = train_surrogates(recs, a.gnn_minutes, a.gnn_from, log=lambda s: print(s, flush=True))
    torch.save(bundle, a.out)
    with open(a.out.replace(".pt", ".metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
