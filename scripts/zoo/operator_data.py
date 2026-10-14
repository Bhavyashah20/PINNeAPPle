"""Operator-learning benchmark datasets of the zoo: 1-D Burgers, 2-D Darcy, 2-D Navier-Stokes (vorticity).

Command-line front end of ``pinneapple_simulation.numerical_solvers.operator_datasets`` (the recipes: problems and
input distributions of Li et al. 2021, solved with PINNeAPPle's reference solvers):

    python scripts/zoo/operator_data.py convergence --out DIR          # resolution / time-step study (evidence)
    python scripts/zoo/operator_data.py generate --out DIR [--only burgers|darcy|ns] [--workers 4]

``generate`` writes ``DIR/<name>/{train,test}[-NN].npz`` (Hub-only: too large to bundle) plus ``DIR/<name>/meta.json``;
``scripts/zoo/build_assets.py --hub-only DIR`` turns them into manifest entries. Every input is drawn in the parent
process from a fixed seed, so the data do not depend on the number of workers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.numerical_solvers.operator_datasets import convergence, generate  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["convergence", "generate"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=["burgers", "darcy", "ns"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    if a.command == "convergence":
        convergence(a.out, a.workers, only=a.only)
    else:
        generate(a.out, a.workers, a.only)


if __name__ == "__main__":
    main()
