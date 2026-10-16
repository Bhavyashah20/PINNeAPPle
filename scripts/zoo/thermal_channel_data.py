"""Heated-channel CFD dataset of the zoo (``thermal_channel_cfd``): solver verification and the 82 trajectories.

    python scripts/zoo/thermal_channel_data.py verify --out DIR      # Poiseuille, Nusselt, three-grid GCI -> verify.json
    python scripts/zoo/thermal_channel_data.py generate --out DIR    # DIR/thermal_channel_cfd/<split>[-NN].npz

Command-line front end of ``pinneapple_simulation.numerical_solvers.thermal_channel_datasets`` (the recipe of
PINNeAPPle-Benchmark experiment 05 on the library's ThermalChannel2D solver).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.numerical_solvers.thermal_channel_datasets import (  # noqa: E402
    generate,
    verify,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "generate"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.command == "verify":
        (a.out / "thermal_channel_verify.json").write_text(json.dumps(verify(), indent=1) + "\n")
    else:
        generate(a.out, a.workers)


if __name__ == "__main__":
    main()
