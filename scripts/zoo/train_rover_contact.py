"""Retrain the zoo's ``rover_contact_grc1_viper`` model (wheel-soil contact PINN surrogate).

Same recipe as ``examples/use_cases/terramechanics/lunar_rover_3d.py``: Wong-Reece reference, GRC-1 lunar simulant,
NASA VIPER wheel (parametric, public figures), slip -0.4..0.9, sinkage 2..60 mm, seed 0, but 16000 epochs
(the example's 4000 leave the held-out RMSE ~4x higher; 16000 take ~5 min on one CPU core). Writes the
TorchScript weights + JSON next to each other and the held-out test set (disjoint LHS) as an npz.

    python scripts/zoo/train_rover_contact.py OUT_DIR [EPOCHS]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from pinneapple_simulation.terramechanics import WongReeceModel, get_soil
from pinneapple_simulation.terramechanics.surrogate import _lhs, train_contact_surrogate
from pinneapple_simulation.vehicle import viper

SLIP, SINK = (-0.4, 0.9), (0.002, 0.06)


def main(out: str, epochs: int = 16000, seed: int = 0) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wr = WongReeceModel(get_soil("GRC-1"), viper().wheel)
    t0 = time.time()
    sur, rep = train_contact_surrogate(wr, SLIP, SINK, epochs=epochs, seed=seed)
    rep["train_seconds"] = time.time() - t0
    sur.save(str(out_dir / "rover_contact"))
    lo, hi = np.array([SLIP[0], SINK[0]]), np.array([SLIP[1], SINK[1]])
    Xt = _lhs(2000, lo, hi, seed + 1000)                      # the held-out set of train_contact_surrogate
    Yt = wr.forces(Xt[:, 0], Xt[:, 1]).as_array()
    Xtr = _lhs(2000, lo, hi, seed + 1)
    Ytr = wr.forces(Xtr[:, 0], Xtr[:, 1]).as_array()
    np.savez_compressed(out_dir / "rover_contact_data.npz", test_x=Xt, test_y=Yt, train_x=Xtr, train_y=Ytr)
    (out_dir / "rover_contact_report.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "history"},
                                                                  indent=1))
    print(json.dumps(rep["test"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
