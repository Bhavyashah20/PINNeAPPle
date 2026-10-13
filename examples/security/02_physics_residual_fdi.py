"""Physics residuals against false data injection: three-tank water network, physics vs data-driven detectors.

A grey-box twin (three storage balances, two valve laws, constants identified from one week of trusted data) is
compared with three detectors that learn only from data: a linear autoregressive twin of all sensors, PCA (Q
statistic) and an isolation forest. Every detector uses the same CUSUM alarm rule and is calibrated to one false
alarm per day on a clean week it was not fitted on. Then six attacks (bias, ramp, stuck sensor, delayed replay of one
meter, metering fraud, coordinated replay of every sensor) are injected in 20 four-hour episodes of a ten-day test
trace, in the nominal operating regime and in a shifted one (25 % more demand, a different pump band) that no
detector saw during fitting.

    python examples/security/02_physics_residual_fdi.py [--seeds 0 1 2] [--out results.json]

About 50 s per seed on a laptop CPU (NumPy; scikit-learn for the isolation forest).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pinneapple_security.cps_bench import ATTACKS, run_benchmark


def _fmt(v, nd=2):
    return "  -  " if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [run_benchmark(seed=s) for s in a.seeds]
    dets = list(runs[0]["results"])
    for regime in ("nominal", "shifted"):
        print(f"\n== {regime} regime: mean over seeds {a.seeds} ==")
        fa = {d: np.mean([r["results"][d][regime]["false_alarms_per_day_clean"] for r in runs]) for d in dets}
        print("false alarms per day on clean data:", {d: round(v, 2) for d, v in fa.items()})
        print(f"{'attack':20s} " + " ".join(f"{d:>26s}" for d in dets))
        print(f"{'':20s} " + " ".join(f"{'AUC  det  delay  loc':>26s}" for _ in dets))
        for kind in ATTACKS:
            cells = []
            for d in dets:
                rows = [r["results"][d][regime][kind] for r in runs]
                delays = [x["median_delay_min"] for x in rows if x["median_delay_min"] is not None]
                locs = [x["localisation"] for x in rows if x["localisation"] is not None]
                cells.append(f"{np.mean([x['auc'] for x in rows]):.3f} {np.mean([x['detection_rate'] for x in rows]):4.2f} "
                             f"{_fmt(np.median(delays) if delays else None, 0):>6s} {_fmt(np.mean(locs) if locs else None):>5s}")
            print(f"{kind:20s} " + " ".join(f"{c:>26s}" for c in cells))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(runs, f, indent=2, default=float)


if __name__ == "__main__":
    main()
