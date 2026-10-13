"""Train the learned NEO propagator shipped in pinneapple_physics/astro/data/neural_propagator.

    python scripts/astro/train_neural_propagator.py [n_bodies]
"""
import json
import os
import sys
import time

import numpy as np

from pinneapple_physics.astro import neural_propagator as npg

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
t0 = time.time()
cache = os.environ.get("NP_CACHE", f"/tmp/neural_propagator_dataset_{n}.npz")
X, Y, meta, tau = npg.make_dataset(n_bodies=n, seed=0, cache=cache)
print(f"dataset {X.shape} in {time.time() - t0:.0f} s", flush=True)
n_tr = int(0.85 * n)
t0 = time.time()
model = npg.NeuralPropagator.train(X[:n_tr], Y[:n_tr], epochs=1200)
print(f"trained in {time.time() - t0:.0f} s, val loss {model.info['val_loss']:.3g}", flush=True)
held = [npg._body(el) for el in meta[n_tr:]]
rows = npg.evaluate(model, held)
model.info.update(n_bodies=n, n_train=n_tr, n_test=n - n_tr, epoch_jd=npg.EPOCH, span_days=npg.SPAN_DAYS,
                  test_max_err_km_median=float(np.median([r["max_err_km"] for r in rows])),
                  test_max_err_km_p90=float(np.percentile([r["max_err_km"] for r in rows], 90)),
                  test_kepler_max_err_km_median=float(np.median([r["kepler_max_err_km"] for r in rows])))
model.save()
json.dump([{k: v for k, v in r.items() if k not in ("t_days", "err_km", "kepler_err_km")} for r in rows],
          open(os.path.join(npg.MODEL_DIR, "heldout.json"), "w"), indent=1)
print(json.dumps(model.info, indent=1))
