"""The OpenFOAM training set of the airfoil surrogates: the run plan, one run, and the steady value of a run.

A run record (``<out>/runs/<id>.json``) holds the job (``id``, ``set`` = "lhs" / "reference" / "verify",
``shape_id``, ``shape`` = 6 CST weights, ``alpha`` in degrees), ``reynolds`` and what :func:`.case.run_case` returned
(``status``, ``Cl``, ``Cd``, ``Cm``, their oscillation ``*_osc`` over the last writes, ``history`` rows
``(iteration, Cl, Cd, Cm)``, the wall ``Cp`` / ``Cf``). ``<id>.npz`` holds the cell fields ``p``, ``Ux``, ``Uy``,
``nut`` on the shared O-grid. The CLI is ``apps/aero_optimizer/tools/generate_dataset.py``.
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .case import read_internal, run_case
from .geometry import BOUNDS, REFERENCE, latin_hypercube, valid

RE = 4e6                                                           # chord Reynolds number of every run
ALPHA_BINS = [(-2, 2), (2, 6), (6, 10), (10, 14)]                  # one random angle per bin and shape (degrees)
REFERENCE_ALPHAS = (-2, 0, 2, 4, 6, 8, 10, 12, 14)                 # full polars of the NACA references
USABLE = ("converged", "steady-ish", "unsteady")                   # run statuses with a usable steady value
MAX_CL_OSC = 0.1                                                   # Cl swing over the kept writes; above: deep stall


def plan(n_shapes: int, n_alpha: int, seed: int = 7) -> list[dict[str, Any]]:
    """The run list: ``n_shapes`` valid Latin-hypercube shapes x ``n_alpha`` angles, plus the NACA reference polars.

    Angles are drawn uniformly inside equal bins of [-2, 14] degrees (``ALPHA_BINS`` for ``n_alpha == 4``), one per
    bin, so every shape covers the whole range. Deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    shapes = []
    while len(shapes) < n_shapes:
        for s in latin_hypercube(n_shapes, BOUNDS, rng):
            if valid(s)[0] and len(shapes) < n_shapes:
                shapes.append(s)
    jobs = []
    for k, s in enumerate(shapes):
        bins = ALPHA_BINS if n_alpha == 4 else [(-2 + 16 * i / n_alpha, -2 + 16 * (i + 1) / n_alpha) for i in range(n_alpha)]
        for b, (lo, hi) in enumerate(bins):
            jobs.append({"id": f"s{k:03d}_a{b}", "set": "lhs", "shape_id": f"s{k:03d}", "shape": s.tolist(),
                         "alpha": float(rng.uniform(lo, hi))})
    for name, s in REFERENCE.items():
        for a in REFERENCE_ALPHAS:
            jobs.append({"id": f"{name.replace(' ', '')}_a{a:+03d}", "set": "reference", "shape_id": name,
                         "shape": s.tolist(), "alpha": float(a)})
    return jobs


def run_job(job: Mapping[str, Any], out: str) -> tuple[str, Any, Any]:
    """Run one job with OpenFOAM and write ``out/runs/<id>.json`` (+ ``.npz`` cell fields when it ran).

    ``out/runs`` must exist. The case directory ``out/cases/<id>`` is deleted afterwards. Returns
    ``(id, status, seconds)``.
    """
    case = os.path.join(out, "cases", job["id"])
    r = run_case(case, np.array(job["shape"]), job["alpha"], reynolds=RE)
    rec = {**job, "reynolds": RE, **{k: v for k, v in r.items() if k not in ("time_dir",)}}
    d = r.get("time_dir")
    if d:
        p = read_internal(os.path.join(d, "p"))
        U = read_internal(os.path.join(d, "U"))
        nut = read_internal(os.path.join(d, "nut"))
        np.savez_compressed(os.path.join(out, "runs", job["id"] + ".npz"), p=p.astype(np.float32),
                            Ux=U[:, 0].astype(np.float32), Uy=U[:, 1].astype(np.float32), nut=nut.astype(np.float32))
    with open(os.path.join(out, "runs", job["id"] + ".json"), "w") as f:
        json.dump(rec, f)
    shutil.rmtree(case, ignore_errors=True)
    return job["id"], r.get("status"), r.get("seconds")


def steady_values(runs: Iterable[Mapping[str, Any]], max_osc: float = MAX_CL_OSC) -> list[dict[str, Any]]:
    """The steady Cl, Cd, Cm of each run, the one rule used for training and for verification.

    Runs with status "unsteady" (oscillating, separated flow) get the mean of their last 5 kept writes
    (``history`` columns 1..3), or are dropped when their Cl swing ``Cl_osc`` exceeds ``max_osc`` (deep stall: no
    meaningful steady value). Other runs pass unchanged. Returns new dicts; the inputs are not modified.
    """
    out = []
    for r in runs:
        if r.get("status") == "unsteady":
            if r.get("Cl_osc", 9) > max_osc:
                continue
            H = np.array(r["history"])[-5:]
            r = {**r, "Cl": float(H[:, 1].mean()), "Cd": float(H[:, 2].mean()), "Cm": float(H[:, 3].mean())}
        out.append(dict(r))
    return out
