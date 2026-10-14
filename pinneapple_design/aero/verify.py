"""Check the design search with OpenFOAM: pick designs along the Pareto front, run their polars, compare.

The aircraft numbers of a design (top speed, CO2 per 100 km, stall speed, cruise L/D) are recomputed from the
OpenFOAM polar exactly as from the surrogate polar (:func:`.aircraft.evaluate`), so the comparison isolates the
surrogate error. The runs use the training-set record format (set "verify", :mod:`.dataset`), so they can be added
to the next training (active learning). The CLI is ``apps/aero_optimizer/tools/verify.py``.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .aircraft import Aircraft, Polar, Requirements, evaluate
from .dataset import steady_values
from .geometry import properties
from .optimize import WING_MASS_PER_M2

CFD_ALPHAS = [-2, 0, 2, 4, 6, 8, 10, 12, 14]                      # angles of every verification polar (degrees)
KEYS = ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "clmax", "feasible")

__all__ = ["CFD_ALPHAS", "pick", "steady_values", "cfd_eval", "verification_jobs", "compare_design"]


def pick(front: Sequence[int], n: int) -> list[int]:
    """``n`` designs spread evenly along the front (in its order, by top speed), always including both ends."""
    if len(front) <= n:
        return list(front)
    idx = np.unique(np.round(np.linspace(0, len(front) - 1, n)).astype(int))
    return [front[i] for i in idx]


def cfd_eval(runs: Sequence[Mapping[str, Any]], shape: np.ndarray, area: float, ac: Aircraft,
             req: Requirements) -> dict[str, Any]:
    """Aircraft performance and checks from the OpenFOAM polar of ``shape`` with wing area ``area`` (m^2).

    The wing mass changes by ``WING_MASS_PER_M2`` per m^2 from ``ac.wing_area`` (as in the search). Adds ``polar``
    (alpha, cl, cd, cm and the run statuses).
    """
    runs = sorted(steady_values(runs), key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in runs])
    cl, cd, cm = (np.array([r[k] for r in runs]) for k in ("Cl", "Cd", "Cm"))
    acx = dataclasses.replace(ac, wing_area=area, mass=ac.mass + WING_MASS_PER_M2 * (area - ac.wing_area))
    e = evaluate(properties(shape), Polar(a, cl, cd, cm), acx, req)
    e["polar"] = {"alpha": a.tolist(), "cl": cl.tolist(), "cd": cd.tolist(), "cm": cm.tolist(),
                  "status": [r["status"] for r in runs]}
    return e


def verification_jobs(designs: Sequence[Mapping[str, Any]], chosen: Sequence[int], round_: int):
    """The OpenFOAM jobs (one per design and angle of ``CFD_ALPHAS``) and ``[(design_id, design)]``.

    Design ids are ``r<round>_d<k>``; job ids ``<design_id>_a<+alpha>``, in the :mod:`.dataset` job format.
    """
    jobs, meta = [], []
    for n, i in enumerate(chosen):
        d = designs[i]
        did = f"r{round_}_d{n}"
        meta.append((did, d))
        for al in CFD_ALPHAS:
            jobs.append({"id": f"{did}_a{al:+03d}", "set": "verify", "shape_id": did, "shape": d["shape"],
                         "alpha": float(al), "wing_area": d["wing_area"]})
    return jobs, meta


def compare_design(engine, did: str, d: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], ac: Aircraft,
                   req: Requirements) -> dict[str, Any]:
    """Surrogate vs OpenFOAM for one design of the search.

    ``surrogate`` uses the MLP ensemble at the CFD angles (same cl_max resolution as the CFD), ``surrogate_fine``
    is the search's own record (1 degree polar), ``openfoam`` comes from :func:`cfd_eval` of ``runs``.
    """
    cfd = cfd_eval(runs, np.array(d["shape"]), d["wing_area"], ac, req)
    co = engine.mlp_coefficients(np.array(d["shape"])[None], np.array(CFD_ALPHAS, float))
    acx = dataclasses.replace(ac, wing_area=d["wing_area"], mass=d["mass"])
    sur = evaluate(d["props"], Polar(np.array(CFD_ALPHAS, float), co["cl"][0], co["cd"][0], co["cm"][0]), acx, req)
    return {"id": did, "shape": d["shape"], "wing_area": d["wing_area"], "props": d["props"],
            "surrogate": {k: sur[k] for k in KEYS + ("violations",)},
            "surrogate_fine": {k: d[k] for k in KEYS + ("trust",)},
            "surrogate_polar": {"alpha": CFD_ALPHAS, "cl": co["cl"][0].tolist(), "cd": co["cd"][0].tolist(),
                                "cm": co["cm"][0].tolist()},
            "openfoam": {k: cfd[k] for k in KEYS + ("violations", "polar")}}
