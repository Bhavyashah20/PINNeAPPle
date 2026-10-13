"""Verification of the rover model against closed forms, the standalone solver and its own balance laws.

* :func:`single_wheel_check` -- every wheel of a full-model state against the standalone ``BekkerWongSolver``
  (adaptive ``quad``): sinkage for the wheel's load and slip, then F_x and M_y at that sinkage.
* :func:`static_sinkage_check` -- the rover at rest on level ground on a pressure-only soil (no shear, front contact
  only) against Bekker's closed form for a rigid wheel, ``W = (3 - n) b k sqrt(D) z^((2n+1)/2) / 3``
  (Bekker 1969), and against the exact integral of the same pressure law.
* :func:`balance_check` -- independent recomputation of the force and moment balance of a state.
* :func:`surrogate_vs_numerical` -- a surrogate against the numerical law at the (slip, sinkage, soil) inputs
  met along a trajectory, with the share of inputs outside the training box.
* :func:`dp_grade_estimate` -- the maximum grade from "drawbar pull = grade resistance" for one wheel carrying
  ``m g cos(a) / 6`` at the limit slip, to compare with :meth:`RoverSimulator.max_slope`.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..numerical_solvers.bekker_wong import BekkerWongSolver
from ..particle_dynamics.terramechanics import SoilParams, WheelParams
from .contact import BekkerWongContact
from .dynamics import RoverSimulator, StepState
from .geometry import RoverParams
from .terrain import LunarTerrain


def single_wheel_check(state: StepState, rover: RoverParams, soils: list[SoilParams],
                       rear_angle_ratio: float = 1.0 / 3.0) -> dict:
    wheel = WheelParams(R=rover.wheel_radius, b=rover.wheel_width)
    rows = []
    for i in range(6):
        if state.Fz[i] <= 0:
            continue
        sol = BekkerWongSolver(soils[state.soil_idx[i]], wheel, rear_angle_ratio=rear_angle_ratio)
        s = float(state.slip[i])
        z_ref = sol.sinkage_from_load(float(state.Fz[i]), s, z_hi=0.5 * wheel.R)
        Fx_ref, _, My_ref = sol.forces(s, z_ref)
        rows.append({"wheel": i, "slip": s, "load_N": float(state.Fz[i]), "z_model_mm": 1e3 * state.sinkage[i],
                     "z_ref_mm": 1e3 * z_ref, "Fx_model": float(state.Fx[i]), "Fx_ref": Fx_ref,
                     "My_model": float(state.My[i]), "My_ref": My_ref})
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-9)  # noqa: E731
    return {"rows": rows,
            "max_rel_sinkage": max(rel(r["z_model_mm"], r["z_ref_mm"]) for r in rows),
            "max_rel_Fx": max(abs(r["Fx_model"] - r["Fx_ref"]) / max(abs(r["load_N"]), 1e-9) for r in rows),
            "max_rel_My": max(rel(r["My_model"], r["My_ref"]) for r in rows)}


def pressure_only(soil: SoilParams) -> SoilParams:
    """The soil's pressure-sinkage law with shear switched off (c = 0, K very large) and the stress peak at the
    bottom of the wheel (a0 = a1 = 0); use with ``rear_angle_ratio=0`` for front-only contact."""
    return replace(soil, c=0.0, K=1e9, a0=0.0, a1=0.0)


def bekker_rigid_wheel_sinkage(W: float, b: float, D: float, k: float, n: float) -> float:
    """Bekker's closed form for the static sinkage of a rigid wheel: z = [3 W / ((3 - n) b k sqrt(D))]^(2/(2n+1))."""
    return (3.0 * W / ((3.0 - n) * b * k * math.sqrt(D))) ** (2.0 / (2.0 * n + 1.0))


def static_sinkage_check(rover: RoverParams | None = None, soil: SoilParams | None = None) -> dict:
    rover = rover or RoverParams()
    soil = pressure_only(soil or SoilParams())
    contact = BekkerWongContact(rover.wheel_radius, rover.wheel_width, rear_angle_ratio=0.0)
    sim = RoverSimulator(rover, LunarTerrain(soil=soil), contact=contact)
    st = sim.run(distance=0.0, v=0.0, dt=0.0, settle=3).states[0]
    k = soil.k_c / rover.wheel_width + soil.k_phi
    solver = BekkerWongSolver(soil, WheelParams(R=rover.wheel_radius, b=rover.wheel_width), rear_angle_ratio=0.0)
    rows = []
    for i in range(6):
        W = float(st.Fz[i])
        rows.append({"wheel": i, "load_N": W, "z_model_mm": 1e3 * st.sinkage[i],
                     "z_integral_mm": 1e3 * solver.sinkage_from_load(W, 0.0, z_hi=0.5 * rover.wheel_radius),
                     "z_closed_form_mm": 1e3 * bekker_rigid_wheel_sinkage(W, rover.wheel_width,
                                                                         2 * rover.wheel_radius, k, soil.n)})
    zm = np.array([r["z_model_mm"] for r in rows])
    zi = np.array([r["z_integral_mm"] for r in rows])
    zc = np.array([r["z_closed_form_mm"] for r in rows])
    return {"rows": rows, "total_load_N": float(st.Fz.sum()), "weight_N": rover.mass * sim.g,
            "max_rel_vs_integral": float(np.max(np.abs(zm - zi) / zi)),
            "max_rel_vs_closed_form": float(np.max(np.abs(zm - zc) / zc)), "state": st}


def balance_check(state: StepState, rover: RoverParams) -> dict:
    """Net force (including gravity, the lateral reaction, the tow load and the d'Alembert force) and net moment
    about the centre of mass, recomputed from the stored wheel forces; scaled by m g and m g L."""
    m, g = state.balance["mass"], state.balance["g"]
    R = state.pose.rotation()
    bx, by = R[:, 0], R[:, 1]
    F = state.Fx[:, None] * state.tangent + state.Fz[:, None] * state.normal
    Flat = state.balance["lateral"]
    G = state.balance["com"]
    pull = state.pull + state.balance["tow"]
    net = F.sum(0) + Flat.sum(0) + np.array([0, 0, -m * g]) - pull * bx - m * state.balance["accel"] * bx
    M = (np.cross(state.wheel_centre - G, F) + np.cross(state.contact_point - G, Flat)).sum(0) \
        - state.My.sum() * by
    L = rover.wheelbase
    return {"force": float(np.linalg.norm(net) / (m * g)),
            "moment_roll_pitch": float(np.linalg.norm([M @ bx, M @ by]) / (m * g * L)),
            "moment_yaw": float(abs(M @ R[:, 2]) / (m * g * L))}


def surrogate_vs_numerical(states, surrogate, numerical, soils: list[SoilParams]) -> dict:
    """Errors of ``surrogate`` against ``numerical`` at every loaded (slip, sinkage) of the given states."""
    S = np.concatenate([s.slip[s.sinkage > 0] for s in states])
    Z = np.concatenate([s.sinkage[s.sinkage > 0] for s in states])
    idx = np.concatenate([s.soil_idx[s.sinkage > 0] for s in states])
    ref = numerical(S, Z, soils, idx)
    sur = surrogate.predict(S, Z)
    inside = surrogate.in_range(S, Z)
    err = sur - ref
    scale = np.maximum(np.abs(ref[:, 1]), 1e-9)               # relative to the wheel load
    out = {"n": int(len(S)), "n_outside": int((~inside).sum()), "fraction_outside": float((~inside).mean()),
           "slip_range_seen": (float(S.min()), float(S.max())), "sinkage_range_seen_mm": (1e3 * Z.min(), 1e3 * Z.max())}
    for j, name in enumerate(("Fx", "Fz", "My")):
        for tag, mask in (("inside", inside), ("outside", ~inside)):
            if mask.any():
                out[f"{name}_rmse_{tag}"] = float(np.sqrt(np.mean(err[mask, j] ** 2)))
                out[f"{name}_max_abs_{tag}"] = float(np.max(np.abs(err[mask, j])))
        out[f"{name}_max_rel_load"] = float(np.max(np.abs(err[:, j]) / scale))
    return out


def dp_grade_estimate(rover: RoverParams, soil: SoilParams, s_limit: float, g: float = 1.62,
                      rear_angle_ratio: float = 1.0 / 3.0, tol: float = 0.01) -> dict:
    """Grade where six equally loaded wheels at slip ``s_limit`` give drawbar pull = m g sin(a) (bisection)."""
    sol = BekkerWongSolver(soil, WheelParams(R=rover.wheel_radius, b=rover.wheel_width),
                           rear_angle_ratio=rear_angle_ratio)
    m = rover.mass

    def excess(a_deg):
        a = math.radians(a_deg)
        W = m * g * math.cos(a) / 6
        z = sol.sinkage_from_load(W, s_limit, z_hi=0.5 * rover.wheel_radius)
        return 6 * sol.forces(s_limit, z)[0] - m * g * math.sin(a)

    lo, hi = 0.0, 60.0
    if excess(lo) < 0:
        return {"max_slope_deg": 0.0}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    a = 0.5 * (lo + hi)
    return {"max_slope_deg": a, "dp_over_weight": math.tan(math.radians(a))}


__all__ = ["single_wheel_check", "static_sinkage_check", "balance_check", "surrogate_vs_numerical",
           "dp_grade_estimate", "bekker_rigid_wheel_sinkage", "pressure_only"]
