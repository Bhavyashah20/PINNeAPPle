"""Checks of a Newton solver against closed-form rigid-body results.

Each case builds a small scene, runs it and compares one quantity with its analytic value. Run them before
trusting a solver setting (``dt``, iterations, solver) for a study: ``validate()`` returns one row per case
with the simulated value, the reference, the error and whether it is within the stated tolerance.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .runner import simulate
from .scene import RigidScene

G = 9.81


def free_fall(solver="xpbd", dt=1e-3, **kw) -> dict:
    """Sphere released from rest, no ground: z(t) = z0 - g t^2 / 2. Error = max |dz| / drop height."""
    s = RigidScene(gravity=G)
    s.sphere("ball", 0.05, pos=(0, 0, 2.0))
    tr = simulate(s, 0.5, dt=dt, solver=solver, record_every=10, **kw)
    exact = 2.0 - 0.5 * G * tr.t ** 2
    err = np.abs(tr.pos[:, 0, 2] - exact).max() / (0.5 * G * 0.5 ** 2)
    return _row("free fall", "max |z - z_exact| / drop", tr.pos[-1, 0, 2], exact[-1], err, 0.01, tr)


def sliding_friction(solver="xpbd", dt=1e-3, mu=0.4, v0=2.0, **kw) -> dict:
    """Box sliding on the ground with Coulomb friction stops after v0^2 / (2 mu g)."""
    s = RigidScene(gravity=G).ground(mu=mu)
    s.box("block", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(v0, 0, 0), mu=mu)
    tr = simulate(s, 1.2 * v0 / (mu * G), dt=dt, solver=solver, record_every=10, **kw)
    ref = v0 ** 2 / (2 * mu * G)
    d = tr.pos[-1, 0, 0]
    return _row("sliding with friction", "stopping distance [m]", d, ref, abs(d - ref) / ref, 0.03, tr)


def pendulum_period(solver="xpbd", dt=1e-3, length=1.0, amp_deg=5.0, **kw) -> dict:
    """Small sphere on a hinge, small amplitude: T = 2 pi sqrt(I_pivot / (m g L))."""
    r, a = 0.05, np.radians(amp_deg)
    s = RigidScene(gravity=G)
    b = s.sphere("bob", r, pos=(length * np.sin(a), 0, 2.0 - length * np.cos(a)))
    s.revolute(-1, b, anchor=(0, 0, 2.0), axis=(0, 1, 0))
    ref = 2 * np.pi * np.sqrt((length ** 2 + 0.4 * r ** 2) / (G * length))
    tr = simulate(s, 3.2 * ref, dt=dt, solver=solver, record_every=1, **kw)
    x, t = tr.pos[:, 0, 0], tr.t
    i = np.where((x[:-1] > 0) & (x[1:] <= 0))[0]
    tc = t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1])         # interpolated zero crossings
    T = float(np.diff(tc).mean()) if len(tc) > 1 else float("nan")
    return _row("pendulum period", "period [s]", T, ref, abs(T - ref) / ref, 0.01, tr)


def bounce(solver="xpbd", dt=1e-4, e=0.5, **kw) -> dict:
    """Ball dropped on the ground with restitution e: first rebound height = e^2 x drop height."""
    s = RigidScene(gravity=G).ground(mu=0.0, restitution=e)
    s.sphere("ball", 0.05, pos=(0, 0, 1.05), mu=0.0, restitution=e)
    tr = simulate(s, 1.0, dt=dt, solver=solver, record_every=10, **kw)
    z = tr.pos[:, 0, 2] - 0.05
    k = int(np.argmin(z[: int(0.6 / (10 * dt))]))                    # first impact (t ~ 0.45 s)
    h = float(z[k:].max())
    ref = e ** 2 * 1.0
    return _row("bounce with restitution", "rebound height [m]", h, ref, abs(h - ref) / ref, 0.10, tr)


def torque_free_spin(solver="xpbd", dt=1e-3, **kw) -> dict:
    """Box spinning about a tilted axis without gravity: the angular momentum vector is conserved."""
    s = RigidScene(gravity=0.0)
    s.box("brick", (0.2, 0.1, 0.05), pos=(0, 0, 1), omega=(0.2, 5.0, 0.3))
    tr = simulate(s, 2.0, dt=dt, solver=solver, record_every=10, **kw)
    from .scene import _rot
    L = np.array([(_rot(q) @ tr.inertia[0] @ _rot(q).T) @ w for q, w in zip(tr.quat[:, 0], tr.omega[:, 0], strict=True)])
    drift = float(np.linalg.norm(L - L[0], axis=1).max() / np.linalg.norm(L[0]))
    return _row("torque-free rotation", "max |L - L0| / |L0|", drift, 0.0, drift, 0.01, tr)


def resting_contact(solver="xpbd", dt=1e-3, **kw) -> dict:
    """Box at rest on the ground for 2 s: it should neither sink nor drift."""
    s = RigidScene(gravity=G).ground(mu=0.5)
    s.box("block", (0.1, 0.1, 0.1), pos=(0, 0, 0.1))
    tr = simulate(s, 2.0, dt=dt, solver=solver, record_every=10, **kw)
    sink = float(0.1 - tr.pos[:, 0, 2].min())
    drift = float(np.linalg.norm(tr.pos[-1, 0, :2] - tr.pos[0, 0, :2]))
    worst = max(sink, drift)
    return _row("resting contact", "max sink or drift [m]", worst, 0.0, worst / 0.1, 0.02, tr)


CASES: dict[str, Callable] = {f.__name__: f for f in
                               (free_fall, sliding_friction, pendulum_period, bounce, torque_free_spin, resting_contact)}


def validate(solver: str = "xpbd", cases: list[str] | None = None, log=print, **kw) -> list[dict]:
    """Run the analytic checks for one solver; returns rows (without the trajectories) and logs a table."""
    rows = []
    for name in cases or list(CASES):
        r = CASES[name](solver=solver, **kw)
        r.pop("trajectory", None)
        rows.append(r)
        if log:
            log(f"{r['case']:<24} {r['quantity']:<28} sim {r['simulated']:.4g}  ref {r['reference']:.4g}  "
                f"err {r['error']:.2%}  tol {r['tolerance']:.0%}  {'ok' if r['passed'] else 'FAIL'}")
    return rows


def _row(case, quantity, sim, ref, err, tol, tr) -> dict:
    return {"case": case, "quantity": quantity, "simulated": float(sim), "reference": float(ref),
            "error": float(err), "tolerance": tol, "passed": bool(err <= tol), "solver": tr.solver, "dt": tr.dt,
            "wall_time_s": tr.wall_time_s, "trajectory": tr}
