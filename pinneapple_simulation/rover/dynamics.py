"""Rocker-bogie rover on lunar terrain with Bekker-Wong wheel forces.

Model
-----
* Rigid body (all mass in the body, links and wheels massless) carried by two rockers through frictionless pivots
  and a differential (right rocker angle = - left rocker angle), each rocker carrying a bogie; six driven wheels.
* Wheel-soil forces from Bekker-Wong (:mod:`.contact`): each wheel's sinkage is its penetration into the terrain
  (``R`` minus the distance from the axle to the local surface), its slip comes from the commanded wheel speed and
  the wheel-centre velocity along the terrain, and the force (F_x along the terrain, F_z along the normal) acts at
  the axle together with the soil torque -M_y about it.
* Quasi-static in heave, pitch, roll and suspension: at every step the unknowns
  ``(z, pitch, roll, rocker, bogie_L, bogie_R)`` satisfy the normal-force balance, the roll and pitch moment balance
  about the centre of mass, the moment balance of each bogie about its pivot and the differential condition (equal
  pivot moments on both rockers). The longitudinal direction is integrated with a backward-Euler step.
* Lateral forces are not given by Bekker-Wong; a lateral reaction proportional to the wheel loads closes the
  lateral force balance and the rover follows its straight path (no side slip, no yaw dynamics).

Drive modes
-----------
``speed``: the rover holds the speed ``v``; the common wheel speed (equivalently the commanded slip ``s_cmd``,
omega = v / (R (1 - s_cmd))) is an unknown fixed by the longitudinal balance. No solution with
``s_cmd <= s_limit`` means traction failure (the rover cannot hold the speed).
``slip``: the rover holds ``v`` and the wheels turn at omega = v / (R (1 - s_cmd)) for a given ``s_cmd``; the
external pull (drawbar pull of the whole rover) is what is left of the longitudinal balance -- a drawbar-pull test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contact import BekkerWongContact
from .geometry import WHEEL_NAMES, Pose, RoverParams, link_frames
from .terrain import LunarTerrain

LUNAR_G = 1.62     # [m/s^2], as in SoilParams


def _cross(a, b):
    """Row-wise cross product of (..., 3) arrays (cheaper than ``np.cross`` for small arrays)."""
    a, b = np.broadcast_arrays(a, b)
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


@dataclass
class StepState:
    """Converged state of one step (arrays per wheel in WHEEL_NAMES order)."""
    t: float
    pose: Pose
    v: float
    omega: float
    s_cmd: float
    wheel_centre: np.ndarray       # (6, 3)
    contact_point: np.ndarray      # (6, 3)
    normal: np.ndarray             # (6, 3)
    tangent: np.ndarray            # (6, 3)
    sinkage: np.ndarray            # (6,)
    slip: np.ndarray               # (6,)
    Fx: np.ndarray                 # drawbar pull per wheel [N]
    Fz: np.ndarray                 # normal load per wheel [N]
    My: np.ndarray                 # drive torque per wheel [N m]
    soil_idx: np.ndarray
    pull: float                    # external pull the rover delivers (slip mode), 0 in speed mode [N]
    residual: float                # max |scaled equilibrium residual|
    converged: bool
    balance: dict = field(default_factory=dict)

    @property
    def power(self) -> np.ndarray:
        return self.My * self.omega


class RoverSimulator:
    """Quasi-static rocker-bogie rover with Bekker-Wong wheels on a :class:`LunarTerrain`.

    >>> sim = RoverSimulator(RoverParams(), LunarTerrain())
    >>> traj = sim.run(distance=2.0, v=0.1, dt=0.5)          # speed mode on flat GRC-1
    >>> traj.energy_per_metre()
    """

    def __init__(self, rover: RoverParams, terrain: LunarTerrain, contact=None, g: float = LUNAR_G,
                 s_limit: float = 0.95, conv_tol: float = 1e-6):
        rover.check()
        self.p = rover
        self.terrain = terrain
        self.contact = contact or BekkerWongContact(rover.wheel_radius, rover.wheel_width)
        if hasattr(self.contact, "check_wheel"):
            self.contact.check_wheel(rover.wheel_radius, rover.wheel_width)
        self.g = float(g)
        self.s_limit = float(s_limit)
        self.conv_tol = conv_tol          # max |scaled residual| accepted as equilibrium
        self.n_contact_samples = 31

    # ------------------------------------------------------------------ geometry
    def _contacts(self, centres: np.ndarray, R_body: np.ndarray):
        """Sinkage, contact point, wheel-plane normal and tangent for each wheel centre."""
        T = self.terrain
        Rw = self.p.wheel_radius
        bx, by = R_body[:, 0], R_body[:, 1]
        fwd = np.array([bx[0], bx[1], 0.0])
        fwd /= max(np.linalg.norm(fwd), 1e-12)
        u = np.linspace(-1.1 * Rw, 1.1 * Rw, self.n_contact_samples)
        q = centres[:, None, :] + u[None, :, None] * fwd[None, None, :]
        P = np.concatenate([q[..., :2], T.height(q[..., 0], q[..., 1])[..., None]], -1)
        d = np.linalg.norm(centres[:, None, :] - P, axis=-1)
        k = np.argmin(d, axis=1)
        p = P[np.arange(len(centres)), k]
        for _ in range(4):                                   # foot point on the local tangent plane
            n = T.normal(p[:, 0], p[:, 1])
            dist = np.einsum("ij,ij->i", centres - p, n)
            foot = centres - dist[:, None] * n
            p = np.concatenate([foot[:, :2], T.height(foot[:, 0], foot[:, 1])[:, None]], 1)
        n = T.normal(p[:, 0], p[:, 1])
        dist = np.einsum("ij,ij->i", centres - p, n)
        z = Rw - dist
        nw = n - np.einsum("ij,j->i", n, by)[:, None] * by[None]     # normal projected on the wheel plane
        nw /= np.linalg.norm(nw, axis=1, keepdims=True)
        t = _cross(by[None], nw)
        return z, p, nw, t

    def _slips(self, centres, prev_centres, tangents, omega, dt):
        Vw = omega * self.p.wheel_radius
        if prev_centres is None or dt <= 0:
            return np.zeros(len(centres))
        vt = np.einsum("ij,ij->i", (centres - prev_centres) / dt, tangents)
        s = np.zeros(len(centres))
        for i, v in enumerate(vt):
            if abs(Vw) < 1e-12 and abs(v) < 1e-12:
                s[i] = 0.0
            elif Vw >= v:
                s[i] = 1.0 - v / Vw if Vw > 1e-12 else -0.99
            else:
                s[i] = Vw / v - 1.0
        return np.clip(s, -0.99, 0.99)

    # ------------------------------------------------------------------ equilibrium
    def _evaluate(self, u, ctx):
        """Residuals for unknowns u = (z, pitch, roll, rocker, bogie_L, bogie_R[, s_cmd])."""
        p, g = self.p, self.g
        pose = Pose(ctx["x"], ctx["y"], u[0], ctx["yaw"], u[1], u[2], u[3], u[4], u[5])
        s_cmd = u[6] if ctx["mode"] == "speed" else ctx["s_cmd"]
        v = ctx["v"]
        omega = v / (p.wheel_radius * (1.0 - s_cmd)) if v != 0 else 0.0
        fr = link_frames(p, pose)
        R = fr["R"]
        bx, by, bz = R[:, 0], R[:, 1], R[:, 2]
        C = fr["wheel"]
        z, P, n, t = self._contacts(C, R)
        slip = self._slips(C, ctx["prev_centres"], t, omega, ctx["dt"])
        sidx = self.terrain.soil_index(P[:, 0], P[:, 1])
        F3 = self.contact(slip, z, self.terrain.soils(), sidx)
        Fx, Fz, My = F3[:, 0], F3[:, 1], F3[:, 2]
        F = Fx[:, None] * t + Fz[:, None] * n
        m = p.mass
        W = np.array([0.0, 0.0, -m * g])
        G = fr["com"]
        tow = -ctx.get("tow", 0.0) * bx
        total = F.sum(0) + W + tow
        lat = total @ by                                      # closed by a lateral reaction ~ wheel loads
        wl = np.clip(Fz, 0, None)
        share = wl / wl.sum() if wl.sum() > 0 else np.full(6, 1 / 6)
        Flat = -lat * share[:, None] * by[None]
        couple = -My[:, None] * by[None]                      # soil torque on each wheel about its axle
        M_G = (_cross(C - G, F) + _cross(P - G, Flat)).sum(0) + couple.sum(0)
        L = p.wheelbase
        res = [total @ bz / (m * g), M_G @ bx / (m * g * L), M_G @ by / (m * g * L)]
        side = np.repeat([0, 1], 3)
        mom_b = _cross(C - fr["bogie_pivot"][side], F) @ by - My        # about each wheel's bogie pivot
        mom_r = _cross(C - fr["rocker_pivot"][side], F) @ by - My       # about each wheel's rocker pivot
        tau = [float(mom_r[0:3].sum()), float(mom_r[3:6].sum())]
        res += [mom_b[1:3].sum() / (m * g * L), mom_b[4:6].sum() / (m * g * L)]
        res.append((tau[0] - tau[1]) / (m * g * L))
        a = (v - ctx["v_prev"]) / ctx["dt"] if ctx["dt"] > 0 else 0.0
        long_res = (total @ bx - m * a) / (m * g)
        if ctx["mode"] == "speed":
            res.append(long_res)
        out = {"pose": pose, "omega": omega, "s_cmd": s_cmd, "C": C, "P": P, "n": n, "t": t, "z": z,
               "slip": slip, "Fx": Fx, "Fz": Fz, "My": My, "sidx": sidx, "F": F, "Flat": Flat, "G": G,
               "pull": (total @ bx - m * a) if ctx["mode"] == "slip" else 0.0, "tau": tau, "M_G": M_G,
               "total": total + Flat.sum(0)}
        return np.array(res), out

    # largest change per step of (z, pitch, roll, rocker, bogie_L, bogie_R) accepted by the continuation; keeps
    # the solution on the branch followed so far (a rover can also balance on four wheels with two lifted)
    STEP_BOUND = np.array([0.05, 0.12, 0.12, 0.15, 0.15, 0.15])

    def solve_step(self, ctx, guess, bounded: bool = True):
        from scipy.optimize import least_squares
        lo = np.full(len(guess), -np.inf)
        hi = np.full(len(guess), np.inf)
        if bounded:
            lo[:6] = guess[:6] - self.STEP_BOUND
            hi[:6] = guess[:6] + self.STEP_BOUND
        if ctx["mode"] == "speed":
            lo[6], hi[6] = -0.99, self.s_limit
            guess = guess.copy()
            guess[6] = np.clip(guess[6], lo[6] + 1e-9, hi[6] - 1e-9)
        sol = least_squares(lambda u: self._evaluate(u, ctx)[0], guess, bounds=(lo, hi), method="trf",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, x_scale=np.r_[0.01, 0.05, 0.05, 0.05, 0.05, 0.05,
                                                                                0.05][:len(guess)],
                            max_nfev=400)
        r, out = self._evaluate(sol.x, ctx)
        return sol.x, r, out

    def conform(self, x, y, yaw, guess6, target_sinkage):
        """Kinematic predictor: the configuration that puts every wheel at ``target_sinkage`` at (x, y) (no forces;
        the distance functions keep a non-zero Jacobian even where a wheel would lose contact)."""
        from scipy.optimize import least_squares

        def res(u):
            pose = Pose(x, y, u[0], yaw, u[1], u[2], u[3], u[4], u[5])
            fr = link_frames(self.p, pose)
            z = self._contacts(fr["wheel"], fr["R"])[0]
            return (z - target_sinkage) / self.p.wheel_radius
        sol = least_squares(res, guess6, method="lm", xtol=1e-10, ftol=1e-10, max_nfev=200)
        return sol.x

    def initial_guess(self, x, y, yaw):
        """Level pose resting on the terrain at (x, y) (no sinkage), suspension straight."""
        p = self.p
        pitch = math.radians(float(self.terrain.slope_along(x, y, math.degrees(yaw))))
        # a few millimetres of sinkage so that every wheel starts in contact (zero sinkage carries no load)
        z0 = float(self.terrain.height(x, y)) + (p.wheel_radius - 0.005) / math.cos(pitch) + p.pivot_height
        return np.array([z0, pitch, 0.0, 0.0, 0.0, 0.0, 0.2])

    # ------------------------------------------------------------------ runs
    def run(self, distance: float, v: float, dt: float, *, mode: str = "speed", s_cmd: float = 0.0,
            start=(0.0, 0.0), yaw_deg: float = 0.0, tow: float = 0.0, settle: int = 3, stop_on_failure=True,
            progress=None) -> Trajectory:
        """Drive ``distance`` metres along a straight line from ``start`` heading ``yaw_deg`` at speed ``v``."""
        if mode not in ("speed", "slip"):
            raise ValueError("mode must be 'speed' or 'slip'")
        yaw = math.radians(yaw_deg)
        x, y = float(start[0]), float(start[1])
        u = self.initial_guess(x, y, yaw)
        u[:6] = self.conform(x, y, yaw, u[:6], np.full(6, 0.005))   # every wheel 5 mm deep to start
        if mode == "slip":
            u = u[:6]
        ctx = {"x": x, "y": y, "yaw": yaw, "v": 0.0, "v_prev": 0.0, "dt": 0.0, "mode": "slip", "s_cmd": 0.0,
               "prev_centres": None, "tow": 0.0}
        for _ in range(settle):                                 # static settling at rest
            u6, r, out = self.solve_step(ctx, u[:6])
            u = np.r_[u6, u[6:]]
        states = [self._state(0.0, out, r, ctx)]
        n_steps = int(math.ceil(distance / (v * dt) - 1e-9)) if v > 0 else 0
        prev = out["C"]
        failed_at = None
        for k in range(1, n_steps + 1):
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
            ctx = {"x": x, "y": y, "yaw": yaw, "v": v, "v_prev": v, "dt": dt, "mode": mode, "s_cmd": s_cmd,
                   "prev_centres": prev, "tow": tow}
            guess = (u if mode == "speed" else u[:6]).copy()
            # predictor: same sinkages on the terrain at the new position; corrector: the force equilibrium
            guess[:6] = self.conform(x, y, yaw, guess[:6], np.clip(states[-1].sinkage, 1e-4, None))
            sol, r, out = self.solve_step(ctx, guess)
            st = self._state(k * dt, out, r, ctx)
            states.append(st)
            u = sol if mode == "speed" else np.r_[sol, u[6:]]
            prev = out["C"]
            if progress:
                progress(k, n_steps, st)
            if not st.converged and failed_at is None:
                failed_at = k
                if stop_on_failure:
                    break
        return Trajectory(states, self.p, mode=mode, failed_at=failed_at, terrain=self.terrain)

    def _state(self, t, out, r, ctx) -> StepState:
        res = float(np.max(np.abs(r)))
        failed = ctx["mode"] == "speed" and out["s_cmd"] >= self.s_limit - 1e-6
        a = (ctx["v"] - ctx["v_prev"]) / ctx["dt"] if ctx["dt"] > 0 else 0.0
        return StepState(t, out["pose"], ctx["v"], out["omega"], out["s_cmd"], out["C"], out["P"], out["n"],
                         out["t"], out["z"], out["slip"], out["Fx"], out["Fz"], out["My"], out["sidx"],
                         float(out["pull"]), res, res < self.conv_tol and not failed,
                         {"lateral": out["Flat"], "com": out["G"], "tow": ctx.get("tow", 0.0), "accel": a,
                          "mass": self.p.mass, "g": self.g})

    # ------------------------------------------------------------------ steady states
    def steady_on_slope(self, slope_deg: float, v: float = 0.1, dt: float = 0.5, mode: str = "speed",
                        s_cmd: float = 0.0, n_steps: int = 4) -> StepState:
        """Steady state on a uniform grade (terrain is replaced by a plane with the same soil)."""
        saved = self.terrain
        self.terrain = LunarTerrain(slope_deg=slope_deg, soil=saved.soil)
        try:
            tr = self.run(distance=v * dt * n_steps, v=v, dt=dt, mode=mode, s_cmd=s_cmd, stop_on_failure=False)
        finally:
            self.terrain = saved
        return tr.states[-1]

    def max_slope(self, lo: float = 0.0, hi: float = 45.0, tol: float = 0.05, v: float = 0.1, dt: float = 0.5):
        """Largest uniform grade [deg] with a speed-mode equilibrium at ``s_cmd <= s_limit`` (bisection)."""
        ok = lambda a: self.steady_on_slope(a, v=v, dt=dt).converged  # noqa: E731
        if not ok(lo):
            return lo
        if ok(hi):
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return 0.5 * (lo + hi)


@dataclass
class Trajectory:
    states: list
    rover: RoverParams
    mode: str
    failed_at: int | None = None
    terrain: LunarTerrain | None = None

    def __len__(self):
        return len(self.states)

    def array(self, name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in self.states])

    @property
    def t(self):
        return self.array("t")

    def pose_array(self, name: str) -> np.ndarray:
        return np.array([getattr(s.pose, name) for s in self.states])

    def com(self):
        return np.array([link_frames(self.rover, s.pose)["com"] for s in self.states])

    def distance(self) -> np.ndarray:
        xy = np.array([[s.pose.x, s.pose.y] for s in self.states])
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

    def energy(self) -> np.ndarray:
        """Cumulative mechanical drive energy (sum over wheels of M_y omega, rectangle rule) [J]."""
        P = np.array([s.power.sum() for s in self.states])
        dt = np.diff(self.t)
        return np.r_[0.0, np.cumsum(P[1:] * dt)]

    def energy_per_metre(self) -> float:
        d = self.distance()[-1]
        return float(self.energy()[-1] / d) if d > 0 else float("nan")

    def wheel_table(self) -> dict:
        """Per-wheel time series (T, 6): sinkage, slip, drawbar pull, normal load, torque, power."""
        return {"sinkage": self.array("sinkage"), "slip": self.array("slip"), "drawbar_pull": self.array("Fx"),
                "normal_load": self.array("Fz"), "torque": self.array("My"),
                "power": np.array([s.power for s in self.states])}

    def summary(self) -> dict:
        ok = [s for s in self.states if s.converged]
        return {"steps": len(self.states), "failed_at_step": self.failed_at,
                "distance_m": float(self.distance()[-1]), "energy_J": float(self.energy()[-1]),
                "energy_per_metre_J_m": self.energy_per_metre(),
                "max_pitch_deg": float(np.degrees(np.abs(self.pose_array("pitch"))).max()),
                "max_roll_deg": float(np.degrees(np.abs(self.pose_array("roll"))).max()),
                "max_sinkage_mm": float(1e3 * self.array("sinkage").max()),
                "max_slip": float(self.array("slip").max()),
                "max_residual": float(max((s.residual for s in ok), default=float("nan")))}

    def to_csv(self, path: str) -> str:
        cols = ["t", "x", "y", "z", "pitch_deg", "roll_deg", "rocker_deg", "bogie_L_deg", "bogie_R_deg", "omega",
                "s_cmd", "pull_N", "residual"]
        wt = self.wheel_table()
        for q in wt:
            cols += [f"{q}_{w}" for w in WHEEL_NAMES]
        rows = []
        for k, s in enumerate(self.states):
            row = [s.t, s.pose.x, s.pose.y, s.pose.z, math.degrees(s.pose.pitch), math.degrees(s.pose.roll),
                   math.degrees(s.pose.rocker), math.degrees(s.pose.bogie_l), math.degrees(s.pose.bogie_r),
                   s.omega, s.s_cmd, s.pull, s.residual]
            for q in wt:
                row += list(wt[q][k])
            rows.append(row)
        np.savetxt(path, np.array(rows), delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
        return path


__all__ = ["RoverSimulator", "Trajectory", "StepState", "LUNAR_G"]
