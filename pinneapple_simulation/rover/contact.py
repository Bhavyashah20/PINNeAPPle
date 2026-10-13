"""Wheel-soil force laws for the rover: (slip, sinkage, soil) -> (F_x, F_z, M_y) per wheel.

* :class:`BekkerWongContact` -- the library's Bekker-Wong solver (``numerical_solvers.bekker_wong``): vectorised
  piecewise Gauss-Legendre quadrature (default, the reference) or adaptive ``quad`` (``method="quad"``, slow, for
  checks). Per-wheel soils are supported, so wheels on different regions use their own parameters.
* :class:`SurrogateContact` -- any fast surrogate of the same law trained for one soil and one wheel, e.g. the
  TorchScript PINN exported by ``examples/use_cases/terramechanics/terramechanics_rover_pinn.py``. Every call
  records which inputs fall outside the training box, so a run can report how much of it was extrapolated.

Forces are in the wheel's contact frame: F_x along the terrain in the travel direction (drawbar pull), F_z along
the terrain normal (load carried), M_y the soil torque about the axle that the motor must supply [N, N, N m].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..numerical_solvers.bekker_wong import BekkerWongSolver, bekker_wong_forces_torch
from ..particle_dynamics.terramechanics import SoilParams, WheelParams


class BekkerWongContact:
    """Bekker-Wong forces for N wheels at once (``soils[k]`` selected per wheel with ``soil_idx``)."""

    name = "bekker_wong"

    def __init__(self, wheel_radius: float, wheel_width: float, rear_angle_ratio: float = 1.0 / 3.0,
                 method: str = "gauss", n_gl: int = 48):
        if method not in ("gauss", "quad"):
            raise ValueError("method must be 'gauss' or 'quad'")
        self.R, self.b = float(wheel_radius), float(wheel_width)
        self.rear_angle_ratio = rear_angle_ratio
        self.method = method
        self.n_gl = n_gl
        self._solvers: dict[int, BekkerWongSolver] = {}

    def solver(self, soil: SoilParams) -> BekkerWongSolver:
        key = id(soil)
        if key not in self._solvers:
            self._solvers[key] = BekkerWongSolver(soil, WheelParams(R=self.R, b=self.b),
                                                  rear_angle_ratio=self.rear_angle_ratio)
        return self._solvers[key]

    def __call__(self, slip, z, soils: list[SoilParams], soil_idx) -> np.ndarray:
        import torch
        slip = np.asarray(slip, float).ravel()
        z = np.asarray(z, float).ravel()
        idx = np.broadcast_to(np.asarray(soil_idx, int), slip.shape)
        out = np.zeros((len(slip), 3))
        on = z > 0
        if not on.any():
            return out
        zc = np.minimum(z[on], 0.95 * self.R)
        if self.method == "quad":
            for j, (s, zz, k) in zip(np.flatnonzero(on), zip(slip[on], zc, idx[on], strict=True), strict=True):
                out[j] = self.solver(soils[k]).forces(float(s), float(zz))
            return out
        S = [soils[k] for k in idx[on]]
        col = lambda name: torch.tensor([getattr(s, name) for s in S], dtype=torch.float64)  # noqa: E731
        res = bekker_wong_forces_torch(
            torch.tensor(slip[on]), torch.tensor(zc), R=self.R, b=self.b, c=col("c"),
            tan_phi=torch.tensor([s.tan_phi for s in S], dtype=torch.float64), K=col("K"), k_c=col("k_c"),
            k_phi=col("k_phi"), n=col("n"), a0=col("a0"), a1=col("a1"),
            rear_angle_ratio=self.rear_angle_ratio, n_gl=self.n_gl)
        out[on] = res.numpy()
        return out


@dataclass
class RangeLog:
    """Counts of surrogate inputs inside / outside the training box."""
    n_calls: int = 0
    n_out: int = 0
    slip_out: int = 0
    sinkage_out: int = 0
    soil_mismatch: int = 0
    worst: dict = field(default_factory=dict)

    @property
    def fraction_out(self) -> float:
        return self.n_out / max(self.n_calls, 1)


class SurrogateContact:
    """Wrap ``predict(slip, z) -> (N, 3)`` trained on ``soil`` for a wheel (R, b) on slip x sinkage box ranges.

    Inputs outside the box are still evaluated (the network extrapolates) but are counted in :attr:`log` and
    flagged per call in :attr:`last_flags`; wheels standing on a soil other than the training soil are counted as
    ``soil_mismatch``.
    """

    name = "surrogate"

    def __init__(self, predict, slip_range, sinkage_range, soil: SoilParams, wheel_radius: float,
                 wheel_width: float):
        self.predict = predict
        self.slip_range = tuple(float(v) for v in slip_range)
        self.sinkage_range = tuple(float(v) for v in sinkage_range)
        self.soil = soil
        self.R, self.b = float(wheel_radius), float(wheel_width)
        self.log = RangeLog()
        self.last_flags = np.zeros(0, bool)

    @classmethod
    def from_torchscript(cls, model_path: str, metadata_path: str) -> SurrogateContact:
        """Load the TorchScript + JSON pair written by the terramechanics use case (inputs/outputs normalised to
        [-1, 1] with ``norm_x_lo/hi`` and ``norm_y_lo/hi``; the input box is the training box)."""
        import torch
        meta = json.load(open(metadata_path))
        model = torch.jit.load(model_path, map_location="cpu").eval()
        lo_x, hi_x = np.array(meta["norm_x_lo"]), np.array(meta["norm_x_hi"])
        lo_y, hi_y = np.array(meta["norm_y_lo"]), np.array(meta["norm_y_hi"])

        def predict(slip, z):
            x = np.stack([np.asarray(slip, float), np.asarray(z, float)], 1)
            xn = 2.0 * (x - lo_x) / np.maximum(hi_x - lo_x, 1e-12) - 1.0
            with torch.no_grad():
                yn = model(torch.tensor(xn, dtype=torch.float32)).double().numpy()
            return (yn + 1.0) * (hi_y - lo_y) / 2.0 + lo_y

        sp = meta.get("soil_params", {})
        soil = SoilParams(c=sp.get("c_Pa", 1400.0), phi_deg=sp.get("phi_deg", 30.0), K=sp.get("K_m", 0.018),
                          k_c=sp.get("k_c", 1370.0), k_phi=sp.get("k_phi", 814000.0), n=sp.get("n", 1.0))
        wp = meta.get("wheel_params", {})
        return cls(predict, (lo_x[0], hi_x[0]), (lo_x[1], hi_x[1]), soil, wp.get("R_m", 0.125), wp.get("b_m", 0.06))

    def check_wheel(self, wheel_radius: float, wheel_width: float, rtol: float = 1e-6) -> None:
        if abs(wheel_radius - self.R) > rtol * self.R or abs(wheel_width - self.b) > rtol * self.b:
            raise ValueError(f"surrogate trained for R={self.R}, b={self.b}; rover wheel is "
                             f"R={wheel_radius}, b={wheel_width}")

    def in_range(self, slip, z) -> np.ndarray:
        slip, z = np.asarray(slip, float), np.asarray(z, float)
        return ((slip >= self.slip_range[0]) & (slip <= self.slip_range[1])
                & (z >= self.sinkage_range[0]) & (z <= self.sinkage_range[1]))

    def __call__(self, slip, z, soils: list[SoilParams], soil_idx) -> np.ndarray:
        slip = np.asarray(slip, float).ravel()
        z = np.asarray(z, float).ravel()
        idx = np.broadcast_to(np.asarray(soil_idx, int), slip.shape)
        out = np.zeros((len(slip), 3))
        on = z > 0
        flags = np.zeros(len(slip), bool)
        if on.any():
            out[on] = self.predict(slip[on], z[on])
            s_ok = (slip >= self.slip_range[0]) & (slip <= self.slip_range[1])
            z_ok = (z >= self.sinkage_range[0]) & (z <= self.sinkage_range[1])
            soil_ok = np.array([_same_soil(soils[k], self.soil) for k in idx])
            flags = on & ~(s_ok & z_ok & soil_ok)
            self.log.n_calls += int(on.sum())
            self.log.n_out += int(flags.sum())
            self.log.slip_out += int((on & ~s_ok).sum())
            self.log.sinkage_out += int((on & ~z_ok).sum())
            self.log.soil_mismatch += int((on & ~soil_ok).sum())
        self.last_flags = flags
        return out


def _same_soil(a: SoilParams, b: SoilParams) -> bool:
    keys = ("c", "phi_deg", "K", "k_c", "k_phi", "n")
    return all(abs(getattr(a, k) - getattr(b, k)) <= 1e-9 * max(1.0, abs(getattr(b, k))) for k in keys)


__all__ = ["BekkerWongContact", "SurrogateContact", "RangeLog"]
