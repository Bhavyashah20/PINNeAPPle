"""Vehicle description shared by the CAD, the multibody model and the simulator: a chassis plus corner modules
(suspension arm -> steering -> driven wheel), each with geometry, joint limits and actuator limits.

All positions are in the vehicle *assembly frame* at the reference (nominal ride height) pose: x forward, y left,
z up, origin on the ground below the geometric centre of the wheel footprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..terramechanics.wheel import Wheel


@dataclass
class Actuator:
    """Torque limit [N m], speed limit [rad/s] and PD/PI gains of a joint servo."""
    torque_limit: float
    speed_limit: float = np.inf
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class CornerModule:
    """One corner: suspension pivot and axis on the chassis, steering axis at the end of the arm, wheel centre.

    ``susp_pivot``, ``steer_origin``, ``wheel_centre`` [m] in the assembly frame; ``susp_axis`` and ``steer_axis`` unit
    vectors; ``wheel_axis`` is the axle direction (+y for both sides: positive spin drives forward). Masses [kg] of the
    arm, the steering module and the wheel (with its drive) and their CAD bodies are set by the vehicle definition.
    """
    name: str
    susp_pivot: np.ndarray
    susp_axis: np.ndarray
    steer_origin: np.ndarray
    steer_axis: np.ndarray
    wheel_centre: np.ndarray
    wheel_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    susp_limits: tuple = (-0.70, 0.70)
    steer_limits: tuple = (-0.87, 0.87)
    susp: Actuator = field(default_factory=lambda: Actuator(200.0))
    steer: Actuator = field(default_factory=lambda: Actuator(100.0))
    drive: Actuator = field(default_factory=lambda: Actuator(100.0, 3.0))
    side: int = 1                  # +1 left, -1 right
    front: int = 1                 # +1 front, -1 rear

    def __post_init__(self):
        for k in ("susp_pivot", "susp_axis", "steer_origin", "steer_axis", "wheel_centre", "wheel_axis"):
            setattr(self, k, np.asarray(getattr(self, k), float))
        for k in ("susp_axis", "steer_axis", "wheel_axis"):
            v = getattr(self, k)
            setattr(self, k, v / np.linalg.norm(v))


@dataclass
class VehicleSpec:
    name: str
    wheel: Wheel
    corners: list[CornerModule]
    total_mass: float
    max_speed: float
    sources: dict = field(default_factory=dict)       # quantity -> citation / "estimate: ..."
    cad: object = None                                # VehicleCAD with bodies "chassis", "arm_<c>", "steer_<c>", "wheel_<c>"

    @property
    def corner_names(self) -> list[str]:
        return [c.name for c in self.corners]

    def track(self) -> float:
        ys = [c.wheel_centre[1] for c in self.corners]
        return max(ys) - min(ys)

    def wheelbase(self) -> float:
        xs = [c.wheel_centre[0] for c in self.corners]
        return max(xs) - min(xs)


__all__ = ["VehicleSpec", "CornerModule", "Actuator"]
