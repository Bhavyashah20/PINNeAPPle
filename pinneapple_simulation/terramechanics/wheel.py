"""Rigid wheel geometry for terramechanics: radius, width, grousers (straight or chevron), mass and spin inertia."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Grousers:
    """Grousers (lugs) on a rigid wheel.

    ``count`` lugs of ``height`` [m] and ``thickness`` [m] around the rim. ``shape`` is ``"straight"`` (lugs across the
    tread) or ``"chevron"`` (two half-lugs meeting at the tread centre line, swept by ``chevron_angle_deg`` from the
    axle direction). ``shear_radius_factor`` is lambda in the equivalent shear radius ``R_s = R + lambda h`` of
    Ding et al. (2011), see :mod:`.wong_reece`.
    """
    count: int = 0
    height: float = 0.0
    thickness: float = 0.004
    shape: str = "straight"
    chevron_angle_deg: float = 0.0
    shear_radius_factor: float = 1.0

    def __post_init__(self):
        if self.count < 0 or self.height < 0:
            raise ValueError("grouser count and height must be >= 0")
        if self.shape not in ("straight", "chevron"):
            raise ValueError("grouser shape must be 'straight' or 'chevron'")

    @property
    def active(self) -> bool:
        return self.count > 0 and self.height > 0

    def pitch_angle(self) -> float:
        """Angle between neighbouring grousers [rad]."""
        return 2.0 * math.pi / self.count if self.count else float("inf")


@dataclass
class Wheel:
    """A rigid wheel: ``radius`` to the rim surface (grousers excluded) and ``width`` [m]; ``mass`` [kg] and
    ``spin_inertia`` [kg m^2] about the axle (None = thin rim, ``m R^2``, an upper bound for a spoked wheel)."""
    radius: float
    width: float
    grousers: Grousers = field(default_factory=Grousers)
    mass: float = 0.0
    spin_inertia: float | None = None

    def __post_init__(self):
        if not (self.radius > 0 and self.width > 0):
            raise ValueError("wheel radius and width must be > 0")

    @property
    def outer_radius(self) -> float:
        return self.radius + (self.grousers.height if self.grousers.active else 0.0)

    @property
    def shear_radius(self) -> float:
        """Equivalent shear radius ``R + lambda h`` (Ding et al. 2011); the rim radius without grousers."""
        g = self.grousers
        return self.radius + (g.shear_radius_factor * g.height if g.active else 0.0)

    def inertia(self) -> float:
        return self.spin_inertia if self.spin_inertia is not None else self.mass * self.radius ** 2


__all__ = ["Wheel", "Grousers"]
