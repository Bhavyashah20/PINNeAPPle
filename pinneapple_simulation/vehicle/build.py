"""Multibody model of a :class:`~.spec.VehicleSpec`: chassis (floating base) -> per corner a suspension arm (revolute),
a steering module (revolute) and a wheel (revolute spin), with masses, centres of mass and inertias from the CAD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..multibody import Body, MultibodyTree
from .spec import VehicleSpec


@dataclass
class VehicleModel:
    spec: VehicleSpec
    tree: MultibodyTree
    susp: list            # joint indices per corner
    steer: list
    wheel: list
    wheel_body: list      # body indices of the wheels
    origins: dict         # body name -> origin of its frame in the assembly frame (for posing the CAD)

    def transforms(self, kin) -> dict:
        """(R, t) per CAD body mapping assembly coordinates to the world for a kinematic state."""
        out = {}
        for i, b in enumerate(self.tree.bodies):
            R = kin.R[i]
            out[b.name] = (R, kin.p[i] - R @ self.origins[b.name])
        return out


def build_vehicle(spec: VehicleSpec, gravity: float = 1.62) -> VehicleModel:
    """Bodies named like the CAD bodies: ``chassis``, ``arm_<c>``, ``steer_<c>``, ``wheel_<c>``."""
    mp = spec.cad.mass_properties()
    tree = MultibodyTree(gravity=(0.0, 0.0, -gravity))
    ch = mp["chassis"]
    tree.add_body(Body("chassis", ch["mass"], ch["com"], ch["inertia"]))
    origins = {"chassis": np.zeros(3)}
    susp, steer, wheel, wb = [], [], [], []
    for c in spec.corners:
        a, s, w = mp[f"arm_{c.name}"], mp[f"steer_{c.name}"], mp[f"wheel_{c.name}"]
        P, K, C = c.susp_pivot, c.steer_origin, c.wheel_centre
        susp.append(tree.add_revolute(f"susp_{c.name}", 0, Body(f"arm_{c.name}", a["mass"], a["com"] - P,
                                                                a["inertia"]), P, c.susp_axis, limits=c.susp_limits))
        steer.append(tree.add_revolute(f"steer_{c.name}", f"arm_{c.name}",
                                       Body(f"steer_{c.name}", s["mass"], s["com"] - K, s["inertia"]), K - P,
                                       c.steer_axis, limits=c.steer_limits))
        Iw = w["inertia"].copy()
        if spec.wheel.spin_inertia is not None:              # an explicit spin inertia overrides the CAD value
            ax = c.wheel_axis
            Iw = Iw + (spec.wheel.spin_inertia - ax @ Iw @ ax) * np.outer(ax, ax)
        wheel.append(tree.add_revolute(f"wheel_{c.name}", f"steer_{c.name}",
                                       Body(f"wheel_{c.name}", w["mass"], w["com"] - C, Iw), C - K, c.wheel_axis))
        wb.append(tree.index(f"wheel_{c.name}"))
        origins[f"arm_{c.name}"], origins[f"steer_{c.name}"], origins[f"wheel_{c.name}"] = P, K, C
    return VehicleModel(spec, tree, susp, steer, wheel, wb, origins)


__all__ = ["VehicleModel", "build_vehicle"]
