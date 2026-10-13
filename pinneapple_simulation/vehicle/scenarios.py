"""Ready-made lunar sites and studies for wheeled vehicles: level ground, uniform grades, a crater with rocks, and a
grade sweep run in parallel processes.

    from pinneapple_simulation.vehicle import scenarios, viper
    site = scenarios.crater_site()                       # DeformableTerrain with a crater, rocks and scenery
    rows = scenarios.grade_sweep(viper, "GRC-1", grades=(0, 5, 10, 15, 20), workers=4)

Scene inputs (crater size, rock sizes, speeds) are choices for demonstrations, not measured site data.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..rover.terrain import Crater, LunarTerrain, random_craters
from ..terramechanics.soils import get_soil
from ..terramechanics.terrain import DeformableTerrain, RigidRock


def flat_site(x_range=(-3.0, 12.0), y_range=(-3.0, 3.0), cell: float = 0.01, relief: float = 0.005, seed: int = 2,
              slope_deg: float = 0.0, slope_dir_deg: float = 0.0) -> DeformableTerrain:
    """Level (or uniformly graded) regolith with a fine random relief of RMS ``relief`` [m]."""
    return DeformableTerrain(LunarTerrain(slope_deg=slope_deg, slope_dir_deg=slope_dir_deg), x_range, y_range,
                             cell=cell, relief=relief, relief_seed=seed)


def crater_site(crater=(5.0, 0.5, 4.0), depth_ratio: float = 0.08, rim_ratio: float = 0.015,
                rock=(2.0, 0.65, 0.16, 0.13, 0.17), x_range=(-3.0, 12.0), y_range=(-4.0, 4.0), cell: float = 0.01,
                relief: float = 0.007, scenery_craters: int = 60, scenery_rocks: int = 5, seed: int = 5
                ) -> DeformableTerrain:
    """A degraded crater ``(x, y, D)`` on the path (depth ``depth_ratio`` D; small lunar craters are shallower than
    Pike's fresh-crater 0.2), one rock ``(x, y, a, b, c)`` under the left wheels, and scenery: small craters away
    from the path and a few rocks."""
    rng = np.random.default_rng(seed)
    cx, cy, D = crater
    small = random_craters(rng, scenery_craters, (x_range[0] - 25, x_range[1] + 25), (y_range[0] - 25, y_range[1] + 25),
                           (0.3, 6.0), keep_out=lambda x, y, d: abs(y) < 1.3 + d / 2 and x_range[0] < x < x_range[1])
    base = LunarTerrain(craters=[Crater(cx, cy, D, rim_ratio=rim_ratio, depth_ratio=depth_ratio)] + small)
    ter = DeformableTerrain(base, x_range, y_range, cell=cell, relief=relief, relief_seed=2)
    if rock is not None:
        x, y, a, b, c = rock
        ter.add_rock(RigidRock(x, y, a, b, c, burial=0.3, seed=3))
    for i in range(scenery_rocks):
        while True:
            x, y = rng.uniform(x_range[0] + 1, x_range[1] - 1), rng.uniform(y_range[0] + 0.5, y_range[1] - 0.5)
            if abs(y) > 1.6 and math.hypot(x - cx, y - cy) > D / 2 + 0.3:
                break
        a = rng.uniform(0.08, 0.25)
        ter.add_rock(RigidRock(x, y, a, 0.8 * a, 0.7 * a, burial=0.35, seed=10 + i))
    return ter


def _one_grade(args):
    factory, soil, grade, duration, speed, cell = args
    from .build import build_vehicle
    from .sim import VehicleSimulator
    spec = factory()
    ter = flat_site(x_range=(-2.0, 2.0 + speed * duration * 1.5), y_range=(-3.0, 3.0), cell=cell, relief=0.0,
                    slope_deg=grade)
    run = VehicleSimulator(build_vehicle(spec), ter, get_soil(soil) if isinstance(soil, str) else soil).run(
        duration, speed=speed, settle=2.0, ramp=1.0, record_stresses=False)
    late = run.t > 0.5 * run.t[-1]
    return {"grade_deg": grade, "slip_mean": float(run.wheel["slip"][late].mean()),
            "speed_m_s": float(run.chassis["speed"][late].mean()), "speed_cmd_m_s": speed,
            "sinkage_mm": float(1e3 * run.wheel["sinkage"][late].mean()),
            "drive_torque_Nm": float(np.abs(run.wheel["drive_torque"][late]).mean()),
            "max_drive_torque_Nm": float(np.abs(run.wheel["drive_torque"]).max()),
            "power_W": float(np.clip(run.wheel["drive_torque"] * run.wheel["omega_joint"], 0, None)[late].sum(1).mean()),
            "yaw_deg_end": float(math.degrees(run.chassis["yaw"][-1])),
            "travel_reduction": float(1 - run.chassis["speed"][late].mean() / speed)}


def grade_sweep(factory, soil="GRC-1", grades=(0, 5, 10, 15, 20), duration: float = 12.0, speed: float = 0.1,
                workers: int = 4, cell: float = 0.01) -> list[dict]:
    """Drive straight up uniform grades (``factory()`` returns a :class:`VehicleSpec`); per grade the mean slip,
    speed, sinkage, torque and power over the second half of the run. Runs in ``workers`` processes."""
    args = [(factory, soil, float(g), duration, speed, cell) for g in grades]
    if workers <= 1:
        return [_one_grade(a) for a in args]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_one_grade, args))


__all__ = ["flat_site", "crater_site", "grade_sweep"]
