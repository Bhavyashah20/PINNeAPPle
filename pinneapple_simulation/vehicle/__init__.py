"""Wheeled planetary vehicles: a description (chassis + corner modules) with its CAD, the multibody model built from
it, and a simulator on deformable terrain with Wong-Reece wheels.

    import pinneapple as pp
    spec = pp.vehicle.viper()                                  # NASA VIPER from public figures (cited in VIPER_SOURCES)
    model = pp.vehicle.build_vehicle(spec)                     # multibody tree, masses and inertias from the CAD
    terrain = pp.terramechanics.terrain.DeformableTerrain(base, (-2, 10), (-3, 3), cell=0.01)
    run = pp.vehicle.VehicleSimulator(model, terrain, pp.terramechanics.get_soil("GRC-1")).run(30.0, speed=0.2)
    run.summary(); pp.vehicle.viz.animate(run, terrain, "run.gif", field="normal_stress")

Modules: :mod:`.spec`, :mod:`.viper`, :mod:`.build`, :mod:`.sim`, :mod:`.scenarios`, :mod:`.viz`. See
``docs/core_concepts/lunar_rover_terramechanics.md``.
"""
from .build import VehicleModel, build_vehicle
from .sim import VehicleRun, VehicleSimulator
from .spec import Actuator, CornerModule, VehicleSpec
from .viper import VIPER_SOURCES, viper, viper_cad

__all__ = ["VehicleSpec", "CornerModule", "Actuator", "VehicleModel", "build_vehicle", "VehicleSimulator",
           "VehicleRun", "viper", "viper_cad", "VIPER_SOURCES"]


def __getattr__(name):
    if name in ("viz", "scenarios"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
