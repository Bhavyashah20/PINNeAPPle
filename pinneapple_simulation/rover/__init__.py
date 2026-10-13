"""Six-wheel rocker-bogie rover on lunar terrain with Bekker-Wong wheel-soil forces.

    from pinneapple_simulation.rover import RoverParams, LunarTerrain, Crater, RoverSimulator
    sim = RoverSimulator(RoverParams(), LunarTerrain(slope_deg=10))
    traj = sim.run(distance=2.0, v=0.1, dt=0.5)           # speed mode: the drive finds the slip it needs
    traj.summary(), traj.wheel_table()                     # per wheel: sinkage, slip, DP, load, torque, power

Modules: :mod:`.geometry` (parameters, kinematics, meshes, glTF/STL), :mod:`.terrain` (slope, craters, rocks,
soil regions), :mod:`.contact` (Bekker-Wong and surrogate force laws), :mod:`.dynamics` (quasi-static
rocker-bogie equilibrium + longitudinal motion), :mod:`.checks` (verification against closed forms and the
standalone solver), :mod:`.viz` (plots, GIF/MP4 and the 3D twin viewer). See
``docs/core_concepts/lunar_rover_terramechanics.md``.
"""
from .contact import BekkerWongContact, RangeLog, SurrogateContact
from .dynamics import LUNAR_G, RoverSimulator, StepState, Trajectory
from .geometry import (
           WHEEL_NAMES,
           Pose,
           RoverParams,
           assembled_mesh,
           export_rover,
           link_frames,
           rover_meshes,
)
from .terrain import PIKE_1977_DEPTH_RATIO, Crater, LunarTerrain, Rock, SoilRegion, random_craters, random_rocks

__all__ = ["RoverParams", "Pose", "WHEEL_NAMES", "link_frames", "rover_meshes", "assembled_mesh", "export_rover",
           "LunarTerrain", "Crater", "Rock", "SoilRegion", "random_rocks", "random_craters", "PIKE_1977_DEPTH_RATIO",
           "BekkerWongContact", "SurrogateContact", "RangeLog", "RoverSimulator", "Trajectory", "StepState",
           "LUNAR_G"]
