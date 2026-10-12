"""NVIDIA Newton rigid-body backend for PINNeAPPle.

`Newton <https://github.com/newton-physics/newton>`_ (Apache-2.0, built on NVIDIA Warp) is a GPU-ready physics
engine with several rigid-body solvers: XPBD (position based, robust contacts), semi-implicit Euler with
penalty contacts, Featherstone (reduced coordinates for joint chains) and MuJoCo-Warp. It runs on CPU too.

Sub-modules
-----------
scene      — RigidScene: bodies (sphere, box, capsule, cylinder, mesh), revolute / fixed joints, ground
runner     — simulate() -> RigidTrajectory (positions, quaternions, velocities, energy, UPD export)
validation — analytic checks (free fall, friction, pendulum, restitution, torque-free spin, resting contact)

Quick start
-----------
>>> from pinneapple_simulation.external_solvers.newton_physics import RigidScene, simulate, validate
>>> scene = RigidScene().ground(mu=0.4)
>>> scene.box("crate", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(2.0, 0, 0), mu=0.4)
>>> traj = simulate(scene, duration=1.0, dt=1e-3)         # stops 0.51 m further, as v0^2 / (2 mu g)
>>> rows = validate("xpbd")                                 # check the solver settings first

Install: ``pip install newton`` (core) or ``pip install "newton[examples]"`` (viewer, importers, MuJoCo-Warp).
"""
from .runner import SOLVERS, RigidTrajectory, simulate
from .scene import Body, Joint, RigidScene, Shape
from .validation import CASES, validate

__all__ = ["RigidScene", "Body", "Shape", "Joint", "simulate", "RigidTrajectory", "SOLVERS", "validate", "CASES"]
