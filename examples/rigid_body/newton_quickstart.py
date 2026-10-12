"""Rigid bodies with NVIDIA Newton: check the solver, then run a small scene.

    pip install newton
    python examples/rigid_body/newton_quickstart.py
"""
import numpy as np

from pinneapple_simulation.external_solvers.newton_physics import RigidScene, simulate, validate

# 1. the solver setting against closed-form results
validate("xpbd")

# 2. a crate sliding to rest next to a pendulum
scene = RigidScene().ground(mu=0.4)
scene.box("crate", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(2.0, 0, 0), mu=0.4)
a = np.radians(20)
bob = scene.sphere("bob", 0.05, pos=(1.0 + np.sin(a), 0, 2.0 - np.cos(a)))
scene.revolute(-1, bob, anchor=(1.0, 0, 2.0), axis=(0, 1, 0))
traj = simulate(scene, duration=2.0, dt=1e-3)

print(f"crate stopped at x = {traj.pos[-1, traj.index('crate'), 0]:.3f} m "
      f"(Coulomb: {2.0 ** 2 / (2 * 0.4 * 9.81):.3f} m); wall time {traj.wall_time_s:.1f} s")
traj.save("newton_quickstart.npz")
