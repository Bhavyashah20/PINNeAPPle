# Rigid bodies with NVIDIA Newton

`pinneapple_simulation.external_solvers.newton_physics` runs rigid-body scenes with
[Newton](https://github.com/newton-physics/newton) (Apache-2.0, built on NVIDIA Warp). Newton runs on CPU or GPU and
offers several rigid-body solvers; PINNeAPPle wraps them behind one scene description, records the trajectory in
SI units and checks a solver setting against closed-form results before it is used in a study.

```bash
pip install newton                 # core: XPBD, semi-implicit, Featherstone
pip install "newton[examples]"     # adds the viewer, mesh/URDF/USD importers and MuJoCo-Warp
```

## Scene, run, result

```python
from pinneapple_simulation.external_solvers.newton_physics import RigidScene, simulate

scene = RigidScene(gravity=9.81).ground(mu=0.4)
scene.box("crate", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(2.0, 0, 0), mu=0.4)   # half extents [m]
bob = scene.sphere("bob", 0.05, pos=(0.17, 0, 1.0))
scene.revolute(-1, bob, anchor=(0, 0, 2.0), axis=(0, 1, 0))                    # hinge to the world

traj = simulate(scene, duration=2.0, dt=1e-3, solver="xpbd", record_every=10)
traj.pos, traj.quat, traj.vel, traj.omega      # (T, N, 3|4), quaternions (w, x, y, z)
traj.energy()["total"]                         # kinetic + potential [J]
traj.to_upd()                                  # PhysicalSample for datasets and surrogates
traj.to_rigid_body_state()                     # torch state for the differentiable rigid_body module
```

- Shapes: `sphere`, `box`, `capsule`, `cylinder` and `mesh` (closed triangle mesh, e.g. from an STL).
  Mass comes from `density` (default 1000 kg/m³) or is set exactly with `mass=`.
- Joints: `revolute` (hinge through an anchor about an axis, both given in world coordinates at t = 0) and `fixed`.
  Joints form trees; a body that is the parent of a joint but has no parent itself gets a free joint to the world.
- `scene.check()` lists problems (non-positive sizes, a body with two parents, invalid friction) before any
  solver is touched; the description imports without Newton installed.
- Solvers: `"xpbd"` (default), `"semi_implicit"`, `"featherstone"` and `"mujoco"` (needs `newton[sim]`).
  Initial body velocities are set for the maximal-coordinate solvers (XPBD, semi-implicit).

## Check the solver before trusting it

`validate(solver)` runs six scenes with a closed-form answer and reports simulated value, reference, error and
pass/fail. Results on CPU, `dt = 1 ms` (bounce `0.1 ms`), Newton 1.6.1:

| check | reference | XPBD | Featherstone | semi-implicit |
|---|---|---|---|---|
| free fall, max error / drop height | z0 − g t²/2 | 0.20 % ✓ | 0.20 % ✓ | 0.20 % ✓ |
| box sliding with friction μ = 0.4 | stops at v0²/(2μg) = 0.510 m | 0.509 m, 0.20 % ✓ | – | 0.282 m, 45 % ✗ |
| pendulum period, 5° | 2π√(I/(mgL)) = 2.007 s | 2.010 s, 0.17 % ✓ | 2.008 s, 0.05 % ✓ | unstable ✗ |
| rebound with restitution e = 0.5 | e² h = 0.250 m | 0.250 m, 0.05 % ✓ | – | 0.068 m ✗ |
| torque-free spin, angular momentum drift | 0 | 0.30 % ✓ | – | 0.29 % ✓ |
| box resting on the ground for 2 s | no sinking, no drift | 2 µm ✓ | 8.2 mm ✗ | 8.2 mm ✗ |

What the table says:

- **XPBD** passes all six and is the default for contact-rich scenes. It is slightly dissipative: a 10° pendulum
  loses about 2.6 % of its swing energy in 2 s at `dt = 1 ms`.
- **Featherstone** is the most accurate for joint chains (pendulum period within 0.05 %), but with its default
  penalty contact a 0.2 m box sinks 8 mm into the ground; use it for articulated mechanisms, not stacking or impacts.
- **Semi-implicit** uses penalty contacts and needs a much smaller step and tuned contact stiffness (`ke`, `kd`)
  for friction, restitution and hinges; with the defaults at 1 ms it fails four checks, and `validate` says so.

Checks marked "–" need initial body velocities, which are set only for the maximal-coordinate solvers.

## Files

- `scene.py`: `RigidScene`, shapes, joints, the Newton model builder.
- `runner.py`: `simulate()`, `RigidTrajectory` (energy, UPD export, `.npz` save).
- `validation.py`: the six analytic checks and `validate()`.
- Tests: `tests/test_newton_rigid.py` (scene checks run without Newton; the solver tests are skipped when it is
  not installed).
