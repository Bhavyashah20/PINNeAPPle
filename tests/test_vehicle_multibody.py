"""Multibody tree (free fall, pendulum period, momentum and energy of a free tree), vehicle CAD mass properties,
the VIPER definition (cited figures) and a short drive of the vehicle simulator on deformable GRC-1."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from pinneapple_design.geometry.vehicle_cad import (  # noqa: E402
    VehicleCAD,
    beveled_box,
    box,
    cylinder,
    mesh_mass_properties,
    rover_wheel,
    signed_volume,
    tube_shell,
)
from pinneapple_simulation.multibody import Body, MultibodyTree  # noqa: E402


def _no_force(tree):
    return lambda kin, q, v: (np.zeros(tree.nv), np.zeros((tree.nv, tree.nv)))


def test_free_fall():
    t = MultibodyTree()
    t.add_body(Body("a", 2.0))
    q, v = t.initial_state()
    for _ in range(1000):
        q, v, _ = t.step(q, v, 1e-3, _no_force(t))
    assert q[2] == pytest.approx(-0.5 * 1.62, rel=2e-3)


def test_pendulum_period():
    t = MultibodyTree(gravity=(0, 0, -9.81))
    t.add_body(Body("base", 1e6, inertia=np.eye(3) * 1e6))
    t.add_revolute("hinge", 0, Body("bob", 1.0, com=(0, 0, -1.0), inertia=np.eye(3) * 1e-6), (0, 0, 0), (0, 1, 0))

    def hold(kin, q, v):
        Q = np.zeros(t.nv)
        Q[:3] = -t.total_mass() * t.gravity
        return Q, np.zeros((t.nv, t.nv))
    q, v = t.initial_state(angles=[0.1])
    th = []
    for _ in range(45000):
        q, v, _ = t.step(q, v, 1e-4, hold)
        th.append(q[7])
    zc = np.where(np.diff(np.sign(th)) < 0)[0]
    T = np.diff(zc).mean() * 1e-4
    assert T == pytest.approx(2 * math.pi * math.sqrt(1 / 9.81) * (1 + 0.1 ** 2 / 16), rel=2e-3)


def test_free_tree_conserves_momentum_and_energy():
    t = MultibodyTree(gravity=(0, 0, 0))
    t.add_body(Body("a", 3, inertia=np.diag([1, 2, 3])))
    t.add_revolute("j1", 0, Body("b", 2, com=(0.3, 0, 0), inertia=np.diag([.1, .2, .3])), (0.2, 0.1, 0), (0, 0, 1))
    t.add_revolute("j2", 1, Body("c", 1, com=(0.2, 0, 0.1), inertia=np.diag([.05, .1, .1])), (0.6, 0, 0), (1, 1, 0))
    q, v = t.initial_state()
    v[:] = [0.1, 0.2, -0.1, 0.5, -0.3, 0.8, 2.0, -1.5]

    def mom(q, v):
        k = t.kinematics(q, v)
        P = sum(b.mass * k.vel[i] for i, b in enumerate(t.bodies))
        H = sum(np.cross(k.c[i], b.mass * k.vel[i]) + k.R[i] @ b.inertia @ k.R[i].T @ k.omega[i]
                for i, b in enumerate(t.bodies))
        return P, H, t.energy(k, v)[0]
    P0, H0, E0 = mom(q, v)
    for _ in range(4000):
        q, v, _ = t.step(q, v, 1e-4, _no_force(t))
    P1, H1, E1 = mom(q, v)
    assert np.abs(P1 - P0).max() < 1e-3 and np.abs(H1 - H0).max() < 1e-3
    assert E1 == pytest.approx(E0, rel=1e-3)


def test_mesh_mass_properties_match_closed_forms():
    c, inertia, vol = mesh_mass_properties(*box((2, 1, 0.5), (1, 2, 3)), 10.0)
    np.testing.assert_allclose(c, [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(np.diag(inertia), 10 / 12 * np.array([1.25, 4.25, 5.0]), rtol=1e-12)
    c, inertia, vol = mesh_mass_properties(*cylinder((0, 0, 0), (0, 0, 2), 0.5, 256), 3.0)
    assert inertia[2, 2] == pytest.approx(3 * 0.25 / 2, rel=1e-3)
    assert inertia[0, 0] == pytest.approx(3 * (3 * 0.25 + 4) / 12, rel=1e-3)
    c, inertia, vol = mesh_mass_properties(*tube_shell((0, -0.1, 0), (0, 0.1, 0), 0.2, 0.25, 256), 2.0)
    assert inertia[1, 1] == pytest.approx(2 * (0.2 ** 2 + 0.25 ** 2) / 2, rel=1e-3)
    assert signed_volume(*beveled_box((1, 1, 1), (0, 0, 0), 0.1)) == pytest.approx(0.97, abs=0.02)
    for _, (V, F), _, share in rover_wheel(0.25, 0.2, grousers=12, grouser_height=0.02, grouser_shape="chevron"):
        assert signed_volume(V, F) > 0 and share >= 0
    cad = VehicleCAD("t")
    cad.add("a", "body", box((1, 1, 1)), "white_paint", 4.0)
    cad.add("b", "body", box((1, 1, 1), (2, 0, 0)), "white_paint", 4.0)
    assert cad.mass_properties()["body"]["com"] == pytest.approx([1, 0, 0])


@pytest.fixture(scope="module")
def viper_spec():
    from pinneapple_simulation.vehicle import viper
    return viper()


def test_viper_follows_the_published_figures(viper_spec):
    s = viper_spec
    assert s.cad.total()["mass"] == pytest.approx(450.0)
    assert s.track() == pytest.approx(1.3) and s.wheelbase() == pytest.approx(1.0)
    assert s.wheel.radius == 0.25 and s.wheel.width == 0.20 and s.wheel.grousers.count == 24
    lo, hi = s.cad.bounds()
    assert hi[2] - lo[2] == pytest.approx(2.5, abs=0.01)
    assert hi[0] - lo[0] <= 1.71 and hi[1] - lo[1] <= 1.71
    for c in s.corners:
        assert c.susp.torque_limit == 200.0 and c.susp_limits[1] == pytest.approx(math.radians(40))
        assert c.steer_limits[1] == pytest.approx(math.radians(50))
    assert all(isinstance(v, str) and v for v in s.sources.values())
    assert any(v.startswith("estimate") for v in s.sources.values())


def test_vehicle_drives_rolls_and_leaves_ruts(viper_spec, tmp_path):
    from pinneapple_simulation.terramechanics import get_soil
    from pinneapple_simulation.vehicle import VehicleSimulator, build_vehicle, scenarios
    model = build_vehicle(viper_spec)
    assert model.tree.nv == 18 and model.tree.total_mass() == pytest.approx(450.0)
    ter = scenarios.flat_site(x_range=(-1.5, 2.5), y_range=(-1.5, 1.5), relief=0.0)
    run = VehicleSimulator(model, ter, get_soil("GRC-1")).run(3.0, speed=0.15, settle=1.0, ramp=0.5,
                                                              output_dt=0.1)
    s = run.summary()
    W = 450.0 * 1.62
    assert run.wheel["load"][-1].sum() == pytest.approx(W, rel=0.05)            # the wheels carry the weight
    assert 8 < s["mean_sinkage_mm"] < 25
    assert 0.0 < run.wheel["slip"][-1].mean() < 0.15
    assert run.chassis["speed"][-1] == pytest.approx(0.15, rel=0.15)
    assert np.all(np.diff(run.wheel["spin"][:, 0]) >= -1e-6)                    # the wheels roll forward
    assert ter.z_max.max() > 0.008 and ter.imprint.max() > 0                     # ruts and grouser marks
    assert len(run.terrain_events) > 10
    replay = ter.copy_empty()
    run.replay_terrain(replay, 1e9)
    np.testing.assert_allclose(replay.z_max, ter.z_max)
    run.to_csv(str(tmp_path / "run.csv"))
