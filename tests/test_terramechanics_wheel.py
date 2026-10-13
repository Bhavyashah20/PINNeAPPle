"""pinneapple_simulation.terramechanics: soils with provenance, the Wong-Reece wheel (against the standalone
Bekker-Wong solver and its own limits), grousers, slip-sinkage, multi-pass, bulldozing, lateral force, the testbed
with the bundled published points, the deformable terrain and the PINN surrogate."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
torch = pytest.importorskip("torch")

from pinneapple_simulation.terramechanics import (  # noqa: E402
    SOILS,
    Grousers,
    Wheel,
    WongReeceModel,
    bearing_capacity_factors,
    bulldozing_resistance,
    get_soil,
    slip_angle,
    slip_ratio,
    testbed,
)
from pinneapple_simulation.terramechanics.terrain import (  # noqa: E402
    DeformableTerrain,
    RigidRock,
    fractal_relief,
)


class _Flat:
    def height(self, x, y):
        return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)


def test_every_soil_parameter_has_a_source():
    for s in SOILS.values():
        for name, value, src in s.provenance():
            assert src != "not stated", (s.name, name)
            assert math.isfinite(value)
    assert get_soil("GRC-1").sources["k_phi"].startswith("secondary")
    assert get_soil("GRC-1").sources["K"].startswith("estimate")
    with pytest.raises(KeyError):
        get_soil("no such soil")


def test_matches_the_standalone_bekker_wong_solver():
    from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver
    from pinneapple_simulation.particle_dynamics.terramechanics import WheelParams
    s = get_soil("Apollo-lunar")
    m = WongReeceModel(s, Wheel(0.125, 0.06), bulldozing="none", n_gl=48)
    ref = BekkerWongSolver(s.to_soil_params(), WheelParams(R=0.125, b=0.06))
    for slip, z in [(0.1, 0.01), (0.5, 0.03), (-0.2, 0.02), (0.0, 0.005)]:
        np.testing.assert_allclose(m.forces(slip, z).as_array()[0], ref.forces(slip, z), rtol=2e-6, atol=1e-6)


def test_physical_trends():
    m = WongReeceModel(get_soil("GRC-1"), Wheel(0.25, 0.2))
    z = np.full(5, 0.02)
    r = m.forces(np.linspace(0.0, 0.6, 5), z, sinkage_slip=0.0)    # at a fixed sinkage, without slip-sinkage
    assert np.all(np.diff(r.Fx) > 0) and np.all(np.diff(r.My) > 0)            # pull and torque grow with slip
    st = m.steady_state(np.full(4, 180.0), np.array([0.0, 0.1, 0.2, 0.3]))   # at the load: slip-sinkage included
    assert np.all(np.diff(st["drawbar_pull"]) > 0) and np.all(np.diff(st["sinkage"]) > 0)
    assert np.all(np.diff(m.forces(0.2, np.linspace(0.005, 0.05, 6)).Fz) > 0)  # load grows with sinkage
    assert m.forces(0.2, 0.0).Fz == 0.0                                          # no contact, no force
    st = m.steady_state(np.full(3, 180.0), np.array([0.1, 0.3, 0.5]))
    np.testing.assert_allclose(st["load"], 180.0, rtol=1e-6)


def test_grousers_slip_sinkage_and_multipass():
    soil = get_soil("GRC-1")
    soil = soil.with_(n1=0.0)                           # without slip-sinkage, to isolate each effect
    smooth = WongReeceModel(soil, Wheel(0.25, 0.2))
    lugs = WongReeceModel(soil, Wheel(0.25, 0.2, grousers=Grousers(24, 0.025)))
    assert lugs.forces(0.2, 0.015).Fx > smooth.forces(0.2, 0.015).Fx                # R_s = R + lambda h
    ss = WongReeceModel(soil.with_(n1=0.3), Wheel(0.25, 0.2))
    z0 = smooth.sinkage_for_load(np.array([180.0, 180.0]), np.array([0.0, 0.6]))
    z1 = ss.sinkage_for_load(np.array([180.0, 180.0]), np.array([0.0, 0.6]))
    assert z1[1] / z1[0] > z0[1] / z0[0]                                              # slip-sinkage
    assert ss.soil.sources["n1"] == "user"
    first = smooth.steady_state(180.0, 0.2)
    second = smooth.steady_state(180.0, 0.2, precompaction=first["sinkage"])
    assert second["sinkage"] > first["sinkage"]           # total rut depth grows with passes
    third = smooth.steady_state(180.0, 0.2, precompaction=first["sinkage"], prior_shear=first["j_exit"])
    assert third["drawbar_pull"] > second["drawbar_pull"]  # soil sheared by the wheel in front


def test_bulldozing_and_lateral_force():
    nc, ng = bearing_capacity_factors(30.0)
    assert nc == pytest.approx(37.16, rel=0.01)                                   # Terzaghi general shear
    assert ng > 0
    soil = get_soil("GRC-1")
    assert bulldozing_resistance(soil, 0.02, 0.2, 1.62) > bulldozing_resistance(soil, 0.01, 0.2, 1.62) > 0
    m = WongReeceModel(soil, Wheel(0.25, 0.2))
    r = m.forces(np.full(3, 0.1), np.full(3, 0.02), np.radians([-5.0, 0.0, 5.0]))
    assert r.Fy[0] > 0 and r.Fy[1] == 0 and r.Fy[2] < 0                          # opposes the side slip
    assert r.Fy[2] == pytest.approx(-r.Fy[0])
    front = WongReeceModel(soil, Wheel(0.25, 0.2), bulldozing="both").forces(0.1, 0.02)
    assert front.Fx < r.Fx[1]


def test_slip_kinematics():
    assert slip_ratio(0.08, 0.4, 0.25) == pytest.approx(0.2)                     # driving: 1 - v / (R w)
    assert slip_ratio(0.1, 0.2, 0.25) == pytest.approx(-0.5)                     # braking: R w / v - 1
    assert abs(slip_ratio(0.0, 0.0, 0.25)) < 1e-12
    assert slip_angle(0.1, 0.1) == pytest.approx(math.atan2(0.1, 0.1 + 1e-3))


def test_testbed_validation_reports_errors_against_the_bundled_points():
    sets = testbed.list_datasets()
    assert ("ding2011", "grousered") in sets and ("hurrell2025", "grousered") in sets
    d = testbed.load_dataset("ding2011", "grousered")
    assert d.wheel.grousers.count == 30 and d.load == 80.0
    rep = testbed.validate(d)
    e = rep.errors["drawbar_pull_N"]
    assert 0 < e["rmse_rel_scale"] < 0.1                                          # 0.05 W with the cited soil
    assert len(rep.model["slip"]) == len(d.slip)


def test_deformable_terrain_ruts_memory_and_marks():
    t = DeformableTerrain(_Flat(), (-1, 1), (-1, 1), cell=0.01, rebound=lambda z: 0.1 * np.asarray(z))
    t.record_wheel(0.0, 0.0, (1.0, 0.0), 0.25, 0.2, 0.02, sigma_max=3e4, j_exit=0.01)
    assert t.precompaction(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), 0.1)[0] == pytest.approx(0.02, abs=1e-3)
    assert t.rut_depth(0.0, 0.0) == pytest.approx(0.018, abs=1e-3)               # minus the elastic rebound
    assert t.rut_depth(0.0, 0.3) == 0.0
    assert t.prior_shear(np.array([[-0.05, 0.0]]), np.array([[1.0, 0.0]]), 0.05)[0] == pytest.approx(0.01)
    t.stamp_grouser(0.0, 0.0, (1.0, 0.0), 0.2, 0.01, 0.004)
    assert t.surface(0.0, 0.0) < -0.018
    V, F, fl = t.mesh((-0.2, 0.2), (-0.2, 0.2))
    assert len(V) and fl["normal_stress"].max() == pytest.approx(30.0)
    e = t.copy_empty()
    assert e.z_max.max() == 0 and t.z_max.max() > 0
    rock = t.add_rock(RigidRock(0.5, 0.5, 0.1, 0.1, 0.1, burial=0.5))
    d, n = rock.penetration(np.array([[0.5, 0.5, rock.zc + 0.05], [0.5, 0.5, 1.0]]))
    assert d[0] > 0 > d[1] and n[0][2] > 0.99
    r = fractal_relief((64, 64), 0.01, 0.005, seed=1)
    assert r.std() == pytest.approx(0.005)


def test_pinn_surrogate_trains_and_reports_held_out_errors(tmp_path):
    from pinneapple_simulation.terramechanics.surrogate import (
        compare_on_states,
        load_surrogate,
        train_contact_surrogate,
    )
    m = WongReeceModel(get_soil("GRC-1"), Wheel(0.25, 0.2, grousers=Grousers(24, 0.025)))
    sur, rep = train_contact_surrogate(m, (0.0, 0.6), (0.004, 0.04), epochs=250, n_lhs=400, n_test=300, n_grid=20)
    assert set(rep["test"]) == {"Fx", "Fz", "My"}
    assert rep["test"]["Fz"]["rmse_over_range"] < 0.05
    assert "fz_monotone" in rep["physics_terms_used"]
    sur.save(str(tmp_path / "s"))
    again = load_surrogate(str(tmp_path / "s"))
    np.testing.assert_allclose(again.predict([0.2], [0.02]), sur.predict([0.2], [0.02]), rtol=1e-5)
    c = compare_on_states(sur, m, [0.1, 0.9], [0.02, 0.02])
    assert c["fraction_outside"] == pytest.approx(0.5)
    from pinneapple_simulation.rover import SurrogateContact
    sc = SurrogateContact.from_torchscript(str(tmp_path / "s.pt"), str(tmp_path / "s.json"))
    assert sc.R == pytest.approx(0.25)
