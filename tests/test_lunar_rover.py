"""Rocker-bogie rover with Bekker-Wong wheels (pinneapple_simulation.rover): verification against the standalone
solver, Bekker's closed form, the balance laws and the drawbar-pull grade estimate; terrain, geometry, surrogate
range flags and the twin3d motion export. The long scenario runs live in
examples/use_cases/terramechanics/lunar_rover_3d.py."""
import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("torch")

from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams  # noqa: E402
from pinneapple_simulation.rover import (  # noqa: E402
    PIKE_1977_DEPTH_RATIO,
    BekkerWongContact,
    Crater,
    LunarTerrain,
    Pose,
    RoverParams,
    RoverSimulator,
    SoilRegion,
    SurrogateContact,
    checks,  # noqa: E402
    export_rover,
    link_frames,
)


@pytest.fixture(scope="module")
def rover():
    return RoverParams()


@pytest.fixture(scope="module")
def flat_sim(rover):
    return RoverSimulator(rover, LunarTerrain())


def test_level_kinematics_equal_spacing(rover):
    fr = link_frames(rover, Pose(0.0, 0.0, 0.0))
    W = fr["wheel"]
    np.testing.assert_allclose(W[:3, 0], [0.35, 0.0, -0.35], atol=1e-12)
    np.testing.assert_allclose(W[:, 2], -rover.pivot_height, atol=1e-12)
    np.testing.assert_allclose(W[:3, 1], 0.3, atol=1e-12)
    np.testing.assert_allclose(W[3:, 1], -0.3, atol=1e-12)


def test_static_sinkage_matches_bekker_closed_form(rover):
    r = checks.static_sinkage_check(rover)
    assert r["max_rel_vs_integral"] < 1e-9           # same pressure law, exact integral (run: 1.5e-13)
    assert r["max_rel_vs_closed_form"] < 0.01        # Bekker's parabolic-arc approximation (run: 0.67 %)
    for row in r["rows"]:
        assert 5.0 < row["z_model_mm"] < 10.0


def test_single_wheel_matches_standalone_solver(flat_sim, rover):
    st = flat_sim.run(distance=0.15, v=0.1, dt=0.5, mode="slip", s_cmd=0.2).states[-1]
    assert st.converged
    c = checks.single_wheel_check(st, rover, flat_sim.terrain.soils())
    assert c["max_rel_sinkage"] < 1e-8 and c["max_rel_Fx"] < 1e-8 and c["max_rel_My"] < 1e-8


@pytest.mark.parametrize("grade", [0.0, 15.0])
def test_force_and_moment_balance_at_steady_state(flat_sim, rover, grade):
    st = flat_sim.steady_on_slope(grade)
    assert st.converged
    b = checks.balance_check(st, rover)
    assert b["force"] < 1e-9 and b["moment_roll_pitch"] < 1e-9
    # on a grade the rear (bogie) wheels carry more than the front ones
    if grade > 0:
        assert st.Fz[2] > st.Fz[0]


def test_speed_mode_needs_more_slip_uphill(flat_sim):
    s = [flat_sim.steady_on_slope(a).s_cmd for a in (0.0, 10.0, 20.0)]
    assert s[0] < s[1] < s[2]


def test_max_grade_vs_drawbar_pull_estimate(rover, flat_sim):
    full = flat_sim.max_slope(lo=20.0, hi=40.0, tol=0.5)
    est = checks.dp_grade_estimate(rover, SoilParams(), s_limit=flat_sim.s_limit, tol=0.1)["max_slope_deg"]
    # run with tol 0.05: full model 28.78 deg, estimate 31.49 deg (load transfer to the rear wheels)
    assert 26.0 < full < est < 34.0
    assert est - full < 4.0


def test_crater_depth_follows_pike_ratio():
    c = Crater(0.0, 0.0, 4.0, rim_ratio=0.02)
    assert c.depth_ratio == PIKE_1977_DEPTH_RATIO == 0.2
    t = LunarTerrain(craters=[c])
    assert float(t.height(0.0, 0.0)) == pytest.approx(0.02 * 4.0 - 0.2 * 4.0, abs=1e-12)
    assert abs(float(t.height(40.0, 0.0))) < 1e-3                    # flank decays as r^-3


def test_soil_regions_select_per_wheel_parameters():
    loose = SoilParams(k_phi=400_000.0)                               # test input, not a published set
    t = LunarTerrain(regions=[SoilRegion(loose, "box", (-1, 1, 0.0, 1.0))])
    assert list(t.soil_index(np.array([0.0, 0.0]), np.array([0.3, -0.3]))) == [1, 0]
    st = RoverSimulator(RoverParams(), t).run(distance=0.0, v=0.0, dt=0.0).states[0]
    assert list(st.soil_idx) == [1, 1, 1, 0, 0, 0]
    assert st.sinkage[:3].mean() > st.sinkage[3:].mean()             # softer soil on the left


def test_crater_rim_crossing_keeps_every_wheel_loaded(rover):
    t = LunarTerrain(craters=[Crater(1.3, 0.3, 2.0, rim_ratio=0.02, depth_ratio=0.1)])
    tr = RoverSimulator(rover, t).run(distance=0.8, v=0.1, dt=0.5)
    assert tr.failed_at is None
    assert all(s.converged and (s.Fz > 0).all() for s in tr.states)
    assert max(abs(math.degrees(s.pose.roll)) for s in tr.states) > 0.5      # off-centre path: the rover rolls


def test_surrogate_range_flags_and_errors(flat_sim):
    soil = SoilParams()
    num = BekkerWongContact(0.125, 0.06)
    exact = SurrogateContact(lambda s, z: num(s, z, [soil], 0), (0.0, 0.75), (0.002, 0.058), soil, 0.125, 0.06)
    st = flat_sim.steady_on_slope(27.0)                               # slip above 0.75: outside the box
    assert st.s_cmd > 0.75
    r = checks.surrogate_vs_numerical([st], exact, num, [soil])
    assert r["n_outside"] == 6 and r["Fx_max_rel_load"] < 1e-12
    exact([0.2, 0.9], [0.01, 0.01], [soil, SoilParams(c=0.0)], [0, 1])
    assert exact.log.slip_out == 1 and exact.log.soil_mismatch == 1 and list(exact.last_flags) == [False, True]
    with pytest.raises(ValueError):
        exact.check_wheel(0.2, 0.06)


def test_rover_exports(tmp_path, rover):
    glb = export_rover(RoverParams(n_grousers=12), str(tmp_path / "rover.glb"))
    assert open(glb, "rb").read(4) == b"glTF"
    stl = export_rover(rover, str(tmp_path / "rover.stl"))
    assert len(open(stl, "rb").read()) > 84


def test_twin_scene_motion_round_trip(tmp_path, rover):
    from pinneapple_simulation.rover import viz
    tr = RoverSimulator(rover, LunarTerrain()).run(distance=0.1, v=0.1, dt=0.5)
    sc = viz.twin_scene(tr)
    path = sc.export(str(tmp_path / "twin"), with_viewer=False)
    m = json.load(open(path))
    assert {mo["part"] for mo in m["motions"]} >= {"body", "L_front", "R_rear"}
    mo = next(mo for mo in m["motions"] if mo["part"] == "body")
    raw = np.fromfile(tmp_path / "twin" / "fields.bin", dtype="<f4")
    data = raw[mo["offset"] // 4: mo["offset"] // 4 + 7 * mo["steps"]].reshape(-1, 7)
    np.testing.assert_allclose(np.linalg.norm(data[:, 3:], axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        sc.add_motion("body", np.zeros((1, 3)), np.tile([1.0, 0, 0, 0], (1, 1)))
