"""OpenModelica bridge: .mo -> FMU (buildModelFMU) and direct simulation, FMU sweeps, Latin hypercube.
The omc tests run when omc is on the PATH or the OpenModelica Docker image is present, and skip otherwise."""
import os

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.modelica.sweep import FMUDataset, latin_hypercube

MODELS = os.path.join(os.path.dirname(__file__), "..", "examples", "modelica", "models")
TANK = os.path.join(MODELS, "DrainingTank.mo")


def h_steady(cd, qin=0.01, a=0.01, g=9.81):
    return (qin / (cd * a)) ** 2 / (2 * g)


def test_latin_hypercube_strata():
    u = latin_hypercube(16, 3, seed=2)
    assert u.shape == (16, 3) and (u >= 0).all() and (u < 1).all()
    for j in range(3):                                     # one point per stratum in every dimension
        assert sorted(np.floor(u[:, j] * 16).astype(int)) == list(range(16))


def test_dataset_split_and_roundtrip(tmp_path):
    ds = FMUDataset(["a", "b"], np.arange(20.0).reshape(10, 2), np.linspace(0, 1, 5),
                    {"y": np.arange(50.0).reshape(10, 5)}, {"a": (0, 20), "b": (0, 20)})
    tr, te = ds.split(3, seed=1)
    assert len(tr.params) == 7 and len(te.params) == 3
    assert not set(map(tuple, tr.params)) & set(map(tuple, te.params))
    ds.save(tmp_path / "d.npz")
    back = FMUDataset.load(tmp_path / "d.npz")
    assert back.names == ["a", "b"] and np.array_equal(back.outputs["y"], ds.outputs["y"])
    assert back.bounds["b"] == (0.0, 20.0)


@pytest.fixture(scope="module")
def omc():
    from pinneapple_simulation.external_solvers.modelica.omc import omc_available

    how = omc_available()
    if how is None:
        pytest.skip("OpenModelica not available (omc or the Docker image)")
    return how


def test_build_fmu_reaches_torricelli_equilibrium(omc, tmp_path):
    pytest.importorskip("fmpy")
    from pinneapple_simulation.external_solvers.modelica import FMUSimConfig, simulate_fmu
    from pinneapple_simulation.external_solvers.modelica.omc import build_fmu

    fmu = build_fmu(TANK, "DrainingTank", tmp_path)
    r = simulate_fmu(fmu, FMUSimConfig(stop_time=3000, output_interval=10, parameters={"Cd": 0.7},
                                       output_variables=["level"]))
    assert r["level"][0] == pytest.approx(1.0)
    assert r["level"][-1] == pytest.approx(h_steady(0.7), rel=1e-3)
    assert np.all(np.diff(r["level"]) <= 1e-9)              # q_out > q_in from h0 = 1 m down: monotone


def test_simulate_mo_override_and_time(omc, tmp_path):
    from pinneapple_simulation.external_solvers.modelica.omc import simulate_mo

    r = simulate_mo(TANK, "DrainingTank", stop_time=3000, parameters={"Cd": 0.5}, workdir=tmp_path,
                    variables=["time", "level", "Cd"])
    assert r["time"][0] == 0.0 and r["time"][-1] == pytest.approx(3000.0)
    assert r["Cd"][0] == pytest.approx(0.5)
    assert r["level"][-1] == pytest.approx(h_steady(0.5), rel=1e-3)


def test_fmu_sweep(omc, tmp_path):
    pytest.importorskip("fmpy")
    from pinneapple_simulation.external_solvers.modelica.omc import build_fmu
    from pinneapple_simulation.external_solvers.modelica.sweep import fmu_sweep

    fmu = build_fmu(TANK, "DrainingTank", tmp_path)
    ds = fmu_sweep(fmu, {"Cd": (0.5, 0.8)}, n=4, stop_time=3000, output_interval=100, outputs=["level"], seed=0,
                   step_size=1.0)
    assert ds.outputs["level"].shape == (4, 31)
    np.testing.assert_allclose(ds.outputs["level"][:, -1], h_steady(ds.params[:, 0]), rtol=2e-3)
