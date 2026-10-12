"""FMU import through fmpy: an FMI 2.0 Co-Simulation FMU built with pythonfmu (damped oscillator) is simulated by
``simulate_fmu`` and packaged by ``fmu_to_upd``; the trajectory matches the exact solution."""
import textwrap

import numpy as np
import pytest

pytest.importorskip("fmpy")
pytest.importorskip("pythonfmu")

from pinneapple_simulation.external_solvers.modelica.fmi_runner import FMUSimConfig, fmu_to_upd, simulate_fmu  # noqa: E402

SOURCE = '''
from pythonfmu import Fmi2Causality, Fmi2Variability, Fmi2Slave, Real


class Oscillator(Fmi2Slave):
    """x'' + 2 zeta w x' + w^2 x = 0, x(0) = 1, v(0) = 0."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w, self.zeta, self.x, self.v = 2.0, 0.1, 1.0, 0.0
        self.register_variable(Real("w", causality=Fmi2Causality.parameter, variability=Fmi2Variability.tunable))
        self.register_variable(Real("zeta", causality=Fmi2Causality.parameter, variability=Fmi2Variability.tunable))
        self.register_variable(Real("x", causality=Fmi2Causality.output))
        self.register_variable(Real("v", causality=Fmi2Causality.output))

    def do_step(self, t, dt):
        h = dt / 20
        for _ in range(20):
            a = -2 * self.zeta * self.w * self.v - self.w ** 2 * self.x
            xm, vm = self.x + 0.5 * h * self.v, self.v + 0.5 * h * a
            am = -2 * self.zeta * self.w * vm - self.w ** 2 * xm
            self.x, self.v = self.x + h * vm, self.v + h * am
        return True
'''


@pytest.fixture(scope="module")
def fmu(tmp_path_factory):
    from pythonfmu.builder import FmuBuilder

    d = tmp_path_factory.mktemp("fmu")
    script = d / "Oscillator.py"
    script.write_text(textwrap.dedent(SOURCE))
    return FmuBuilder.build_FMU(script, dest=d)


def exact(t, w, zeta):
    wd = w * np.sqrt(1 - zeta ** 2)
    return np.exp(-zeta * w * t) * (np.cos(wd * t) + zeta * w / wd * np.sin(wd * t))


def test_simulate_fmu_matches_exact_solution(fmu):
    r = simulate_fmu(fmu, FMUSimConfig(stop_time=5.0, step_size=0.01, parameters={"w": 3.0}))
    assert {"time", "x", "v"} <= set(r)
    assert np.max(np.abs(r["x"] - exact(r["time"], 3.0, 0.1))) < 1e-5


def test_simulate_fmu_without_parameters_and_to_upd(fmu):
    r = simulate_fmu(fmu, FMUSimConfig(stop_time=1.0, step_size=0.01, output_variables=["x"]))
    assert set(r) == {"time", "x"}
    assert np.max(np.abs(r["x"] - exact(r["time"], 2.0, 0.1))) < 1e-5
    s = fmu_to_upd(fmu, FMUSimConfig(stop_time=1.0, step_size=0.01))
    assert s.provenance["source"] == "modelica_fmu" and "x" in s.state
