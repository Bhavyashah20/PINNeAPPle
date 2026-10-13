"""``pp.`` shortcuts of the helpers consolidated from apps, examples and satellite repos resolve lazily and work."""
import json

import numpy as np
import pytest

import pinneapple as pp


@pytest.mark.parametrize("name, module", [
    ("uncertainty", "pinneapple_analysis.uncertainty"), ("closed_form", "pinneapple_physics.closed_form"),
    ("excitation", "pinneapple_systems.excitation"), ("reporting", "pinneapple_tools.reporting")])
def test_submodule_shortcuts(name, module):
    assert getattr(pp, name).__name__ == module


def test_function_shortcuts_against_independent_values():
    t = np.arange(4000) * 0.01
    assert pp.dominant_frequency(np.sin(2 * np.pi * 3.3 * t), 0.01) == pytest.approx(3.3, rel=1e-3)
    assert json.loads(json.dumps(pp.to_jsonable({"a": np.float32(1.5), "b": float("nan")}))) == {"a": 1.5, "b": None}
    r = pp.propagate_tolerances(lambda X: 2 * X[:, 0] + 3 * X[:, 1], [1.0, 2.0], [0.1, 0.2], n=200_000, seed=0)
    assert float(np.ravel(r.mean)[0]) == pytest.approx(8.0, abs=0.01)                     # 2·1 + 3·2
    assert float(np.ravel(r.std)[0]) == pytest.approx(np.hypot(2 * 0.1, 3 * 0.2), rel=0.01)
    assert callable(pp.assess_drift) and callable(pp.score_against_reference)


def test_as_tensor_shortcut():
    torch = pytest.importorskip("torch")
    x = torch.ones(3, 1)
    assert pp.as_tensor(x) is x

