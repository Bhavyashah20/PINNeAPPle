"""Derivative providers of `pinneapple check` against manufactured solutions: the grid residual converges at the
order of its stencil, so an exact solution passes the default tolerance on coarse grids."""
import numpy as np
import pytest

from pinneapple_check.config import DEFAULT_TOLERANCES
from pinneapple_check.numerics import GridD, PointD, rel_rms
from pinneapple_check.physics import equations

HEAT = {"k": 1.0, "q": 0.0, "rho_cp": 1.0}


def _heat_residual(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)          # harmonic
    e = equations("heat", GridD({"x": x, "y": y}, {"T": T}), HEAT)[0][0]
    return rel_rms(e.residual, e.terms, e.mask, e.scale)


@pytest.mark.parametrize("stretch", [False, True])
def test_grid_heat_residual_is_second_order(stretch):
    # regression: d(d(u)/dx)/dx used the one-sided edge derivative / h next to the boundary (O(h) there), so the
    # residual converged at order 1.5 and a harmonic field on a 17 x 17 grid scored 0.095 (> the 0.05 default)
    vals = []
    for n in (17, 33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        if stretch:
            x = 0.5 * (1.0 - np.cos(np.pi * x))
        vals.append(_heat_residual(x, np.linspace(0.0, 1.0, n)))
    orders = np.log2(np.array(vals[:-1]) / np.array(vals[1:]))
    assert np.all(orders > 1.85), orders
    assert vals[0] < DEFAULT_TOLERANCES["residual"]


def test_grid_mixed_derivative_and_linear_field():
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 2, 31)
    X, Y = np.meshgrid(x, y, indexing="ij")
    D = GridD({"x": x, "y": y}, {"f": X ** 2 * Y + 3 * X * Y, "g": 2 * X - Y + 5})
    assert np.allclose(D.dd("f", "x", "x"), 2 * Y)
    assert np.allclose(D.dd("f", "x", "y"), 2 * X + 3)
    e = equations("heat", GridD({"x": x, "y": y}, {"T": 300.0 + X + Y}), HEAT)[0][0]
    assert rel_rms(e.residual, e.terms, e.mask, e.scale) < 1e-10      # linear: exact up to round-off


def test_point_cloud_residual_decreases_with_density():
    # the weighted quadratic fit gives first-order second derivatives on a scattered cloud: documented, not 2nd order
    rng = np.random.default_rng(0)
    vals = []
    for n in (400, 1600, 6400):
        P = np.c_[rng.random((n, 2)), np.zeros(n)]
        T = np.sin(np.pi * P[:, 0]) * np.sinh(np.pi * P[:, 1]) / np.sinh(np.pi)
        e = equations("heat", PointD(P, {"T": T}, ["x", "y"]), HEAT)[0][0]
        vals.append(rel_rms(e.residual, e.terms, e.mask, e.scale))
    assert vals[0] > vals[1] > vals[2]


def _write_case(tmp_path, n_x):
    x, y = np.linspace(0, 1, n_x), np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y, indexing="ij")
    np.savez(tmp_path / "plate.npz", x=x, y=y, T=300.0 + X + Y)
    np.savez(tmp_path / "other.npz", x=y, y=y, T=np.full((30, 30), np.nan))
    (tmp_path / "pinneapple-check.toml").write_text(
        '[[result]]\nname = "plate"\npath = "plate.npz"\nphysics = "heat"\nunits = { T = "K" }\n'
        '[[result]]\nname = "other"\npath = "other.npz"\nunits = { T = "K" }\n')
    return tmp_path


def test_two_point_axis_is_not_evaluated_instead_of_crashing(tmp_path):
    # regression: numpy.gradient(edge_order=2) raised "Shape of array too small" and the whole run ended with a
    # traceback; an empty interior must not count as a passing residual either
    from pinneapple_check import run
    rep = run([_write_case(tmp_path, 2)])
    assert not [o for o in rep.outcomes if o.rule == "PC301"]
    assert any("residual not evaluated" in n for n in rep.notes)
    assert any(o.rule == "PC403" and not o.passed for o in rep.outcomes)      # the other target was still checked


def test_a_crashing_check_is_an_error_on_its_target_and_the_run_goes_on(tmp_path, monkeypatch):
    import pinneapple_check.engine as engine
    real = engine.check_result

    def boom(ctx, t):
        if t.name == "plate":
            raise RuntimeError("simulated bug")
        return real(ctx, t)
    monkeypatch.setattr(engine, "check_result", boom)
    rep = engine.run([_write_case(tmp_path, 30)])
    crash = [o for o in rep.outcomes if o.rule == "PC001" and not o.passed]
    assert crash and "simulated bug" in crash[0].message
    assert any(o.rule == "PC403" and o.target == "other" for o in rep.outcomes)
    assert rep.exit_code("error") != 0
