"""pp.trust_card / field_checks against independent references: exact harmonic and Poisson solutions, the
divergence theorem, the order of the three-point residual, and degenerate inputs."""
import numpy as np
import pytest

from pinneapple_analysis.trust.field_card import field_checks, trust_card


def _grid(n, m=None, stretch=False):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, m or n)
    if stretch:                                   # a non-uniform grid, clustered at both ends
        x = 0.5 * (1.0 - np.cos(np.pi * x))
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def _check(checks, name):
    return next(c for c in checks if c.name.startswith(name))


def _harmonic(X, Y):
    return np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)


@pytest.mark.parametrize("n", [11, 41, 201])
@pytest.mark.parametrize("offset", [0.0, 300.0])
def test_linear_field_is_harmonic(n, offset):
    # regression: u = x + 2 y has laplace(u) = 0 exactly, but its finite-difference terms are pure round-off and
    # the residual used to be judged against that round-off, failing a correct field (and rejecting the card)
    x, y, X, Y = _grid(n)
    card = trust_card({"x": x, "y": y}, offset + X + 2 * Y, pde="laplace")
    assert _check(card.checks, "PDE residual").passed is True
    assert card.decision != "REJECT"


def test_linear_field_in_float32_is_harmonic():
    x, y, X, Y = _grid(101)
    u = (300.0 + X + Y).astype(np.float32)
    assert _check(field_checks({"x": x, "y": y}, u, pde="laplace"), "PDE residual").passed is True


def test_non_harmonic_field_still_fails():
    x, y, X, Y = _grid(41)
    r = _check(field_checks({"x": x, "y": y}, X ** 2 + Y ** 2, pde="laplace"), "PDE residual")
    assert r.passed is False and r.value > 0.5


@pytest.mark.parametrize("stretch", [False, True])
def test_residual_converges_at_second_order(stretch):
    # the three-point formula is second order on a uniform grid; on a smooth stretched grid the truncation error
    # of the non-uniform formula is first order locally, but the mapped grid keeps the global rate near two
    vals = []
    for n in (21, 41, 81):
        x, y, X, Y = _grid(n, stretch=stretch)
        vals.append(_check(field_checks({"x": x, "y": y}, _harmonic(X, Y), pde="laplace"), "PDE residual").value)
    orders = np.log2(np.array(vals[:-1]) / np.array(vals[1:]))
    assert np.all(orders > 1.7), orders


def test_poisson_conservation_matches_divergence_theorem():
    # u = sin(pi x) sin(pi y): laplace(u) = -2 pi^2 u, net outward flux of grad u = integral of f = -8
    x, y, X, Y = _grid(81)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = lambda P: -2 * np.pi ** 2 * np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])  # noqa: E731
    checks = field_checks({"x": x, "y": y}, u, pde="poisson", source=f,
                          dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0})
    cons = _check(checks, "conservation")
    assert cons.passed is True and cons.value < 1e-3
    assert _check(checks, "PDE residual").passed is True


def test_conservation_on_stretched_grid():
    x, y, X, Y = _grid(61, stretch=True)
    cons = _check(field_checks({"x": x, "y": y}, _harmonic(X, Y), pde="laplace"), "conservation")
    assert cons.passed is True and cons.value < 1e-2


def test_wrong_source_breaks_conservation():
    x, y, X, Y = _grid(41)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    cons = _check(field_checks({"x": x, "y": y}, u, pde="poisson", source=1.0), "conservation")
    assert cons.passed is False


def test_heat_exact_solution_and_steady_linear_profile():
    x = np.linspace(0, 1, 41)
    t = np.linspace(0, 0.1, 41)
    X, T = np.meshgrid(x, t, indexing="ij")
    u = np.exp(-np.pi ** 2 * T) * np.sin(np.pi * X)
    assert _check(field_checks({"x": x, "t": t}, u, pde="heat"), "PDE residual").passed is True
    # steady and linear in x: every term is round-off
    assert _check(field_checks({"x": x, "t": t}, 2.0 + X + 0 * T, pde="heat"), "PDE residual").passed is True


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_field_is_rejected(bad):
    x, y, X, Y = _grid(21)
    u = _harmonic(X, Y)
    u[5, 5] = bad
    card = trust_card({"x": x, "y": y}, u, pde="laplace")
    assert card.decision == "REJECT"
    assert _check(card.checks, "finite values").passed is False


def test_two_point_grid_reports_not_run_instead_of_crashing():
    # regression: the conservation check needs three points per axis and used to raise NumPy's
    # "Shape of array too small to calculate a numerical gradient"
    x = y = np.array([0.0, 1.0])
    checks = field_checks({"x": x, "y": y}, np.zeros((2, 2)), pde="laplace")
    assert _check(checks, "conservation").passed is None
    assert _check(checks, "PDE residual").passed is None


def test_torch_tensor_with_grad_is_accepted():
    torch = pytest.importorskip("torch")
    x, y, X, Y = _grid(21)
    u = torch.tensor(_harmonic(X, Y), requires_grad=True)
    card = trust_card({"x": x, "y": y}, u, pde="laplace",
                      dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0,
                                 "y_max": lambda P: np.sin(np.pi * P[:, 0])})
    assert card.decision == "APPROVED"


def test_reference_relative_l2_hand_computed():
    x, y, X, Y = _grid(11)
    ref = _harmonic(X, Y)
    c = _check(field_checks({"x": x, "y": y}, 1.03 * ref, reference=ref), "reference")
    assert c.value == pytest.approx(0.03, rel=1e-12) and c.passed is True
