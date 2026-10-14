"""Tests for ``pinneapple_analysis.verification.convergence`` (Richardson
extrapolation / Grid Convergence Index and the ``mesh_independence_study``
driver), added to close the capability-audit gap: PINNeAPPle had a real
mesh-*quality* module (``mesh_intelligence.py``) but nothing that checks
whether a SOLUTION is actually converging under grid refinement.

Verification strategy: build synthetic "solvers" whose discretization
error is ANALYTICALLY known to scale as ``C * h^p`` for a chosen order
``p`` -- a second-order central-difference approximation of a known
derivative (p=2) and a first-order forward-difference approximation
(p=1) -- and confirm ``richardson_extrapolate``/``mesh_independence_study``
recover the correct observed order, an extrapolated value close to the
true analytical value, and a GCI that shrinks under refinement. This
mirrors this repo's own verification ethic (see
``tests/test_lane_emden_numerical_validation.py``,
``tests/test_manufactured_solutions.py``): construct a case whose right
answer is known in closed form, then check the tool recovers it -- never
just check the tool runs without error.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.verification.convergence import (
    ConvergenceResult,
    MeshIndependenceStudyResult,
    mesh_independence_study,
    richardson_extrapolate,
)

# ---------------------------------------------------------------------------
# Synthetic known-order problems
# ---------------------------------------------------------------------------
# f(x) = sin(x); f'(x0) = cos(x0) exactly. Finite-difference approximations
# of a known, analytically-derived order are used as the "solver" so the
# TRUE discretization-error exponent is known ahead of time (not estimated
# from anything -- the whole point is to check the tool recovers it).

_X0 = 0.6  # an arbitrary point away from 0/symmetry points, to avoid accidental error cancellation
_TRUE_DERIVATIVE = float(np.cos(_X0))


def _central_difference(n_points: int) -> float:
    """Second-order-accurate (O(h^2)) central-difference approximation of
    f'(x0) for f=sin, on a domain of fixed physical extent [0, 2] subdivided
    into ``n_points`` cells (so h = 2/n_points, and "resolution" = n_points
    is a natural ascending refinement knob, exactly the convention
    ``pinneapple_simulation.numerical_solvers`` solvers use: more points =
    finer grid)."""
    h = 2.0 / n_points
    return (np.sin(_X0 + h) - np.sin(_X0 - h)) / (2.0 * h)


def _forward_difference(n_points: int) -> float:
    """First-order-accurate (O(h)) forward-difference approximation of
    f'(x0) for f=sin."""
    h = 2.0 / n_points
    return (np.sin(_X0 + h) - np.sin(_X0)) / h


# ---------------------------------------------------------------------------
# richardson_extrapolate: direct unit tests
# ---------------------------------------------------------------------------

def test_richardson_extrapolate_recovers_second_order_scheme():
    r = 2.0
    n_fine = 64
    f_coarse = _central_difference(n_fine // (r * r))
    f_medium = _central_difference(int(n_fine // r))
    f_fine = _central_difference(n_fine)

    result = richardson_extrapolate(f_coarse, f_medium, f_fine, r)
    assert isinstance(result, ConvergenceResult)
    assert result.observed_order == pytest.approx(2.0, abs=0.05)
    assert result.extrapolated_value == pytest.approx(_TRUE_DERIVATIVE, abs=1e-6)
    assert result.gci_fine > 0.0
    assert result.gci_coarse > 0.0
    # With a clean, asymptotic-range synthetic problem the GCI self-consistency
    # ratio should land very close to 1.
    assert result.asymptotic_ratio == pytest.approx(1.0, abs=0.05)
    assert result.is_asymptotic is True


def test_richardson_extrapolate_recovers_first_order_scheme():
    r = 2.0
    n_fine = 512  # forward difference converges slowly (O(h)); use a finer base
    f_coarse = _forward_difference(n_fine // int(r * r))
    f_medium = _forward_difference(n_fine // int(r))
    f_fine = _forward_difference(n_fine)

    result = richardson_extrapolate(f_coarse, f_medium, f_fine, r)
    assert result.observed_order == pytest.approx(1.0, abs=0.05)
    assert result.extrapolated_value == pytest.approx(_TRUE_DERIVATIVE, abs=1e-3)


def test_gci_shrinks_under_refinement():
    """Refining further (larger n_fine base) should shrink both GCI values --
    the discretization-error uncertainty band must tighten as the grid gets finer."""
    r = 2.0

    def _study(n_fine):
        f_coarse = _central_difference(n_fine // 4)
        f_medium = _central_difference(n_fine // 2)
        f_fine = _central_difference(n_fine)
        return richardson_extrapolate(f_coarse, f_medium, f_fine, r)

    coarse_study = _study(32)
    fine_study = _study(256)
    assert fine_study.gci_fine < coarse_study.gci_fine
    assert fine_study.gci_coarse < coarse_study.gci_coarse


def test_richardson_extrapolate_with_p_assumed_skips_order_estimation():
    r = 2.0
    f_coarse = _central_difference(16)
    f_medium = _central_difference(32)
    f_fine = _central_difference(64)
    result = richardson_extrapolate(f_coarse, f_medium, f_fine, r, p_assumed=2.0)
    assert result.observed_order == 2.0
    assert result.extrapolated_value == pytest.approx(_TRUE_DERIVATIVE, abs=1e-6)


def test_richardson_extrapolate_handles_array_valued_quantity():
    """The quantity of interest may be an array (e.g. a full field) -- the
    extrapolated value should keep that shape, while order/GCI reduce to
    scalars via the L2 norm of the pairwise differences (see the module
    docstring's ``_flat_norm`` note)."""
    r = 2.0
    xs = np.array([0.3, 0.6, 0.9])

    def _central_difference_vec(n_points):
        h = 2.0 / n_points
        return (np.sin(xs + h) - np.sin(xs - h)) / (2.0 * h)

    f_coarse = _central_difference_vec(16)
    f_medium = _central_difference_vec(32)
    f_fine = _central_difference_vec(64)

    result = richardson_extrapolate(f_coarse, f_medium, f_fine, r)
    assert result.observed_order == pytest.approx(2.0, abs=0.1)
    assert isinstance(result.extrapolated_value, np.ndarray)
    assert result.extrapolated_value.shape == xs.shape
    np.testing.assert_allclose(result.extrapolated_value, np.cos(xs), atol=1e-5)


def test_richardson_extrapolate_rejects_refinement_ratio_not_greater_than_one():
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 1.0, 1.0, r=1.0)
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 1.0, 1.0, r=0.5)


def test_richardson_extrapolate_rejects_identical_fine_and_medium_values():
    with pytest.raises(ValueError):
        richardson_extrapolate(2.0, 1.0, 1.0, r=2.0)


def test_richardson_extrapolate_warns_on_oscillatory_convergence():
    # f_coarse - f_medium and f_medium - f_fine have opposite signs.
    with pytest.warns(UserWarning, match="oscillatory"):
        richardson_extrapolate(1.0, 2.0, 1.5, r=2.0)


# ---------------------------------------------------------------------------
# mesh_independence_study: driver tests
# ---------------------------------------------------------------------------

def test_mesh_independence_study_recovers_second_order_and_infers_ratio():
    resolutions = [16, 32, 64]

    def solve_fn(resolution):
        return _central_difference(resolution)  # the "solution" is already the scalar derivative estimate

    def quantity_extractor(solution):
        return solution

    study = mesh_independence_study(solve_fn, resolutions, quantity_extractor)
    assert isinstance(study, MeshIndependenceStudyResult)
    assert study.resolutions == resolutions
    assert len(study.values) == 3
    assert study.refinement_ratio_used == pytest.approx(2.0)
    assert study.refinement_ratios == pytest.approx([2.0, 2.0])
    assert study.convergence.observed_order == pytest.approx(2.0, abs=0.05)
    assert study.convergence.extrapolated_value == pytest.approx(_TRUE_DERIVATIVE, abs=1e-6)


def test_mesh_independence_study_uses_more_than_three_resolutions_finest_three():
    """Extra, coarser resolutions may be included for plotting a full
    convergence curve -- the Richardson analysis itself must still only use
    the three FINEST."""
    resolutions = [8, 16, 32, 64]

    def solve_fn(resolution):
        return _central_difference(resolution)

    study = mesh_independence_study(solve_fn, resolutions, lambda s: s)
    assert len(study.values) == 4
    # Recompute expected convergence result directly from the three finest values.
    expected = richardson_extrapolate(study.values[-3], study.values[-2], study.values[-1], r=2.0)
    assert study.convergence.observed_order == pytest.approx(expected.observed_order)
    assert study.convergence.extrapolated_value == pytest.approx(expected.extrapolated_value)


def test_mesh_independence_study_requires_at_least_three_resolutions():
    with pytest.raises(ValueError):
        mesh_independence_study(lambda r: _central_difference(r), [16, 32], lambda s: s)


def test_mesh_independence_study_requires_ascending_resolutions():
    with pytest.raises(ValueError):
        mesh_independence_study(lambda r: _central_difference(r), [64, 32, 16], lambda s: s)


def test_mesh_independence_study_warns_on_non_constant_refinement_ratio():
    resolutions = [16, 32, 96]  # ratios 2.0, 3.0 -- not constant

    def solve_fn(resolution):
        return _central_difference(resolution)

    with pytest.warns(UserWarning, match="not roughly constant"):
        study = mesh_independence_study(solve_fn, resolutions, lambda s: s)
    assert study.refinement_ratio_used == pytest.approx(3.0)  # ratio between the two finest
    # regression: both pairs used to be treated as r = 3, which reported this second-order scheme as order 1.1
    assert study.convergence.observed_order == pytest.approx(2.0, abs=0.01)
    assert study.convergence.extrapolated_value == pytest.approx(_TRUE_DERIVATIVE, abs=1e-6)
    assert study.convergence.refinement_ratio_coarse == pytest.approx(2.0)


def test_richardson_extrapolate_reproduces_celik_2008_table_1():
    """Celik et al. (2008), ASME J. Fluids Eng. 130(7) 078001, Table 1, column phi: N = 18000, 8000, 4500 cells
    (r21 = 1.5, r32 = 1.333), phi = 6.063, 5.972, 5.863 -> p = 1.53, phi_ext = 6.1685, e_a21 = 1.5 %,
    GCI_fine = 2.17 %."""
    res = richardson_extrapolate(5.863, 5.972, 6.063, r=(1.5, 4.0 / 3.0))
    assert res.observed_order == pytest.approx(1.53, abs=0.005)
    assert res.extrapolated_value == pytest.approx(6.1685, abs=5e-4)
    assert res.gci_fine == pytest.approx(0.0217, abs=5e-4)


def test_richardson_extrapolate_non_constant_ratio_known_order():
    # f(h) = 1 + 0.3 h^2 on h = 0.3, 0.1, 0.05: ratios 3 and 2, exact order 2 and exact limit 1
    f = lambda h: 1.0 + 0.3 * h ** 2  # noqa: E731
    res = richardson_extrapolate(f(0.3), f(0.1), f(0.05), r=(2.0, 3.0))
    assert res.observed_order == pytest.approx(2.0, abs=1e-9)
    assert res.extrapolated_value == pytest.approx(1.0, abs=1e-12)
    assert res.asymptotic_ratio == pytest.approx(1.0, abs=5e-3)   # exactly f_fine / f_medium (relative errors)
    # a constant-ratio call gives the same answer as the general formula
    g = richardson_extrapolate(f(0.2), f(0.1), f(0.05), r=2.0)
    assert g.observed_order == pytest.approx(2.0) and g.refinement_ratio_coarse == 2.0


def test_richardson_extrapolate_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        richardson_extrapolate(1.0, float("nan"), 1.5, r=2.0)


def test_richardson_extrapolate_rejects_bad_ratio_pair():
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 1.2, 1.3, r=(2.0, 1.0))
    with pytest.raises(ValueError):
        richardson_extrapolate(1.0, 1.2, 1.3, r=(2.0, 2.0, 2.0))


def test_mesh_independence_study_accepts_explicit_refinement_ratio():
    resolutions = [16, 32, 64]

    def solve_fn(resolution):
        return _central_difference(resolution)

    study = mesh_independence_study(solve_fn, resolutions, lambda s: s, refinement_ratio=2.0)
    assert study.refinement_ratio_used == 2.0
    assert study.convergence.observed_order == pytest.approx(2.0, abs=0.05)


def test_mesh_independence_study_is_solver_agnostic_and_never_imports_solvers():
    """Confirm the module truly has no coupling to any specific PDE solver
    -- it must not IMPORT anything from pinneapple_simulation (mentioning
    FDM/FEM/LBM in the module docstring's motivation is fine; importing
    those modules would be the actual coupling this test guards against)."""
    import ast
    import inspect

    import pinneapple_analysis.verification.convergence as convergence_module

    source = inspect.getsource(convergence_module)
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(m.startswith("pinneapple_simulation") for m in imported_modules)
    allowed = {"__future__", "warnings", "dataclasses", "typing", "collections.abc", "numpy"}
    assert imported_modules <= allowed, f"unexpected import(s) coupling convergence.py to something solver-specific: {imported_modules - allowed}"
