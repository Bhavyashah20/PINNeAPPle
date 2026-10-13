"""TrustedLatentSurrogate: conformal error bound, design envelope and physics residual of a POD latent surrogate,
checked against a full finite-difference solve of 1-D reaction-diffusion."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pinneapple_neural.architectures.rom import ParametricPOD, TrustedLatentSurrogate

N = 120
_H = 1.0 / (N + 1)
_X = np.linspace(_H, 1 - _H, N)
_LAP = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N, N), format="csr") / _H ** 2
LO, HI = np.array([0.0, 0.3]), np.array([50.0, 0.7])        # decay k, source position xs


def _op(p):
    k, xs = p
    return (-_LAP + k * sp.identity(N)).tocsr(), np.exp(-((_X - xs) ** 2) / (2 * 0.06 ** 2))


def solve(p):
    L, f = _op(p)
    return spsolve(L.tocsc(), f)


def residual(p, u):
    L, f = _op(p)
    return float(np.linalg.norm(L @ u - f) / np.linalg.norm(f))


def _data(n, rng, lo=LO, hi=HI):
    P = lo + (hi - lo) * rng.random((n, 2))
    return P, np.array([solve(p) for p in P])


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    P_tr, X_tr = _data(40, rng)
    P_cal, X_cal = _data(60, rng)
    sur = TrustedLatentSurrogate(ParametricPOD(r=20, regressor="gpr", normalize=True), alpha=0.1,
                                 residual=residual).fit(P_tr, X_tr, P_cal, X_cal)
    return sur, rng


def test_physics_residual_of_the_full_solution_is_zero():
    p = np.array([10.0, 0.5])
    assert residual(p, solve(p)) < 1e-10


def test_conformal_bound_covers_in_distribution_errors(fitted):
    sur, rng = fitted
    P, X = _data(120, rng)
    a = sur.assess(P, tol=0.01)
    err = np.linalg.norm(a["X"] - X, axis=1) / np.linalg.norm(X, axis=1)
    assert np.median(err) < 0.01                                      # the surrogate itself is accurate
    assert np.mean(err <= a["error_bound"]) >= 0.8                    # 90 % target, 120 test points
    assert np.mean(a["decision"] == "REJECT") <= 0.15                 # in-envelope queries mostly pass


def test_far_designs_are_rejected_and_the_report_agrees(fitted):
    sur, _ = fitted
    far = np.array([[120.0, 0.5], [10.0, 0.95]])                      # decay 2.4x the box, source near the wall
    a = sur.assess(far, tol=0.01)
    assert np.all(a["envelope_p"] < sur.alpha_envelope)
    assert list(a["decision"]) == ["REJECT", "REJECT"]
    rep = sur.trust_report(far[0], tol=0.01)
    assert rep.decisao == "REJECT"
    assert {c.name for c in rep.checks} == {"conformal error bound", "design envelope", "physics residual"}
    inside = np.array([20.0, 0.5])
    assert sur.trust_report(inside, tol=1.0).decisao == sur.assess(inside[None], tol=1.0)["decision"][0]


def test_ensemble_spread_without_a_gp():
    rng = np.random.default_rng(1)
    P_tr, X_tr = _data(40, rng)
    P_cal, X_cal = _data(40, rng)
    sur = TrustedLatentSurrogate(ParametricPOD(r=20, regressor="rbf"), members=8).fit(P_tr, X_tr, P_cal, X_cal)
    a = sur.assess(P_cal[:5], tol=0.05)
    assert np.all(np.isfinite(a["error_bound"])) and np.all(np.isnan(a["residual"]))
    rep = sur.trust_report(P_cal[0], tol=0.05)
    assert any(c.status == "not_run" for c in rep.checks)
    with pytest.raises(ValueError):
        TrustedLatentSurrogate(ParametricPOD(regressor="rbf"), spread="gp").fit(P_tr, X_tr, P_cal, X_cal)
