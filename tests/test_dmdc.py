"""DMD with control (pinneapple_neural.architectures.rom.DMDc): identifies a known linear system exactly."""
import numpy as np
import pytest

from pinneapple_neural.architectures.rom import DMDc


def _system(rng, n=4, m=2, steps=60):
    A = 0.9 * np.linalg.qr(rng.normal(size=(n, n)))[0]          # stable: spectral radius 0.9
    B = rng.normal(size=(n, m))
    U = rng.normal(size=(steps, m))
    X = np.empty((steps + 1, n))
    X[0] = rng.normal(size=n)
    for k in range(steps):
        X[k + 1] = A @ X[k] + B @ U[k]
    return A, B, X, U


def test_recovers_A_and_B_from_one_trajectory():
    A, B, X, U = _system(np.random.default_rng(0))
    model = DMDc().fit(X, U)
    np.testing.assert_allclose(model.A_, A, atol=1e-10)
    np.testing.assert_allclose(model.B_, B, atol=1e-10)
    np.testing.assert_allclose(model.c_, 0.0)
    np.testing.assert_allclose(model.rollout(X[0], U), X, atol=1e-8)


def test_affine_pairs_and_predict():
    rng = np.random.default_rng(1)
    A, B, _, _ = _system(rng)
    c = rng.normal(size=4)
    Xp, Up = rng.normal(size=(50, 4)), rng.normal(size=(50, 2))
    Yp = Xp @ A.T + Up @ B.T + c
    model = DMDc(affine=True).fit(Xp, Up, Yp)
    np.testing.assert_allclose(model.A_, A, atol=1e-10)
    np.testing.assert_allclose(model.B_, B, atol=1e-10)
    np.testing.assert_allclose(model.c_, c, atol=1e-10)
    np.testing.assert_allclose(model.predict(Xp, Up), Yp, atol=1e-10)
    np.testing.assert_allclose(model.predict(Xp[3], Up[3]), Yp[3], atol=1e-10)


def test_ridge_matches_normal_equations():
    rng = np.random.default_rng(2)
    X, U, Y = rng.normal(size=(30, 3)), rng.normal(size=(30, 1)), rng.normal(size=(30, 3))
    model = DMDc().fit(X, U, Y, ridge=0.5)
    Z = np.hstack([X, U])
    W = np.linalg.solve(Z.T @ Z + 0.5 * np.eye(4), Z.T @ Y)
    np.testing.assert_allclose(np.hstack([model.A_, model.B_]), W.T, atol=1e-12)


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        DMDc().predict(np.zeros(2), np.zeros(1))
