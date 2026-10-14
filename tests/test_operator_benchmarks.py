"""Reference solvers of the operator-learning benchmarks (operator_benchmarks.py), checked on exact solutions."""
import numpy as np
import pytest
import torch

from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
    FourierInterpolant,
    burgers_1d,
    darcy_2d,
    li_forcing,
    navier_stokes_2d_vorticity,
    periodic_grf,
    threshold_coefficient,
)


def test_taylor_green_vorticity_decays_exactly():
    n, nu, t = 32, 0.01, 1.0
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    w0 = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)        # eigenfunction: u . grad w = 0
    _, w = navier_stokes_2d_vorticity(w0, nu, t, dt=0.01)
    assert np.allclose(w[0, -1], w0 * np.exp(-nu * 8 * np.pi ** 2 * t), atol=1e-10)


def test_navier_stokes_forced_random_field_is_finite_and_mean_free():
    w0 = periodic_grf(2, (32, 32), alpha=2.5, tau=7.0, seed=0)
    t, w = navier_stokes_2d_vorticity(w0, 1e-3, 1.0, forcing=li_forcing(32), dt=1e-2, n_save=4)
    assert w.shape == (2, 5, 32, 32) and np.allclose(t, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.all(np.isfinite(w)) and np.abs(w[:, -1].mean(axis=(1, 2))).max() < 1e-10


def test_burgers_matches_cole_hopf():
    n, nu, t_end = 128, 0.1, 0.5
    x = np.arange(n) / n

    def exact(t):
        e = np.exp(-nu * 4 * np.pi ** 2 * t)
        return 4 * np.pi * nu * e * np.sin(2 * np.pi * x) / (2 + e * np.cos(2 * np.pi * x))

    t, u = burgers_1d(np.stack([exact(0.0)] * 2), nu, t_end, dt=1e-3, n_save=2)
    assert u.shape == (2, 3, n)
    assert np.abs(u[0, -1] - exact(t_end)).max() < 1e-8
    assert np.abs(u[0, 1] - exact(t[1])).max() < 1e-8


def test_burgers_conserves_the_mean():
    u0 = periodic_grf(3, (256,), alpha=2.0, tau=5.0, sigma=25.0, seed=1) + 0.3
    _, u = burgers_1d(u0, 0.1, 1.0, dt=2e-4)
    assert np.allclose(u[:, -1].mean(1), u0.mean(1), atol=1e-10)


def test_darcy_manufactured_solution_second_order():
    errs = []
    for n in (16, 32, 64):
        h = 1.0 / n
        xc = (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        a = 1 + X
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        f = -(np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)) + a * 2 * np.pi ** 2 * exact
        errs.append(np.abs(darcy_2d(a, f) - exact).max())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 2e-3 and np.all(rates > 1.8)


def test_darcy_piecewise_constant_is_symmetric_and_positive():
    a = threshold_coefficient(periodic_grf(1, (32, 32), alpha=2.0, tau=3.0, seed=2)[0])
    assert set(np.unique(a)) <= {3.0, 12.0}
    u = darcy_2d(a)
    assert u.min() > 0 and u.max() < 1 / 8 / 3 + 1e-12          # bounded by the a = 3 solution's maximum


def test_periodic_grf_variance_matches_the_spectrum():
    n, alpha, tau, sigma = 64, 2.0, 5.0, 25.0
    u = periodic_grf(400, (n,), alpha=alpha, tau=tau, sigma=sigma, seed=3)
    k = np.fft.fftfreq(n, 1.0 / n)
    amp2 = (sigma * (4 * np.pi ** 2 * k ** 2 + tau ** 2) ** (-alpha / 2)) ** 2
    amp2[0] = 0
    assert u.var() == pytest.approx(amp2.sum(), rel=0.1)
    assert np.allclose(periodic_grf(2, (8, 8), alpha=2, tau=3, seed=5), periodic_grf(2, (8, 8), alpha=2, tau=3, seed=5))


def test_fourier_interpolant_values_and_derivatives():
    n = 32
    x = np.arange(n) / n
    f1 = FourierInterpolant(np.sin(2 * np.pi * x) + 0.5 * np.cos(6 * np.pi * x), dtype=torch.float64)
    xt = torch.rand(50, 1, dtype=torch.float64, requires_grad=True)
    v = f1(xt)
    dv = torch.autograd.grad(v.sum(), xt)[0]
    xs = xt.detach().numpy()
    assert np.allclose(v.detach().numpy(), np.sin(2 * np.pi * xs) + 0.5 * np.cos(6 * np.pi * xs), atol=1e-10)
    assert np.allclose(dv.numpy(), 2 * np.pi * np.cos(2 * np.pi * xs) - 3 * np.pi * np.sin(6 * np.pi * xs), atol=1e-9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f2 = FourierInterpolant(np.cos(2 * np.pi * X) * np.sin(4 * np.pi * Y))
    pts = torch.rand(40, 2)
    got = f2(pts[:, :1], pts[:, 1:]).numpy().ravel()
    p = pts.numpy()
    assert np.allclose(got, np.cos(2 * np.pi * p[:, 0]) * np.sin(4 * np.pi * p[:, 1]), atol=1e-4)
