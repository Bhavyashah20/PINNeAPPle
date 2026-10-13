"""Periodic viscous Burgers generator in pinneapple_data.datasets.physics against independent references."""
import numpy as np

from pinneapple_data.datasets.physics import burgers_periodic_pairs, burgers_periodic_solve
from pinneapple_physics.closed_form.burgers import burgers_sine_exact


def test_sine_ic_matches_cole_hopf_exact():
    """u0 = -sin(pi x) is 2-periodic: on [0, 2) the solution is the Cole-Hopf one (Basdevant et al. 1986)."""
    nx, nu = 128, 0.05
    x = np.arange(nx) * (2.0 / nx)
    for T in (0.2, 0.6):
        u = burgers_periodic_solve(-np.sin(np.pi * x), T, nu, length=2.0)
        ref = burgers_sine_exact(x, np.full_like(x, T), nu)
        assert np.abs(u - ref).max() < 1e-6, (T, np.abs(u - ref).max())


def test_steep_front_resolved_by_refinement():
    """nu = 0.01/pi (the PINN benchmark): the automatic refinement resolves the front at x = 1."""
    nx, nu, T = 64, 0.01 / np.pi, 0.5
    x = np.arange(nx) * (2.0 / nx)
    u = burgers_periodic_solve(-np.sin(np.pi * x), T, nu, length=2.0)
    ref = burgers_sine_exact(x, np.full_like(x, T), nu)
    assert np.abs(u - ref).max() < 1e-4, np.abs(u - ref).max()                    # measured 2.9e-6


def test_small_amplitude_follows_heat_decay():
    """eps sin(2 pi k x): the nonlinear term is O(eps^2), so u(T) ~ eps exp(-nu (2 pi k)^2 T) sin(2 pi k x)."""
    nx, nu, T, eps, k = 64, 0.01, 1.0, 1e-4, 3
    x = np.arange(nx) / nx
    u = burgers_periodic_solve(eps * np.sin(2 * np.pi * k * x), T, nu)
    ref = eps * np.exp(-nu * (2 * np.pi * k) ** 2 * T) * np.sin(2 * np.pi * k * x)
    assert np.abs(u - ref).max() < 1e-3 * eps


def test_pairs_shapes_determinism_and_conservation():
    x, u0, uT = burgers_periodic_pairs(6, nx=64, T=0.3, nu=0.02, seed=3)
    assert x.shape == (64,) and u0.shape == uT.shape == (6, 64) and uT.dtype == np.float32
    assert np.allclose(x[1] - x[0], 1 / 64) and x[-1] < 1.0                     # periodic grid, no duplicate end
    x2, u02, uT2 = burgers_periodic_pairs(6, nx=64, T=0.3, nu=0.02, seed=3)
    assert np.array_equal(u0, u02) and np.array_equal(uT, uT2)
    assert not np.array_equal(u0, burgers_periodic_pairs(6, nx=64, T=0.3, nu=0.02, seed=4)[1])
    assert np.allclose(uT.mean(1), u0.mean(1), atol=1e-5)                       # mean of u is conserved
    assert np.abs(uT).max() <= np.abs(u0).max() + 1e-5                           # maximum principle
    assert np.isfinite(uT).all()
