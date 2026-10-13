"""Reference tests for the three numerical components harvested with REA
(see docs/dev/rea_harvest/ledger.json). Every test compares against an
independent reference -- closed form, exact solution or an identity -- in the
same test function (D1 in .agent/DECISIONS.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.validation.field_metrics import field_error_metrics, spectral_error_shells
from pinneapple_simulation.numerical_solvers.etdrk4 import (
    ETDRK4, SimulationBlowUp, check_finite, fourier_wavenumbers)
from pinneapple_simulation.numerical_solvers.stable_fluids import StableFluids2D, stable_time_step


# --------------------------------------------------------------------------
# ETDRK4
# --------------------------------------------------------------------------

def test_etdrk4_linear_part_is_exact_closed_form():
    """N = 0: ETDRK4 must reproduce exp(L t) exactly (closed form)."""
    L = np.array([-1.0, -50.0, -1e4, 0.0, -1e-9])
    u0 = np.ones_like(L)
    res = ETDRK4(L, lambda u, t: np.zeros_like(u), dt=0.1).solve(u0, 1.0)
    np.testing.assert_allclose(res.u[-1], np.exp(L * res.t[-1]), rtol=1e-12, atol=1e-300)


def test_etdrk4_fourth_order_against_exact_exponential():
    """N(u) = c u with diagonal L: exact u = exp((L + c) t). Observed order ~4."""
    L = np.array([-2.0, -20.0, -200.0])
    c = 1.5
    u0 = np.array([1.0, 0.5, 2.0])
    exact = u0 * np.exp((L + c) * 1.0)
    errs = []
    for dt in (0.1, 0.05, 0.025):
        res = ETDRK4(L, lambda u, t: c * u, dt=dt).solve(u0, 1.0)
        errs.append(np.max(np.abs(res.u[-1] - exact)))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 3.6), orders


def test_etdrk4_complex_linear_part_dispersion_closed_form():
    """Purely dispersive L = i k^3 (linear KdV): |u_hat| preserved, phase exact."""
    k = fourier_wavenumbers(32)
    L = 1j * k ** 3
    u0 = np.exp(-np.arange(32) / 4.0).astype(complex)
    res = ETDRK4(L, lambda u, t: np.zeros_like(u), dt=0.01).solve(u0, 0.5)
    np.testing.assert_allclose(res.u[-1], u0 * np.exp(L * 0.5), rtol=1e-11, atol=1e-13)


def test_etdrk4_burgers_against_cole_hopf_exact_solution():
    """Viscous Burgers u_t + u u_x = nu u_xx on [0, 2pi), pseudo-spectral.

    Cole-Hopf exact solution with phi = 1 + a exp(-nu k0^2 t) cos(k0 x):
    u = 2 nu a k0 e^{-nu k0^2 t} sin(k0 x) / (1 + a e^{-nu k0^2 t} cos(k0 x)).
    """
    n, nu, a, k0 = 128, 0.1, 0.5, 1
    x = 2 * np.pi * np.arange(n) / n
    k = fourier_wavenumbers(n)

    def exact(t):
        e = a * np.exp(-nu * k0 ** 2 * t)
        return 2 * nu * e * k0 * np.sin(k0 * x) / (1 + e * np.cos(k0 * x))

    dealias = np.abs(k) < (2 / 3) * (n / 2)

    def N(u_hat, t):
        u = np.fft.ifft(u_hat).real
        return -0.5j * k * np.fft.fft(u * u) * dealias

    res = ETDRK4(-nu * k ** 2, N, dt=0.01).solve(np.fft.fft(exact(0.0)), 2.0)
    u_end = np.fft.ifft(res.u[-1]).real
    assert np.max(np.abs(u_end - exact(res.t[-1]))) < 1e-9


def test_etdrk4_blowup_is_detected_not_returned_as_data():
    """u' = u^2, u(0) = 1 blows up at t = 1 (closed form 1/(1-t))."""
    res = ETDRK4(np.array([0.0]), lambda u, t: u * u, dt=0.01).solve(
        np.array([1.0]), 2.0, check_every=1, blowup_limit=1e6)
    assert res.blew_up is not None
    assert 0.9 < res.blew_up.t <= 1.05
    assert np.all(np.isfinite(res.u))
    # Before the singularity the saved states follow the exact 1/(1-t).
    i = int(np.argmin(np.abs(res.t - 0.5)))
    assert res.u[i][0] == pytest.approx(1 / (1 - res.t[i]), rel=1e-7)
    with pytest.raises(SimulationBlowUp):
        check_finite(np.array([1.0, np.nan]))


# --------------------------------------------------------------------------
# Stable fluids
# --------------------------------------------------------------------------

def _random_velocity(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, n, n))


def test_projection_is_discretely_divergence_free_and_idempotent():
    """Identity checks: div(P u) = 0 to round-off and P(P u) = P u."""
    sf = StableFluids2D(64, 64)
    v = sf.project(_random_velocity(64))
    assert np.max(np.abs(sf.divergence(v))) < 1e-12 * np.max(np.abs(v)) * 64
    np.testing.assert_allclose(sf.project(v), v, atol=1e-12)


def test_projection_keeps_exact_divergence_free_field():
    """A stream-function field (Taylor-Green) is already solenoidal: P u = u."""
    sf = StableFluids2D(64, 64)
    u = np.stack([np.sin(sf.X) * np.cos(sf.Y), -np.cos(sf.X) * np.sin(sf.Y)])
    np.testing.assert_allclose(sf.project(u), u, atol=1e-12)


def test_taylor_green_viscous_decay_converges_to_exact_solution():
    """Taylor-Green vortex: u(t) = u(0) exp(-2 nu t) (exact NS solution, k=1).

    The nonlinear term is a pure gradient for this flow, so the error is the
    semi-Lagrangian interpolation error -- first order at fixed Courant
    number (Stam 1999). Check the observed order and the magnitude."""
    nu, t_end, errs = 0.05, 1.0, []
    for n, dt in ((32, 0.04), (64, 0.02), (128, 0.01)):
        sf = StableFluids2D(n, n, viscosity=nu)
        u0 = np.stack([np.sin(sf.X) * np.cos(sf.Y), -np.cos(sf.X) * np.sin(sf.Y)])
        state = sf.run(sf.initial_state(u0), t_end, dt=dt)
        exact = u0 * np.exp(-2 * nu * state.t)
        errs.append(np.linalg.norm(state.velocity - exact) / np.linalg.norm(exact))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((orders > 0.9) & (orders < 1.2)), orders
    assert errs[-1] < 2.5e-2, errs


def test_semi_lagrangian_translation_of_scalar_is_exact_for_whole_cells():
    """Uniform velocity moving exactly one cell per step: pure shift, exact."""
    n = 32
    sf = StableFluids2D(n, n, length=(1.0, 1.0))
    dye = np.random.default_rng(1).random((n, n))
    vel = np.zeros((2, n, n))
    vel[0] = 1.0                                   # u_x = 1
    out = sf.advect(dye, vel, sf.dx)               # one cell
    np.testing.assert_allclose(out, np.roll(dye, 1, axis=1), atol=1e-12)


def test_confinement_off_is_identity_and_on_injects_rotational_force():
    sf0 = StableFluids2D(64, 64)
    sf1 = StableFluids2D(64, 64, confinement=2.0)
    v = sf0.project(_random_velocity(64, 3))
    np.testing.assert_array_equal(sf0.confine(v, 0.1), v)
    w0 = np.abs(sf1.vorticity(v)).sum()
    w1 = np.abs(sf1.vorticity(sf1.project(sf1.confine(v, 0.1)))).sum()
    assert w1 > w0          # confinement amplifies vorticity (its purpose)


def test_stable_time_step_closed_forms():
    assert stable_time_step(2.0, 0.1, courant=0.5) == pytest.approx(0.025)
    assert stable_time_step(2.0, 0.1, courant=0.5, viscosity=1.0,
                            implicit_diffusion=False) == pytest.approx(0.1 ** 2 / 4)
    assert stable_time_step(0.0, 0.1) == np.inf


# --------------------------------------------------------------------------
# Field metrics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ndim,shape", [(1, (64,)), (2, (32, 32)), (3, (16, 16, 16))])
def test_spectral_shells_satisfy_parseval(ndim, shape):
    rng = np.random.default_rng(ndim)
    p, t = rng.standard_normal((3, *shape)), rng.standard_normal((3, *shape))
    shells = spectral_error_shells(p, t, ndim)
    mse = ((p - t) ** 2).mean(axis=tuple(range(-ndim, 0)))
    np.testing.assert_allclose(shells.sum(axis=-1), mse, rtol=1e-12)


def test_single_mode_error_lands_in_the_right_band_closed_form():
    """Error = A sin(m x) on n points: rmse = A/sqrt(2), all of it in shell m."""
    n, A = 64, 0.3
    x = 2 * np.pi * np.arange(n) / n
    t = np.cos(x)
    for m, band in ((2, "low"), (7, "mid"), (20, "high")):
        r = field_error_metrics(t + A * np.sin(m * x), t, ndim=1, bands=(4, 12))
        assert r.rmse == pytest.approx(A / np.sqrt(2), rel=1e-12)
        vals = {"low": r.frmse_low, "mid": r.frmse_mid, "high": r.frmse_high}
        assert vals[band] > 0
        assert all(v < 1e-12 for k, v in vals.items() if k != band)


def test_2d_negative_frequencies_are_counted():
    """sin(m x - m y) lives partly in negative-frequency quadrants (PDEBench's
    2-D loop skips them); its full energy must still be accounted for."""
    n, A, m = 32, 1.0, 3
    xs = 2 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    err = A * np.sin(m * X - m * Y)
    shells = spectral_error_shells(err, np.zeros_like(err), 2)
    assert shells.sum() == pytest.approx(A ** 2 / 2, rel=1e-12)
    assert shells[int(np.floor(np.sqrt(2) * m))] == pytest.approx(A ** 2 / 2, rel=1e-12)


def test_point_metrics_closed_forms():
    t = np.ones((4, 10, 10))
    p = t + 0.1                                   # constant offset
    r = field_error_metrics(p, t, ndim=2)
    assert r.rmse == pytest.approx(0.1)
    assert r.nrmse == pytest.approx(0.1)
    assert r.max_error == pytest.approx(0.1)
    assert r.crmse == pytest.approx(0.1)          # mean shifts by 0.1
    assert r.brmse == pytest.approx(0.1)
    assert r.frmse_low == pytest.approx(0.1 / 4)  # shell 0 holds it; averaged over 4 shells
    # Zero-mean error: conserved quantity untouched.
    q = t.copy()
    q[:, 3, 3] += 1.0
    q[:, 6, 6] -= 1.0
    assert field_error_metrics(q, t, ndim=2).crmse == pytest.approx(0.0, abs=1e-15)
    # Interior-only error: boundary RMSE is zero.
    assert field_error_metrics(q, t, ndim=2).brmse == pytest.approx(0.0, abs=1e-15)


def test_vrmse_and_pearson_closed_forms():
    """pred = target + c (target - mean): VRMSE = |c| exactly, Pearson = 1 (c > -1)."""
    rng = np.random.default_rng(7)
    t = rng.standard_normal((3, 16, 16)) + 5.0          # large offset: nRMSE would look tiny
    for c in (0.1, 0.5, 2.0):
        p = t + c * (t - t.mean(axis=(-2, -1), keepdims=True))
        r = field_error_metrics(p, t, ndim=2)
        assert r.vrmse == pytest.approx(c, rel=1e-9)
        assert r.pearson == pytest.approx(1.0, abs=1e-9)
        assert r.nrmse < r.vrmse                         # offset hides the error in nRMSE
    # Predicting the spatial mean everywhere: VRMSE = 1, whatever the offset.
    m = np.broadcast_to(t.mean(axis=(-2, -1), keepdims=True), t.shape)
    assert field_error_metrics(m, t, ndim=2).vrmse == pytest.approx(1.0, rel=1e-9)


def test_etdrk4_matches_exponax_to_round_off():
    """Cross-validation against an independent ETDRK4 implementation
    (Exponax, MIT, JAX): Kuramoto-Sivashinsky in the Kassam-Trefethen setup,
    u_t + (u^2/2)_x + u_xx + u_xxxx = 0 on [0, 32 pi), integrated into the
    chaotic regime. With the same 2/3 dealiasing convention (keep |m| <
    int(2/3 * N/2)) the trajectories agree to round-off, so any difference
    would be in the scheme, not in the setup."""
    ex = pytest.importorskip("exponax")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    L, N, dt, T = 32 * np.pi, 128, 0.0625, 30.0
    x = L * np.arange(N) / N
    u0 = np.cos(x / 16) * (1 + np.sin(x / 16))
    k = fourier_wavenumbers(N, L)
    mask = np.abs(np.fft.fftfreq(N) * N) < int((2 / 3) * (N // 2))

    def nonlin(u_hat, t):
        u = np.fft.ifft(u_hat).real
        return -0.5j * k * np.fft.fft(u * u) * mask

    ours = np.fft.ifft(ETDRK4(k ** 2 - k ** 4, nonlin, dt=dt).solve(np.fft.fft(u0), T).u[-1]).real
    stepper = ex.stepper.KuramotoSivashinskyConservative(1, L, N, dt, order=4)
    u = jax.numpy.asarray(u0[None, :], dtype=jax.numpy.float64)
    for _ in range(int(round(T / dt))):
        u = stepper(u)
    assert np.max(np.abs(ours - np.asarray(u[0]))) < 1e-10
