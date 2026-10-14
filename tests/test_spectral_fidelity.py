"""Spectral fidelity metrics (#702): references are an analytical power-law spectrum, closed-form structure functions
and derivatives, and a decaying 2-D turbulence field from the pseudo-spectral solver of ``pinneapple_physics``."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.validation.spectral import (energy_spectrum, fit_power_law, gradient_metrics,
                                                     high_wavenumber_bias, spectral_comparison, spectral_fidelity,
                                                     structure_functions)


def gaussian_blur(u: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian filter of width ``sigma`` grid cells (exact transfer function)."""
    k = [np.fft.fftfreq(n) * 2 * np.pi for n in u.shape]
    K2 = sum(g ** 2 for g in np.meshgrid(*k, indexing="ij"))
    return np.fft.ifftn(np.fft.fftn(u) * np.exp(-0.5 * sigma ** 2 * K2)).real


def power_law_field(n: int, alpha: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Real periodic field whose shell-summed spectrum is exactly ``k^-alpha`` (random phases)."""
    rng = np.random.default_rng(seed)
    m = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(m, m, indexing="ij")
    K = np.rint(np.hypot(KX, KY)).astype(int)
    phase = np.fft.fft2(rng.standard_normal((n, n)))
    phase /= np.abs(phase)                                    # Hermitian, unit modulus: the field stays real
    E = np.zeros(K.max() + 1)
    E[1:] = np.arange(1, K.max() + 1, dtype=float) ** -alpha
    count = np.bincount(K.ravel())
    amp = np.sqrt(E[K] / count[K])
    return np.fft.ifft2(amp * phase * n * n).real, E


@pytest.fixture(scope="module")
def turbulence():
    """Velocity component of 2-D decaying turbulence (Re ~ 1e3) after an eddy turnover or two."""
    from pinneapple_physics.multiscale.ns2d import VorticityNS2D, random_vorticity, velocity_from_vorticity
    n = 64
    w0 = random_vorticity(np.random.default_rng(0), n, k0=6.0)
    w = VorticityNS2D(n, 5e-4).simulate(w0, [1.5], dt=1e-2)[0]
    return velocity_from_vorticity(w)[0], w


def test_spectrum_matches_a_prescribed_power_law():
    u, E_true = power_law_field(128, 5 / 3)
    k, E = energy_spectrum(u, 2)
    rel = np.abs(E[1:] / E_true[1:len(E)] - 1)
    assert rel.max() < 1e-2                                   # acceptance: < 1 %
    slope, _ = fit_power_law(k, E, 2, 60)
    assert slope == pytest.approx(-5 / 3, rel=1e-2)
    assert E.sum() == pytest.approx(np.mean(u ** 2), rel=1e-2)   # Parseval (shells beyond Nyquist are tiny)


def test_parseval_and_window_on_a_non_periodic_field():
    x = np.linspace(0, 1, 128)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = np.sin(2 * np.pi * 6 * X) + 0.3 * X                  # not periodic: a ramp and a sine
    k, E = energy_spectrum(f, 2, periodic=False)
    assert np.argmax(E[1:]) + 1 == 6                         # the window keeps the peak where it is
    k2, E2 = energy_spectrum(f, 2, periodic=False, window=None)
    assert E[1:].sum() == pytest.approx(np.var(f), rel=0.05)  # unit mean-square window: energy kept on average
    assert E2[1:].sum() == pytest.approx(np.var(f), rel=1e-6)


def test_reference_against_itself_has_zero_bias(turbulence):
    u, _ = turbulence
    r = spectral_fidelity(u, u, 2)
    assert r.hf_bias == 0.0 and r.passed
    assert all(v == 0.0 for v in r.spectrum.band_phase_error.values())
    assert all(v == pytest.approx(1.0) for v in r.spectrum.band_coherence.values())


def test_blurred_turbulence_has_small_l2_error_but_is_flagged(turbulence):
    u, _ = turbulence
    blurred = gaussian_blur(u, 1.0)
    r = spectral_fidelity(blurred, u, 2)
    assert r.rel_l2 < 0.05                                    # looks good by L2 ...
    assert r.hf_bias < -0.5 and not r.passed                  # ... but most of the small-scale energy is gone
    assert r.spectrum.band_ratio["low"] > 0.95
    assert r.structure_pred.S[2][0] < r.structure_ref.S[2][0]
    noisy = u + 0.02 * u.std() * np.random.default_rng(1).standard_normal(u.shape)
    assert high_wavenumber_bias(noisy, u, 2) > 0.25           # added noise: positive bias


def test_phase_error_of_a_shifted_field():
    u, _ = power_law_field(64, 2.0, seed=3)
    shifted = np.roll(u, 1, axis=0)                           # mode (m, n) gains phase 2 pi m / 64
    c = spectral_comparison(shifted, u, 2)
    assert c.band_ratio["high"] == pytest.approx(1.0)        # same energy everywhere ...
    assert c.band_phase_error["low"] < c.band_phase_error["high"]   # ... phase error grows with k
    assert c.band_coherence["high"] < 0.9


def test_bands_follow_the_resolved_range():
    u, _ = power_law_field(64, 3.0)
    m = np.fft.fftfreq(64) * 64
    KX, KY = np.meshgrid(m, m, indexing="ij")
    dealiased = np.fft.ifft2(np.fft.fft2(u) * ((np.abs(KX) <= 21) & (np.abs(KY) <= 21))).real
    c = spectral_comparison(dealiased, dealiased, 2)
    assert c.bands["high"][1] <= 30 and np.isfinite(c.hf_bias) and c.hf_bias == 0.0


def test_structure_functions_closed_form_and_gaussian_flatness():
    x = np.arange(64, dtype=float)
    u = np.broadcast_to(2.0 * x[:, None], (64, 64)).copy()   # u = a x: S2(r) = a^2 r^2 / 2 (axis average)
    sf = structure_functions(u, 2, periodic=False, lags=[1, 2, 4])
    assert np.allclose(sf.S[2], 4.0 * np.array([1, 4, 16]) / 2)
    g = np.random.default_rng(0).standard_normal((256, 256))
    sfg = structure_functions(g, 2, orders=(2, 3))
    assert np.allclose(sfg.flatness, 3.0, atol=0.05)          # Gaussian increments
    assert abs(sfg.S3_signed[0]) < 0.05


def test_gradient_metrics_against_analytic_derivatives():
    n = 64
    h = 2 * np.pi / n
    x = np.arange(n) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    ref = np.stack([-np.sin(Y), np.sin(X)])                   # vorticity cos(x) + cos(y), strain from the shear
    assert gradient_metrics(ref, ref, 2, h, periodic=True, vector_axis=0) == {
        "grad_rel_err": 0.0, "vorticity_rel_err": 0.0, "strain_rate_rel_err": 0.0}
    g = gradient_metrics(0.9 * ref, ref, 2, h, periodic=True, vector_axis=0)
    assert g["vorticity_rel_err"] == pytest.approx(0.1) and g["strain_rate_rel_err"] == pytest.approx(0.1)
    xs = np.linspace(0, 1, 41)
    Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
    T = Xs ** 2 + Ys                                          # wall flux -dT/dn at x = 1 is -2 (outward +x)
    Tp = Xs ** 2 * 1.05 + Ys
    out = gradient_metrics(Tp, T, 2, xs[1] - xs[0], walls=[(0, -1)])
    assert out["wall_flux_rel_err[0,-1]"] == pytest.approx(0.05, rel=1e-6)
    assert out["grad_mag_rel_err"] > 0


def test_report_serialises():
    u, _ = power_law_field(32, 2.0)
    d = spectral_fidelity(u, u, 2).to_dict()
    assert d["passed"] and d["checks"][0]["id"] == "spectral.hf_bias"
    import json
    json.dumps(d)
