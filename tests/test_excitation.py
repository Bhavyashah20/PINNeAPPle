"""pinneapple_systems.excitation: actuator excitation signals."""
import numpy as np
import pytest

from pinneapple_systems.excitation import (
    SHAPES,
    actuator_series,
    rate_limit,
    shape_chirp,
    shape_prbs,
)


def test_chirp_matches_scipy_linear_chirp():
    """Linear chirp f(tau) = f0 + (f1 - f0) tau: same signal as scipy.signal.chirp (phi = -90 deg turns cos into sin)."""
    signal = pytest.importorskip("scipy.signal")
    tau = np.linspace(0.0, 1.0, 2001)
    for f0, f1 in [(0.5, 6.0), (1.0, 9.0), (3.0, 0.5)]:
        ref = 0.5 + 0.5 * signal.chirp(tau, f0=f0, t1=1.0, f1=f1, method="linear", phi=-90)
        assert np.allclose(shape_chirp(tau, f0=f0, f1=f1), ref, atol=1e-12)


def test_chirp_instantaneous_frequency():
    """Instantaneous frequency from the analytic-signal phase (scipy.signal.hilbert) follows f0 + (f1 - f0) tau."""
    signal = pytest.importorskip("scipy.signal")
    n, f0, f1 = 20001, 20.0, 60.0
    tau = np.linspace(0.0, 1.0, n)
    phase = np.unwrap(np.angle(signal.hilbert(shape_chirp(tau, f0=f0, f1=f1) - 0.5)))
    f_inst = np.gradient(phase, tau) / (2 * np.pi)
    core = slice(n // 10, -n // 10)                     # Hilbert transform edge effects
    assert np.allclose(f_inst[core], (f0 + (f1 - f0) * tau)[core], rtol=0.02)


def test_prbs_two_levels_and_switches_only_on_clock():
    n, dwell, lo, hi = 1000, 0.05, -2.0, 3.0
    u = actuator_series("prbs", n, lo=lo, hi=hi, seed=7, dwell=dwell)
    assert set(np.unique(u)) == {lo, hi}
    hold = int(round(dwell * n))
    switches = np.flatnonzero(np.diff(u)) + 1           # first sample of each new level
    assert switches.size > 0 and np.all(switches % hold == 0)


def test_rate_limit_bounds_slew_and_stays_in_range():
    x = actuator_series("step", 400, lo=0.0, hi=10.0, t0=0.25)
    tau = 20.0
    y = rate_limit(x, tau)
    dy = np.abs(np.diff(y))
    assert np.all(dy <= np.abs(x[1:] - y[:-1]) / tau + 1e-12)          # per-sample change <= gap / tau
    assert dy.max() <= 10.0 / tau + 1e-12 and y.min() >= 0.0 and y.max() <= 10.0
    i0 = int(np.argmax(x > 0))                                         # first sample of the step
    k = np.arange(1, 400 - i0 + 1)                                     # discrete step response 10 (1 - (1 - 1/tau)^k)
    assert np.allclose(y[i0:], 10.0 * (1.0 - (1.0 - 1.0 / tau) ** k)) and np.all(y[:i0] == 0.0)
    assert rate_limit(x, 1) is x


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_shape_deterministic_with_seed_and_in_range(name):
    a = actuator_series(name, 300, lo=1.0, hi=2.0, seed=11)
    b = actuator_series(name, 300, lo=1.0, hi=2.0, seed=11)
    assert np.array_equal(a, b) and a.shape == (300,)
    assert a.min() >= 1.0 - 1e-12 and a.max() <= 2.0 + 1e-12


def test_random_shapes_change_with_seed():
    for name in ("piecewise_constant", "prbs", "smoothed_random_walk", "gaussian_bumps"):
        assert not np.array_equal(actuator_series(name, 300, seed=1), actuator_series(name, 300, seed=2)), name
    tau = np.linspace(0, 1, 50)
    assert np.array_equal(shape_prbs(tau, rng=np.random.default_rng(4)), shape_prbs(tau, rng=np.random.default_rng(4)))
