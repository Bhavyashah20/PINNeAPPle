"""CalibrationMetrics against closed forms and Monte Carlo: a calibrated Gaussian model has ECE ~ 0 and coverage
1 - alpha (whatever the number of outputs); over- and under-confident models are detected."""
import math

import numpy as np
import pytest
import torch

from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics as C


def _data(n=40000, d=1, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros(n, d)
    y = torch.randn(n, d, generator=g)
    return mu, y, torch.full((n, d), scale)


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.32])
def test_calibrated_gaussian_has_nominal_coverage(d, alpha):
    # regression: with d = 3 the joint coverage of all outputs (~(1-alpha)^3 = 0.73) was returned, against the
    # documented element-wise average
    mu, y, s = _data(d=d)
    assert C.coverage_at_level(mu, y, s, alpha=alpha) == pytest.approx(1 - alpha, abs=0.006)


def test_ece_zero_when_calibrated_and_large_when_not():
    mu, y, s = _data()
    assert C.expected_calibration_error(mu, y, s) < 0.01
    # std halved: the PIT of z = 2 eps is 2 Phi(...) shaped; its distance to uniform is about 0.1
    over = C.expected_calibration_error(mu, y, 0.5 * s)
    under = C.expected_calibration_error(mu, y, 2.0 * s)
    assert over > 0.08 and under > 0.08


def test_ece_closed_form_for_scaled_std():
    # with y ~ N(0, 1) and a predicted std k, the empirical level at nominal p is Phi(k Phi^-1(p)); the ECE is the
    # mean |Phi(k Phi^-1(p)) - p| over the n_bins levels
    from scipy.stats import norm
    mu, y, s = _data(n=200000, seed=3)
    k, n_bins = 0.5, 15
    levels = np.linspace(0, 1, n_bins + 2)[1:-1]
    expected = float(np.mean(np.abs(norm.cdf(k * norm.ppf(levels)) - levels)))
    assert C.expected_calibration_error(mu, y, k * s, n_bins=n_bins) == pytest.approx(expected, abs=0.003)


def test_nll_closed_form():
    # E[NLL] of a calibrated unit Gaussian = 0.5 log(2 pi) + 0.5
    mu, y, s = _data(n=200000, seed=1)
    assert C.nll_gaussian(mu, y, s) == pytest.approx(0.5 * math.log(2 * math.pi) + 0.5, abs=0.01)


def test_numpy_inputs_and_plot_data():
    mu, y, s = (t.numpy() for t in _data(n=5000))
    assert C.coverage_at_level(mu, y, s, alpha=0.1) == pytest.approx(0.9, abs=0.02)
    d = C.calibration_plot_data(mu, y, s, n_bins=9)
    assert len(d["expected"]) == len(d["observed"]) == 9
    assert d["ece"][0] == pytest.approx(C.expected_calibration_error(mu, y, s, n_bins=9))
    assert C.sharpness(s) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [C.expected_calibration_error, C.coverage_at_level, C.nll_gaussian,
                                C.calibration_plot_data])
def test_non_finite_and_shape_errors(fn):
    mu, y, s = _data(n=50)
    y[3, 0] = float("nan")                     # used to count as "outside every interval", silently
    with pytest.raises(ValueError, match="NaN"):
        fn(mu, y, s)
    with pytest.raises(ValueError, match="shape"):
        fn(mu[:10], y[:10, 0], s[:10])
    with pytest.raises(ValueError, match="non-negative"):
        fn(mu, torch.zeros_like(mu), -s)
