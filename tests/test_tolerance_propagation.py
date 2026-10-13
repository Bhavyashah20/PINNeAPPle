"""Tests of pinneapple_analysis.uncertainty.tolerance against closed forms and scipy."""
import numpy as np
import pytest
from scipy import stats

from pinneapple_analysis.uncertainty.tolerance import propagate, wilson_interval

A = np.array([2.0, -1.0, 0.5])
MU = np.array([1.0, 3.0, -2.0])
SIG = np.array([0.1, 0.2, 0.4])


def test_linear_normal_mean_std():
    n = 20000
    res = propagate(lambda X: X @ A, MU, SIG, n=n)
    std = np.sqrt(np.sum(A ** 2 * SIG ** 2))
    assert res.mean[0] == pytest.approx(A @ MU, abs=4 * std / np.sqrt(n))
    assert res.std[0] == pytest.approx(std, rel=4 / np.sqrt(2 * n))
    assert res.percentiles[50.0][0] == pytest.approx(A @ MU, abs=0.02)


def test_probability_below_limit_against_normal_cdf():
    n = 20000
    std = np.sqrt(np.sum(A ** 2 * SIG ** 2))
    limit = A @ MU + 1.2 * std
    res = propagate(lambda X: X @ A, MU, SIG, n=n, limits=(None, limit))
    p = stats.norm.cdf(1.2)
    assert res.p_within == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / n))
    lo, hi = res.p_within_ci
    assert lo < p < hi


def test_uniform_variance():
    half = np.array([0.3, 1.0])
    n = 40000
    res = propagate(lambda X: X, [0.0, 5.0], half, n=n, dist="uniform")
    var = (2 * half) ** 2 / 12
    assert res.std ** 2 == pytest.approx(var, rel=0.03)
    assert res.inputs.min(0)[1] >= 4.0 and res.inputs.max(0)[1] <= 6.0


def test_mixed_dist_and_joint_limits():
    res = propagate(lambda X: X, [0.0, 0.0], [1.0, 1.0], n=20000, dist=["normal", "uniform"],
                    limits=([-1.0, -0.5], [1.0, 0.5]))
    p0, p1 = stats.norm.cdf(1) - stats.norm.cdf(-1), 0.5
    assert res.p_within_each == pytest.approx([p0, p1], abs=0.02)
    assert res.p_within == pytest.approx(p0 * p1, abs=0.02)   # independent inputs


@pytest.mark.parametrize("k,n", [(0, 50), (7, 50), (950, 1000), (1000, 1000)])
def test_wilson_matches_scipy(k, n):
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert wilson_interval(k, n, 0.95) == pytest.approx((ref.low, ref.high), abs=1e-12)


def test_bad_inputs():
    with pytest.raises(ValueError):
        propagate(lambda X: X, [0.0], [1.0], dist="triangular")
    with pytest.raises(ValueError):
        propagate(lambda X: X, [0.0], [-1.0])
