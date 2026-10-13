"""Resampling CIs (pinneapple_analysis.uncertainty.resampling) against closed-form references."""
import numpy as np
import pytest

from pinneapple_analysis.uncertainty.resampling import (
    block_bootstrap,
    group_bootstrap,
    jackknife,
    skill_score,
)


def test_jackknife_of_mean_is_standard_error():
    # delete-one jackknife of the mean equals s / sqrt(n) exactly (Efron & Tibshirani 1993, ch. 11)
    x = np.random.default_rng(1).normal(size=200)
    est, se = jackknife(x, np.mean, n_blocks=len(x))
    assert est == pytest.approx(x.mean())
    assert se == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)), rel=1e-10)


def test_jackknife_array_valued():
    x = np.random.default_rng(2).normal(size=(120, 3))
    est, se = jackknife(x, lambda a: a.mean(0), n_blocks=120)
    np.testing.assert_allclose(est, x.mean(0))
    np.testing.assert_allclose(se, x.std(0, ddof=1) / np.sqrt(120), rtol=1e-10)


@pytest.mark.parametrize("block", [1, 20])
def test_block_bootstrap_iid_mean_width(block):
    n_obs = 4000
    x = np.random.default_rng(3).normal(size=n_obs)
    est, lo, hi = block_bootstrap(x, np.mean, block=block, n=1000, seed=0)
    assert est == pytest.approx(x.mean())
    assert lo < 0 < hi
    # normal-theory 95 % interval of the mean of N(0, 1): width 2 * 1.96 / sqrt(N)
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(n_obs), rel=0.15)


def test_block_bootstrap_ar1_wider_than_iid():
    rng = np.random.default_rng(4)
    phi, n_obs = 0.8, 4000
    e = rng.normal(size=n_obs)
    x = np.empty(n_obs)
    x[0] = e[0]
    for k in range(1, n_obs):
        x[k] = phi * x[k - 1] + e[k]
    _, lo1, hi1 = block_bootstrap(x, block=1, n=1000)
    _, lob, hib = block_bootstrap(x, block=100, n=1000)
    # long-run variance factor of AR(1) is (1 + phi) / (1 - phi) = 9, so the width ratio is about 3
    ratio = (hib - lob) / (hi1 - lo1)
    assert 2.0 < ratio < 4.0


def test_block_bootstrap_drops_nonfinite():
    x = np.array([1.0, np.nan, 3.0, np.inf])
    est, lo, hi = block_bootstrap(x, block=1, n=200)
    assert est == 2.0 and 1.0 <= lo <= hi <= 3.0


def test_group_bootstrap_singletons_equal_iid_bootstrap():
    x = np.random.default_rng(5).normal(size=300)
    a = group_bootstrap(x, np.arange(300), np.mean, n=500, seed=7)
    b = block_bootstrap(x, np.mean, block=1, n=500, seed=7)
    assert a == pytest.approx(b, rel=1e-12)


def test_group_bootstrap_wider_for_correlated_groups():
    rng = np.random.default_rng(6)
    g = np.repeat(np.arange(40), 25)
    x = rng.normal(size=40)[g] + 0.1 * rng.normal(size=g.size)    # the cases of a group share one shock
    _, lo_g, hi_g = group_bootstrap(x, g, n=1000)
    _, lo_i, hi_i = block_bootstrap(x, block=1, n=1000)
    assert (hi_g - lo_g) > 3 * (hi_i - lo_i)                      # design effect about sqrt(25) = 5


def test_skill_score_reference_values():
    e = np.abs(np.random.default_rng(8).normal(size=500))
    for kw in ({}, {"block": 20}, {"groups": np.arange(500) // 10}):
        assert skill_score(e, e, **kw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
        assert skill_score(0.5 * e, e, **kw) == pytest.approx((0.5, 0.5, 0.5), abs=1e-12)
    s, lo, hi = skill_score(e, 1.0 + e, block=10)
    assert lo < s < hi


def test_skill_score_rejects_block_and_groups():
    with pytest.raises(ValueError):
        skill_score([1.0], [1.0], block=2, groups=[0])
