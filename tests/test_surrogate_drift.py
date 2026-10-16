"""Tests of pinneapple_analysis.uncertainty.drift against constructed cases and scipy references."""
import numpy as np
import pytest
from scipy import stats

from pinneapple_analysis.uncertainty.drift import DriftThresholds, assess_drift


def _batch(n=2000, d=3, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) + shift
    y = X.sum(1)
    pred = y + 0.1 * rng.normal(size=n)
    return X, y, pred


def test_no_drift_on_iid_data():
    X_ref, _, _ = _batch(seed=1)
    X_new, y, pred = _batch(seed=2)
    rep = assess_drift(y, pred, sigma=np.full_like(pred, 0.1), X_ref=X_ref, X_new=X_new)
    assert np.all(rep.ks_pvalue > 0.01)
    assert rep.decision == "ok", rep.reasons
    assert rep.nrmse[0] == pytest.approx(0.1 / np.std(y), rel=0.1)


def test_mean_shift_detected():
    X_ref, _, _ = _batch(seed=1)
    X_new, y, pred = _batch(shift=0.3, seed=2)
    rep = assess_drift(y, pred, X_ref=X_ref, X_new=X_new)
    assert np.all(rep.ks_pvalue < 1e-6)
    assert rep.decision in ("review", "retrain")
    assert any("input shift" in r for r in rep.reasons)
    # KS statistic is scipy's
    assert rep.ks_stat[0] == pytest.approx(stats.ks_2samp(X_ref[:, 0], X_new[:, 0]).statistic)


def test_prediction_bias_and_trend_detected():
    n = 500
    rng = np.random.default_rng(3)
    y = rng.normal(size=n)
    pred = y + 0.05 * rng.normal(size=n) + np.linspace(0.0, 0.3, n)   # slowly growing error
    rep = assess_drift(y, pred)
    assert rep.trend_pvalue[0] < 1e-6
    assert rep.trend_tau[0] == pytest.approx(stats.kendalltau(np.arange(n), pred - y).statistic)
    assert any("bias" in r for r in rep.reasons) and any("trend" in r for r in rep.reasons)


def test_out_of_range_fraction_exact():
    X_new = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.1], [0.2, 0.9], [1.0, 1.0]])
    y = np.zeros(5)
    rep = assess_drift(y, y + 1e-3 * np.arange(5), X_new=X_new, bounds=(np.zeros(2), np.ones(2)))
    assert rep.out_of_bounds == pytest.approx(2 / 5)
    rep2 = assess_drift(y, y + 1e-3 * np.arange(5), X_new=X_new, bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert rep2.out_of_bounds == pytest.approx(2 / 5)
    assert rep.decision == "retrain"


def test_coverage_of_exact_gaussian_sigma():
    n, level = 20000, 0.9
    rng = np.random.default_rng(4)
    sigma = rng.uniform(0.5, 2.0, size=n)
    y = rng.normal(size=n)
    pred = y + sigma * rng.normal(size=n)
    rep = assess_drift(y, pred, sigma=sigma, level=level)
    assert abs(rep.coverage[0] - level) < 4 * np.sqrt(level * (1 - level) / n)
    # an overconfident sigma is flagged
    rep_bad = assess_drift(y, pred, sigma=0.5 * sigma, level=level)
    assert any("coverage" in r for r in rep_bad.reasons)


def test_multi_output_and_thresholds():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(300, 2))
    pred = y + np.array([0.0, 1.0]) * 0.4 * rng.normal(size=(300, 2))
    rep = assess_drift(y, pred, thresholds=DriftThresholds(nrmse_review=0.2, nrmse_retrain=0.5))
    assert rep.nrmse.shape == (2,) and rep.nrmse[0] == 0.0
    assert rep.decision == "review"
    assert "decision: review" in rep.summary()
