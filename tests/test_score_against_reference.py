"""CalibrationSummary.from_predictions and score_against_reference on a tiny synthetic case."""
import numpy as np
import pytest
import torch

from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
from pinneapple_analysis.verification.physics_confidence_score import (
    CalibrationSummary,
    compute_physics_confidence,
    score_against_reference,
)


def _case():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.repeat(np.arange(4), 25), np.tile(np.arange(25), 4)]).astype(np.float32)
    y = np.column_stack([np.sin(x[:, 1] / 4), np.cos(x[:, 1] / 4)]).astype(np.float32)
    good = y + 0.01 * rng.normal(size=y.shape).astype(np.float32)
    bad = y + 0.5 * rng.normal(size=y.shape).astype(np.float32)
    return x, y, good, bad


def test_calibration_summary_matches_calibration_metrics():
    rng = np.random.default_rng(1)
    t = rng.normal(size=500)
    p = t + rng.normal(size=500)
    s = np.full(500, 1.0)
    c = CalibrationSummary.from_predictions(p, t, s, level=0.9)
    tp, tt, ts = (torch.tensor(a) for a in (p, t, s))
    assert c.ece == CalibrationMetrics.expected_calibration_error(tp, tt, ts)
    assert c.coverage == CalibrationMetrics.coverage_at_level(tp, tt, ts, alpha=0.1)
    assert c.target_coverage == 0.9 and c.sharpness == pytest.approx(1.0)
    # residuals are N(0, 1) and sigma = 1: the 90 % interval covers about 90 % (binomial sd 1.3 %)
    assert abs(c.coverage - 0.9) < 0.05


def test_score_against_reference_matches_manual_scaffold():
    x, y, good, bad = _case()
    sig = np.full_like(good, 0.01)
    out = score_against_reference("tiny_synthetic_ref", x, y, {"good": good, "bad": bad}, x_vars=("run", "frame"),
                                  y_vars=("u", "v"), sigma={"good": sig}, not_run={"geometry_ood": "no geometry"})
    assert out["not_run"] == {"geometry_ood": "no geometry"}
    for name, p in (("good", good), ("bad", bad)):
        rel = np.sqrt(np.mean((p - y).astype(np.float64) ** 2)) / np.sqrt(np.mean(y.astype(np.float64) ** 2))
        assert out[name]["rel_l2"] == pytest.approx(rel, rel=1e-6)
        assert out[name]["n_points"] == len(x)
    assert out["good"]["overall_score"] > out["bad"]["overall_score"]
    assert out["good"]["coverage"] == pytest.approx(2 / 5) and out["bad"]["coverage"] == pytest.approx(1 / 5)
    assert out["good"]["calibration"]["target_coverage"] == 0.9 and "calibration" not in out["bad"]
    # same numbers as the hand-written scaffold: compute_physics_confidence on the same calibration + benchmark
    from types import SimpleNamespace
    cal = CalibrationSummary.from_predictions(good, y, sig)
    bench = SimpleNamespace(relative_l2_error=out["good"]["rel_l2"], rmse=out["good"]["rmse"], benchmark="b",
                            n_compared_points=len(x))
    manual = compute_physics_confidence(calibration_metrics=cal, benchmark_comparison=bench)
    assert out["good"]["overall_score"] == pytest.approx(manual.overall_score)


def test_non_finite_prediction_is_not_scored():
    x, y, good, _ = _case()
    good = good.copy()
    good[3, 0] = np.nan
    out = score_against_reference("tiny_synthetic_ref_nan", x, y, {"m": good}, x_vars=("run", "frame"),
                                  y_vars=("u", "v"))
    assert out["m"]["overall_score"] is None and out["m"]["coverage"] == 0.0 and "note" in out["m"]


def test_unknown_model_in_per_model_dict_raises():
    x, y, good, _ = _case()
    with pytest.raises(ValueError, match="unknown model"):
        score_against_reference("tiny_synthetic_ref_bad", x, y, {"m": good}, x_vars=("run", "frame"),
                                y_vars=("u", "v"), sigma={"other": good})
