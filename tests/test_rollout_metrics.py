"""pinneapple_physics.metrics.rollout_relative_l2, checked against hand-computed values."""
import numpy as np
import pytest

from pinneapple_physics.metrics import rollout_relative_l2


def test_per_step_mean_by_hand():
    true = np.array([[3.0, 4.0], [0.0, 2.0]])        # step norms 5 and 2
    pred = np.array([[3.0, 4.0 + 1.0], [0.0, 3.0]])  # step errors 1 and 1
    assert rollout_relative_l2(pred, true) == pytest.approx((1 / 5 + 1 / 2) / 2)
    # whole trajectory: sqrt(1 + 1) / sqrt(25 + 4)
    assert rollout_relative_l2(pred, true, aggregate="trajectory") == pytest.approx(np.sqrt(2 / 29))


def test_fields_pooled_and_zero_steps_skipped():
    true = np.zeros((3, 2, 2))
    true[1:] = 1.0                                    # step 0 has zero norm: skipped
    pred = true * 1.1
    assert rollout_relative_l2(pred, true) == pytest.approx(0.1)
    assert np.isnan(rollout_relative_l2(np.ones((2, 2)), np.zeros((2, 2))))


def test_shape_and_option_errors():
    with pytest.raises(ValueError):
        rollout_relative_l2(np.ones((2, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        rollout_relative_l2(np.ones((2, 2)), np.ones((2, 2)), aggregate="max")
