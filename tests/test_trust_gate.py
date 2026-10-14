"""Tests for pinneapple_analysis.trust.TrustGate."""
from __future__ import annotations

import pytest
import torch

from pinneapple_analysis import TrustGate
from pinneapple_systems.component_library import ComponentRegistry
from pinneapple_systems.component_modeling.physics_residuals import heat_conduction_residual


def test_fit_and_ood_score_in_distribution_vs_far_out():
    gate = TrustGate()
    training_coords = torch.rand(200, 2)  # roughly Uniform[0,1]^2
    gate.fit(training_coords)

    in_dist = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    far_out = gate.score(model=None, x=torch.tensor([[50.0, 50.0]]))

    assert 0.0 <= in_dist.ood_score <= 1.0
    assert 0.0 <= far_out.ood_score <= 1.0
    assert in_dist.ood_score > far_out.ood_score


def test_fit_from_bounds():
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    result = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    assert result.ood_score is not None


def test_score_without_any_evidence_raises():
    gate = TrustGate()
    with pytest.raises(ValueError):
        gate.score(model=None, x=torch.rand(1, 2))


def test_residual_score_low_for_trained_model_high_error_for_random():
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(16, 16))
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    x = torch.rand(16, 2)
    result = gate.score(model=model, x=x, residual_fn=heat_conduction_residual, residual_kwargs={"k": 1.0, "Q": 0.0})
    assert result.residual_score is not None
    assert 0.0 < result.combined <= 1.0


def test_ensemble_score():
    class FakeEnsemble:
        def predict(self, x):
            mean = torch.full((x.shape[0], 1), 10.0)
            std = torch.full((x.shape[0], 1), 0.01)  # 0.1% relative std -> high confidence
            return mean, std

    gate = TrustGate()
    result = gate.score(model=None, x=torch.rand(4, 2), ensemble=FakeEnsemble())
    assert result.ensemble_score is not None
    assert result.ensemble_score > 0.9  # tiny std -> high confidence


def test_ood_batch_of_far_points_is_not_averaged_into_the_training_mean():
    # regression: the p-value of the batch MEAN was scored, so two points at -50 and +50 sigma averaged to the
    # training mean and scored 0.9998 ("perfectly in distribution")
    torch.manual_seed(0)
    gate = TrustGate().fit(torch.randn(4000, 2))
    far = gate.score(model=None, x=torch.tensor([[-50.0, 0.0], [50.0, 0.0]]))
    assert far.ood_score == 0.0
    half = gate.score(model=None, x=torch.cat([torch.randn(50, 2), torch.full((50, 2), 40.0)]))
    assert 0.4 <= half.ood_score <= 0.55
    assert half.details["ood_p_value_min"] < 1e-100


def test_ood_in_distribution_batch_matches_the_nominal_level():
    # points drawn from the training distribution fall inside the 99 % region about 99 % of the time
    torch.manual_seed(1)
    gate = TrustGate(ood_level=0.01).fit(torch.randn(20000, 3))
    frac = gate.score(model=None, x=torch.randn(20000, 3)).ood_score
    assert abs(frac - 0.99) < 0.005


def test_ood_nan_coordinates_and_wrong_dimension():
    gate = TrustGate().fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    assert gate.score(model=None, x=torch.tensor([[float("nan"), 0.5]])).ood_score == 0.0
    with pytest.raises(ValueError, match="coordinates"):
        gate.score(model=None, x=torch.rand(4, 3))
