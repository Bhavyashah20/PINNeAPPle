"""CRPS against independent references: direct numerical integration of
(F(x) - 1{x >= y})^2, the O(M^2) pairwise definition, and the deterministic limit."""
import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from pinneapple_analysis.uncertainty.posterior_metrics import crps_ensemble, crps_gaussian


@pytest.mark.parametrize("mu,sigma,y", [(0.0, 1.0, 0.3), (2.0, 0.5, -1.0), (-1.0, 3.0, -1.0)])
def test_gaussian_closed_form_matches_integral_definition(mu, sigma, y):
    f = lambda x: (norm.cdf(x, mu, sigma) - (x >= y)) ** 2  # noqa: E731
    ref = integrate.quad(f, -np.inf, y, limit=200)[0] + integrate.quad(f, y, np.inf, limit=200)[0]
    assert crps_gaussian(mu, sigma, y) == pytest.approx(ref, rel=1e-8)


def test_deterministic_limit():
    assert crps_gaussian(1.0, 0.0, 3.5) == pytest.approx(2.5)
    assert crps_ensemble(np.full((5, 1), 1.0), np.array([3.5]))[0] == pytest.approx(2.5)


def test_ensemble_sorted_identity_equals_pairwise_definition():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((17, 4, 3)), rng.standard_normal((4, 3))
    pair = np.abs(x[:, None] - x[None]).sum(axis=(0, 1))
    ref = np.abs(x - y).mean(0) - 0.5 * pair / 17 ** 2
    ref_fair = np.abs(x - y).mean(0) - 0.5 * pair / (17 * 16)
    np.testing.assert_allclose(crps_ensemble(x, y), ref, rtol=1e-12)
    np.testing.assert_allclose(crps_ensemble(x, y, fair=True), ref_fair, rtol=1e-12)


def test_large_gaussian_ensemble_converges_to_closed_form():
    rng = np.random.default_rng(1)
    mu, sigma, y = 0.7, 1.3, -0.2
    x = mu + sigma * rng.standard_normal((200_000, 1))
    assert crps_ensemble(x, np.array([y]), fair=True)[0] == pytest.approx(crps_gaussian(mu, sigma, y), rel=1e-2)
