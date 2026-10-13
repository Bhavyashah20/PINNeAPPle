"""heat_fv_reference.footprint_area against the closed-form integral of the smooth footprint."""
import math

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers import heat_fv_reference as fv


def _exact_area(half: float, width: float) -> float:
    # 0.5 (1 + tanh(s / w)) is the logistic function of 2 s / w, so the integral over the whole line of
    # 0.5 (1 + tanh((a - |x|) / w)) is w ln(1 + exp(2 a / w)) = 2 a + w ln(1 + exp(-2 a / w)); the footprint is
    # the product of two such factors.
    side = 2 * half + width * math.log1p(math.exp(-2 * half / width))
    return side * side


@pytest.mark.parametrize("half,width", [(0.010, 0.002), (0.005, 0.001), (0.006, 0.001)])
def test_footprint_area_matches_closed_form(half, width):
    area = fv.footprint_area(0.03, 0.03, half, width, 0.1, 0.06)       # footprint far from the plate edges
    assert area == pytest.approx(_exact_area(half, width), rel=1e-6)


def test_private_alias_and_public_api():
    assert fv._footprint_area is fv.footprint_area
    assert {"footprint_area", "interp_grid", "convergence"} <= set(fv.__all__)


def test_plate_power_balance():
    # steady state: all the device power leaves through the two convective faces
    h, t_air, power = 25.0, 25.0, 8.0
    x, y, T = fv.plate_2d(h, lx=0.1, ly=0.06, t=0.001, k=200.0, power=power, t_air=t_air,
                          device=(0.03, 0.03, 0.01, 0.002), nx=50, ny=30)
    dx, dy = 0.1 / 50, 0.06 / 30
    dev = (0.03, 0.03, 0.01, 0.002)
    q_in = power / fv.footprint_area(*dev, 0.1, 0.06) * np.sum(fv.footprint(x[None, :], y[:, None], *dev)) * dx * dy
    q_out = 2 * h * np.sum(T - t_air) * dx * dy
    assert q_out == pytest.approx(q_in, rel=1e-9)        # discrete source of the 50 x 30 grid
    assert q_in == pytest.approx(power, rel=1e-4)         # which is the device power up to the grid resolution
