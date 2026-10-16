"""Hess-Smith panel method (pinneapple_design.aero.panel) against thin-airfoil theory and its own consistency."""
import math

import numpy as np
import pytest

from pinneapple_design.aero.geometry import REFERENCE
from pinneapple_design.aero.panel import airfoil_nodes, panel_solve


def test_symmetric_airfoil_at_zero_incidence_has_no_lift():
    r = panel_solve(REFERENCE["NACA 0012"], 0.0)
    assert abs(r.cl_kutta_joukowski) < 1e-6 and abs(r.cm_quarter) < 1e-6
    assert np.allclose(r.cp, r.cp[::-1], atol=1e-6)


def test_lift_slope_with_thickness_correction_and_kutta_joukowski_equals_pressure_lift():
    r = panel_solve(REFERENCE["NACA 0012"], 5.0)
    theory = 2 * math.pi * math.sin(math.radians(5.0)) * (1 + 0.77 * 0.12)
    assert r.cl_kutta_joukowski == pytest.approx(theory, rel=0.01)
    assert r.cl_pressure == pytest.approx(r.cl_kutta_joukowski, rel=0.01)
    assert abs(r.cd_pressure) < 1e-3                                     # d'Alembert


def test_cambered_airfoil_zero_lift_angle_and_moment():
    r0 = panel_solve(REFERENCE["NACA 2412"], 0.0)
    r4 = panel_solve(REFERENCE["NACA 2412"], 4.0)
    slope = (r4.cl_kutta_joukowski - r0.cl_kutta_joukowski) / 4.0
    alpha0 = -r0.cl_kutta_joukowski / slope
    assert alpha0 == pytest.approx(-2.1, abs=0.3)                        # thin-airfoil theory: -2.08 deg
    assert r4.cm_quarter == pytest.approx(-0.053, abs=0.01)


def test_flow_is_tangent_to_the_surface_and_uniform_far_away():
    shape = REFERENCE["NACA 2412"]
    r = panel_solve(shape, 4.0)
    n = airfoil_nodes(shape)
    mid = 0.5 * (n[:-1] + n[1:])
    d = n[1:] - n[:-1]
    L = np.hypot(*d.T)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / L[:, None]
    p = mid + 0.25 * L[:, None] * nrm                                     # a quarter panel outside
    u, v = r.velocity(p[:, 0], p[:, 1])
    vn = u * nrm[:, 0] + v * nrm[:, 1]
    assert np.median(np.abs(vn)) < 0.02
    u, v = r.velocity(np.array([1e4]), np.array([0.0]))
    assert u[0] == pytest.approx(math.cos(math.radians(4)), abs=1e-3) and v[0] == pytest.approx(math.sin(math.radians(4)), abs=1e-3)
