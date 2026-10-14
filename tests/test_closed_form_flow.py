"""Closed-form flow and heat helpers: pipe friction, compressible relations, slab conduction, inviscid Burgers,
exact Sod solution."""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from pinneapple_physics.closed_form import compressible as cf
from pinneapple_physics.closed_form import pipe_flow as pf
from pinneapple_physics.closed_form.burgers import burgers_sine_exact, burgers_sine_inviscid_exact
from pinneapple_physics.closed_form.heat import slab_transient


# ----------------------------------------------------------------------------------------------- pipe flow
def test_colebrook_moody_chart_values():
    # Moody chart / White, Fluid Mechanics, Fig. 6.13: eps/D = 1e-4 at Re = 1e5 -> f ~ 0.0185; smooth -> 0.0180
    assert pf.colebrook(1e5, 1e-4) == pytest.approx(0.0185, abs=2e-4)
    assert pf.colebrook(1e5) == pytest.approx(0.0180, abs=2e-4)
    f = pf.colebrook(3e6, 1e-3)
    assert 1 / math.sqrt(f) == pytest.approx(-2 * math.log10(1e-3 / 3.7 + 2.51 / (3e6 * math.sqrt(f))), rel=1e-12)


def test_haaland_within_two_percent_of_colebrook():
    for Re in (4e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        for e in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05):
            assert pf.haaland(Re, e) == pytest.approx(pf.colebrook(Re, e), rel=0.02)


def test_blasius_and_laminar():
    assert pf.blasius(1e4) == pytest.approx(0.316 * 1e4 ** -0.25)
    for Re in (5e3, 2e4, 1e5):                     # Blasius follows smooth-pipe Colebrook within a few percent
        assert pf.blasius(Re) == pytest.approx(pf.colebrook(Re), rel=0.04)
    assert pf.laminar(1000.0) == 0.064
    with pytest.warns(RuntimeWarning):
        pf.laminar(5000.0)
    with pytest.warns(RuntimeWarning):
        pf.blasius(1e6)
    with pytest.warns(RuntimeWarning):
        pf.haaland(1e3)
    with pytest.raises(ValueError):
        pf.colebrook(-1.0)


def test_hagen_poiseuille_equals_darcy_with_laminar_f():
    mu, rho, D, L, U = 1e-3, 1000.0, 0.01, 2.0, 0.1         # Re = 1000
    Re = rho * U * D / mu
    assert pf.darcy_pressure_drop(pf.laminar(Re), L, D, rho, U) == pytest.approx(32 * mu * L * U / D ** 2, rel=1e-12)
    # plane Poiseuille: u_max = 1.5 U, dp/L = 12 mu U / h^2 (White eq. 4.143)
    h = 0.002
    assert pf.parallel_plates_pressure_drop(mu, L, h, U) == pytest.approx(8 * mu * 1.5 * U / h ** 2 * L)


def test_first_cell_height():
    nu, D, U = 1.5e-5, 0.1, 15.0                       # Re = 1e5
    Re = U * D / nu
    y = pf.first_cell_height(1.0, Re, D, U, nu)
    f = pf.colebrook(Re)
    assert y == pytest.approx(nu / (U * math.sqrt(f / 8)), rel=1e-12)
    assert pf.first_cell_height(30.0, Re, D, U, nu) == pytest.approx(30 * y)
    assert pf.first_cell_height(1.0, Re, D, U, nu, f=0.02) == pytest.approx(nu / (U * math.sqrt(0.0025)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pf.first_cell_height(1.0, Re, D, U, nu)
    with pytest.warns(RuntimeWarning):
        pf.first_cell_height(1.0, 2 * Re, D, U, nu)


def test_internal_flow_colebrook_delegates():
    from pinneapple_simulation.numerical_solvers.internal_flow import colebrook
    assert colebrook(1e5) == pf.colebrook(1e5)
    assert colebrook(2e5, 1e-4) == pf.colebrook(2e5, 1e-4)


# ----------------------------------------------------------------------------------------------- compressible
def test_normal_shock_naca1135_table():
    # NACA Report 1135, Table II (gamma = 1.4, M1 = 2): p2/p1 = 4.500, rho2/rho1 = 2.667, T2/T1 = 1.688,
    # M2 = 0.5774, p02/p01 = 0.7209; Pitot p02/p1 = 5.640
    s = cf.normal_shock(2.0)
    assert float(s["p2_p1"]) == pytest.approx(4.5, abs=1e-12)
    assert float(s["rho2_rho1"]) == pytest.approx(2.6667, abs=1e-4)
    assert float(s["T2_T1"]) == pytest.approx(1.6875, abs=1e-4)
    assert float(s["M2"]) == pytest.approx(0.5774, abs=1e-4)
    assert float(s["p02_p01"]) == pytest.approx(0.7209, abs=1e-4)
    assert cf.pitot_pressure_ratio(2.0) == pytest.approx(5.640, abs=1e-3)
    # p02/p1 = (p02/p01) (p01/p1): shock recovery times the isentropic ratio
    M = np.array([1.2, 2.0, 3.5, 6.0])
    assert np.allclose(cf.pitot_pressure_ratio(M), cf.normal_shock(M)["p02_p01"] * cf.isentropic_ratios(M)["p0_p"])
    assert np.allclose(cf.normal_shock(1.0)["M2"], 1.0)
    with pytest.raises(ValueError):
        cf.normal_shock(0.8)


def test_isentropic_ratios_naca1135():
    r = cf.isentropic_ratios(2.0)                  # NACA 1135 Table I, M = 2: p/p0 = 0.1278, T/T0 = 0.5556, A/A* = 1.688
    assert 1 / float(r["p0_p"]) == pytest.approx(0.1278, abs=1e-4)
    assert 1 / float(r["T0_T"]) == pytest.approx(0.5556, abs=1e-4)
    assert float(r["A_Astar"]) == pytest.approx(1.6875, abs=1e-4)
    assert float(cf.isentropic_ratios(1.0)["A_Astar"]) == pytest.approx(1.0)
    assert cf.pitot_pressure_ratio(0.5) == pytest.approx(float(cf.isentropic_ratios(0.5)["p0_p"]))


def test_cp_max_newtonian():
    g, M = 1.4, 2.0
    assert cf.cp_max_newtonian(M) == pytest.approx((5.6404 - 1) / (0.5 * g * M * M), abs=1e-4)   # 1.657
    limit = ((g + 1) ** 2 / (4 * g)) ** (g / (g - 1)) * 4 / (g + 1)                              # 1.8394
    assert cf.cp_max_newtonian(1e4) == pytest.approx(limit, rel=1e-6)
    assert limit == pytest.approx(1.8394, abs=1e-4)


# ----------------------------------------------------------------------------------------------- heat
L, ALPHA, TL, TR, T0 = 0.5, 1e-4, 100.0, 20.0, 50.0


def test_slab_steady_limit_and_boundaries():
    x = np.linspace(0, L, 21)
    late = slab_transient(x, 50 * L ** 2 / ALPHA, L, ALPHA, TL, TR, T0)
    assert np.allclose(late, TL + (TR - TL) * x / L, atol=1e-10)
    t = np.array([1.0, 30.0, 300.0])
    assert np.allclose(slab_transient(0.0, t, L, ALPHA, TL, TR, T0), TL, atol=1e-8)
    assert np.allclose(slab_transient(L, t, L, ALPHA, TL, TR, T0), TR, atol=1e-8)
    assert np.allclose(slab_transient(x[1:-1], 0.0, L, ALPHA, TL, TR, T0), T0)
    # early time near a wall: the semi-infinite solid, T0 + (TL - T0) erfc(x / (2 sqrt(alpha t))) (Carslaw &
    # Jaeger §2.4), since the penetration depth sqrt(alpha t) = 0.03 is far below L
    from scipy.special import erfc
    xe = np.linspace(0, 0.15, 16)
    semi = T0 + (TL - T0) * erfc(xe / (2 * np.sqrt(ALPHA * 10.0)))
    assert np.allclose(slab_transient(xe, 10.0, L, ALPHA, TL, TR, T0), semi, atol=1e-6)


def test_slab_satisfies_the_heat_equation():
    x = np.linspace(0.05, 0.45, 9)
    t, dt, dx = 200.0, 1e-2, 1e-4
    T = lambda xx, tt: slab_transient(xx, tt, L, ALPHA, TL, TR, T0)  # noqa: E731
    Tt = (T(x, t + dt) - T(x, t - dt)) / (2 * dt)
    Txx = (T(x + dx, t) - 2 * T(x, t) + T(x - dx, t)) / dx ** 2
    scale = np.abs(Tt).max()
    assert scale > 1e-3
    assert np.abs(Tt - ALPHA * Txx).max() < 1e-3 * scale


def test_slab_matches_explicit_finite_differences():
    n = 101
    x = np.linspace(0, L, n)
    h = x[1] - x[0]
    dt = 0.4 * h * h / ALPHA
    T = np.full(n, T0)
    T[0], T[-1] = TL, TR
    t_end = 300.0
    steps = int(round(t_end / dt))
    for _ in range(steps):
        T[1:-1] += ALPHA * dt / h ** 2 * (T[2:] - 2 * T[1:-1] + T[:-2])
    exact = slab_transient(x, steps * dt, L, ALPHA, TL, TR, T0)
    assert np.abs(T - exact).max() < 0.1          # O(h^2 + dt) explicit FTCS on a 101-point grid


# ----------------------------------------------------------------------------------------------- Burgers
def test_inviscid_burgers_before_shock_matches_fixed_point():
    x = np.linspace(-0.95, 0.95, 39)
    for t in (0.05, 0.15, 0.3):                    # t < 1/pi: smooth, u = -sin(pi (x - u t)) is a contraction
        u = -np.sin(np.pi * x)
        for _ in range(2000):
            u = -np.sin(np.pi * (x - u * t))
        assert np.allclose(burgers_sine_inviscid_exact(x, t), u, atol=1e-10)
    assert np.allclose(burgers_sine_inviscid_exact(x, 0.0), -np.sin(np.pi * x))


def test_inviscid_burgers_is_the_vanishing_viscosity_limit():
    x = np.linspace(-0.9, 0.9, 37)
    x = x[np.abs(x) > 0.15]                        # away from the shock at x = 0
    for t in (0.6, 1.0):                           # after the shock forms at t = 1/pi
        e3 = np.abs(burgers_sine_exact(x, t, 3e-3, n_nodes=200) - burgers_sine_inviscid_exact(x, t)).max()
        e1 = np.abs(burgers_sine_exact(x, t, 1e-3, n_nodes=200) - burgers_sine_inviscid_exact(x, t)).max()
        assert e1 < 2e-3 and e1 < 0.5 * e3         # error O(nu)
    u = burgers_sine_inviscid_exact(np.array([-1e-6, 0.0, 1e-6]), 0.8)
    assert u[0] > 0.5 and u[1] == 0.0 and u[2] < -0.5   # shock pinned at x = 0


# ----------------------------------------------------------------------------------------------- Sod
def test_sod_exact_solutions_agree_and_match_toro():
    from pinneapple_physics.pde_environment.presets.astrophysics import sod_exact_solution
    from pinneapple_simulation.numerical_solvers.compressible_fv import sod_exact

    x = np.linspace(0, 1, 801)
    rho, u, p = sod_exact(x, 0.2)
    ref = sod_exact_solution(x, 0.2)
    for a, b in ((rho, ref["rho"]), (u, ref["u"]), (p, ref["p"])):
        assert np.array_equal(a, b)
    # Toro (2009), Table 4.3, test 1: p* = 0.30313, u* = 0.92745
    assert ref["p_star"] == pytest.approx(0.30313, abs=1e-5)
    assert ref["u_star"] == pytest.approx(0.92745, abs=1e-5)
    # independent star pressure: Brent's method on Toro's pressure function (eq. 4.5-4.7)
    g = 1.4

    def fk(pp, r, pk):
        c = math.sqrt(g * pk / r)
        if pp > pk:
            return (pp - pk) * math.sqrt(2 / ((g + 1) * r) / (pp + (g - 1) / (g + 1) * pk))
        return 2 * c / (g - 1) * ((pp / pk) ** ((g - 1) / (2 * g)) - 1)

    ps = brentq(lambda q: fk(q, 1.0, 1.0) + fk(q, 0.125, 0.1), 1e-8, 1.0, xtol=1e-14)
    assert ref["p_star"] == pytest.approx(ps, rel=1e-9)
    assert np.isclose(p[(x > 0.7) & (x < 0.8)], ps).all()            # star region
    # mirrored problem (right-moving rarefaction, left shock) is the reflection of the standard one
    rho_m, u_m, p_m = sod_exact(1 - x, 0.2, left=(0.125, 0.0, 0.1), right=(1.0, 0.0, 1.0))
    keep = np.abs(np.diff(rho, prepend=rho[0])) < 1e-3                 # skip points on a discontinuity
    assert np.allclose(rho_m[keep], rho[keep]) and np.allclose(u_m[keep], -u[keep]) and np.allclose(p_m[keep], p[keep])
