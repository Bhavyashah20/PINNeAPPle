"""PT-symmetric dimers and the no-go theorem for linear exceptional-point tuning (Zenodo 23287622)."""
import math

import numpy as np
import pytest
import sympy as sp

from pinneapple_physics.non_hermitian import (
    PTChain,
    PTDimer,
    analyze_ep_tuning,
    classify_coupling,
    design_features,
    feedback_coupling,
    petermann_factor,
    pt_violation,
    route_a_sqrt_coupling,
    route_c_asymmetric,
    theorem_profile,
)
from pinneapple_physics.non_hermitian.ep_tuning import _symbols
from pinneapple_physics.physics_audit import VALID, VIOLATED, audit_assumptions

LAM, J, G = _symbols()


def test_dimer_ep_closed_form_matches_matrix():
    for b in (1.0, 0.7 + 0.4j, 2.5j):
        d = PTDimer(b)
        assert d.is_pt_symmetric()
        assert d.ep_gamma() == pytest.approx(abs(b))
        assert d.ep_gamma_numeric() == pytest.approx(abs(b), rel=1e-7)
        g = abs(b)
        assert d.phase(0.5 * g) == "unbroken" and d.phase(1.5 * g) == "broken" and d.phase(g) == "EP"
        assert np.all(np.abs(np.linalg.eigvals(d.hamiltonian(0.5 * g)).imag) < 1e-12)     # real spectrum
        w = np.sort_complex(np.linalg.eigvals(d.hamiltonian(0.7 * g)))
        assert np.allclose(w, d.eigenvalues(0.7 * g))


def test_coalescence_at_the_ep():
    d = PTDimer(1.0)
    assert d.eigenvector_overlap(0.0) < 1e-12                              # Hermitian: orthogonal modes
    assert d.eigenvector_overlap(1.0 - 1e-8) > 0.999                        # eigenvectors coalesce
    assert d.splitting(1.0 - 1e-6) == pytest.approx(2 * math.sqrt(2e-6), rel=1e-3)   # square-root splitting
    k_far, k_near = petermann_factor(d.hamiltonian(0.2)), petermann_factor(d.hamiltonian(0.999))
    assert np.all(k_near > 100 * k_far)


def test_no_go_for_reciprocal_polynomial_couplings():
    c, c1, c2 = sp.Symbol("c"), sp.Symbol("c1"), sp.Symbol("c2")
    r = analyze_ep_tuning(J + c * LAM)
    assert r.theorem_applies and not r.exact and r.degree == 2 and r.perturbation_class == "reciprocal polynomial"
    assert sp.simplify(sp.Poly(r.ep_law, LAM).coeff_monomial(LAM ** 2) - c * sp.conjugate(c)) == 0   # |c|^2
    r2 = analyze_ep_tuning(J + c1 * LAM + c2 * LAM ** 2)
    assert r2.degree == 4 and not r2.exact and r2.routes
    # the best a real linear coupling can do: right slope, but the |c|^2 lam^2 term stays
    r3 = analyze_ep_tuning(J + G / 2 * LAM)
    assert sp.simplify(r3.residual - G ** 2 * LAM ** 2 / 4) == 0


def test_phase_only_and_the_three_routes():
    ph = analyze_ep_tuning(J * sp.exp(sp.I * LAM))
    assert ph.perturbation_class == "phase-only" and sp.simplify(ph.ep_law - J ** 2) == 0
    ra = analyze_ep_tuning(sp.sqrt(J ** 2 + LAM * J * G))
    assert ra.exact and ra.assumptions["A"] and not ra.assumptions["C"] and not ra.theorem_applies
    rc = analyze_ep_tuning(J, J + LAM * G)
    assert rc.exact and rc.perturbation_class == "non-reciprocal" and not rc.assumptions["A"]
    # numerically too
    d_a = PTDimer(route_a_sqrt_coupling(1.0, 0.5))
    ab, ba = route_c_asymmetric(1.0, 0.5)
    d_c = PTDimer(ab, ba)
    for lam in (-1.0, 0.0, 0.8):
        assert d_a.ep_gamma(lam) ** 2 == pytest.approx(1 + 0.5 * lam)
        assert d_a.ep_gamma_numeric(lam) ** 2 == pytest.approx(1 + 0.5 * lam, rel=1e-6)
        assert d_c.ep_gamma(lam) ** 2 == pytest.approx(1 + 0.5 * lam)
    assert d_a.is_pt_symmetric(lam=0.8) and not d_c.is_pt_symmetric(lam=0.8)
    b, k = feedback_coupling(1.0, 0.5, 0.8)
    assert b == pytest.approx(math.sqrt(1.4)) and k < 10


def test_classification():
    assert classify_coupling(J) == "constant"
    assert classify_coupling(J + 2 * LAM) == "reciprocal polynomial"
    assert classify_coupling(sp.sqrt(J ** 2 + LAM)) == "reciprocal non-polynomial"
    assert classify_coupling(J, J + LAM) == "non-reciprocal"


def test_hypothesis_map_as_physics_audit():
    prof = theorem_profile()
    pt = audit_assumptions(prof, design_features(PTDimer(1 + 0.3j).hamiltonian(0.5), polynomial=True))
    assert pt.status == VALID                                             # the theorem constrains this design
    ab, ba = route_c_asymmetric(1.0, 0.5)
    nr = audit_assumptions(prof, design_features(PTDimer(ab, ba).hamiltonian(0.5, 1.0), polynomial=True))
    assert nr.status == VIOLATED and {c.name for c in nr.violated} == {"pt_symmetry", "reciprocal_coupling"}
    tri = audit_assumptions(prof, design_features(PTChain(3, 1.0).hamiltonian(0.5), polynomial=True))
    assert [c.name for c in tri.violated] == ["two_level"]
    sq = audit_assumptions(prof, design_features(PTDimer(1.0).hamiltonian(0.5), polynomial=False))
    assert [c.name for c in sq.violated] == ["polynomial_coupling"]
    assert pt_violation(PTDimer(1.0).hamiltonian(0.3)) < 1e-15


def test_chains_threshold_scales_with_coupling():
    for n, kappa2, order in [(2, 1.0, 2), (3, 2.0, 3), (4, 1.0, 2), (5, 1.5, 3)]:
        for b in (0.5, 1.3, 0.8 + 0.6j):
            g = PTChain(n, b).ep_gamma_numeric()
            assert g ** 2 == pytest.approx(kappa2 * abs(b) ** 2, rel=1e-6)
        assert PTChain(n, 1.0).ep_order(PTChain(n, 1.0).ep_gamma_numeric()) == order


def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from pinneapple_physics.non_hermitian.viz import ep_tuning_figure

    p = ep_tuning_figure(str(tmp_path / "ep.png"))
    assert (tmp_path / "ep.png").stat().st_size > 10_000 and p.endswith("ep.png")
