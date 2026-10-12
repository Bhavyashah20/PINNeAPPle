"""Does the no-go theorem for linear exceptional-point tuning extend from PT dimers to longer chains?

The dimer theorem (Zenodo 23287622): with a reciprocal polynomial coupling b(lam), gamma_EP^2 = |b(lam)|^2 has
degree 2 deg(b) and a positive leading coefficient, so it can never be the linear target
l(lam) = J^2 + lam J G (G != 0). The paper leaves open (Q2 / Conjecture 2) whether chains obey the same restriction.
Uniform chains do, by scaling. This graph attacks the first non-uniform case, the PT tetramer

    H = [[ i gamma, b1, 0, 0 ], [ b1*, 0, b2, 0 ], [ 0, b2*, 0, b3 ], [ 0, 0, b3*, -i gamma ]]

with the couplings polynomials in a real control lam, where the inner and outer bonds can be modulated
independently. Lemmas, checked by sympy (exact) and numerically:

* PT symmetry (P = exchange) holds iff b3 = b1; b2 is free;
* the characteristic polynomial depends only on A = |b1|^2 and B = |b2|^2:
  E^4 + (gamma^2 - 2A - B) E^2 + (A^2 - B gamma^2);
* the PT-breaking threshold lies on branch I (an E^2 root reaches 0: gamma^2 = A^2 / B) or on branch II (two E^2
  roots collide: (gamma^2 - 2A + B)^2 = 4 A (A - 2B)), the region of real spectrum being bounded by these two
  curves (argued, and checked numerically against the matrix);
* branch I cannot be linear: A^2 = B l would need an even degree (4 deg b1) to equal an odd one (2 deg b2 + 1);
* branch II cannot be linear: it squares to (B + l)^2 = 4 A (l - B), whose two sides have leading coefficients of
  opposite sign when their degrees match, and degrees of different parity otherwise.

Hence the theorem: no PT-preserving polynomial modulation of a tetramer realises exact linear EP tuning. The
general-degree steps are written arguments; the machine checks them exactly for every degree pair up to a bound
and numerically for random couplings, so the theorem's status is ARGUED (not PROVED). Chains with N >= 5 stay a
conjecture, supported by a counterexample search that found nothing.
"""
from __future__ import annotations

import itertools

import numpy as np

from .graph import ProofGraph, counterexample_search, custom, exhaustive, numeric, symbolic_identity

__all__ = ["build_pt_chain_graph", "threshold_numeric", "tetramer_threshold_branches", "chain_hamiltonian"]


def chain_hamiltonian(gamma: float, bonds: np.ndarray) -> np.ndarray:
    """N-site chain, +i gamma on the first site and -i gamma on the last, H[k, k+1] = b_k, H[k+1, k] = conj(b_k)."""
    n = len(bonds) + 1
    H = np.zeros((n, n), dtype=complex)
    for k, b in enumerate(bonds):
        H[k, k + 1], H[k + 1, k] = b, np.conj(b)
    H[0, 0], H[-1, -1] = 1j * gamma, -1j * gamma
    return H


def threshold_numeric(bonds, gmax: float | None = None, tol: float = 1e-11) -> float:
    """Smallest gamma at which the spectrum stops being real, by bisection on the matrix alone. The real-spectrum
    set in gamma need not be an interval; a scan on a fine grid finds the first breaking before bisecting."""
    bonds = np.asarray(bonds, dtype=complex)
    gmax = gmax or 3.0 * float(np.sum(np.abs(bonds))) + 1.0

    def broken(g):
        w = np.linalg.eigvals(chain_hamiltonian(g, bonds))
        return np.max(np.abs(w.imag)) > 1e-7 * max(1.0, np.max(np.abs(w)))
    if broken(0.0):
        return 0.0
    grid = np.linspace(0.0, gmax, 241)
    prev = 0.0
    for g in grid[1:]:
        if broken(g):
            lo, hi = prev, g
            while hi - lo > tol * max(1.0, hi):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if broken(mid) else (mid, hi)
            return 0.5 * (lo + hi)
        prev = g
    return float("nan")


def tetramer_threshold_branches(A: float, B: float) -> list[float]:
    """Candidate gamma^2 values: branch I (A^2 / B) and the real non-negative roots of branch II
    gamma^2 = 2A - B +/- 2 sqrt(A (A - 2B))."""
    out = [A * A / B] if B > 0 else []
    d = A * (A - 2 * B)
    if d >= 0:
        for s in (+1, -1):
            v = 2 * A - B + s * 2 * np.sqrt(d)
            if v >= 0:
                out.append(float(v))
    return out


def _poly_degree_checks(max_deg: int):
    """Exact checks over every degree pair (p, q) <= max_deg with generic coefficients (A, B with positive leading
    coefficients, as squared moduli of polynomials), for both branches."""
    import sympy as sp

    lam = sp.Symbol("lam", real=True)
    J, G = sp.Symbol("J", positive=True), sp.Symbol("G", nonzero=True, real=True)
    ell = J ** 2 + lam * J * G

    def generic(name, deg):
        cs = sp.symbols(f"{name}0:{deg + 1}", real=True)
        lead = sp.Symbol(f"{name}_lead", positive=True)
        return sum(c * lam ** k for k, c in enumerate(cs[:-1])) + lead * lam ** deg, lead

    def branch_I(pq):
        p, q = pq
        A, _ = generic("a", 2 * p)
        B, _ = generic("b", 2 * q)
        lhs, rhs = sp.Poly(A ** 2, lam), sp.Poly(B * ell, lam)
        return lhs.degree() != rhs.degree()                       # even (4p) vs odd (2q + 1)

    def branch_II(pq):
        p, q = pq
        A, a_l = generic("a", 2 * p)
        B, b_l = generic("b", 2 * q)
        diff = sp.Poly(sp.expand((B + ell) ** 2 - 4 * A * (ell - B)), lam)
        lead = sp.simplify(diff.LC())
        # nonzero for every admissible coefficient choice: provably positive, or a nonzero multiple of G or J
        if lead.is_positive:
            return True
        return bool(sp.simplify(lead / (J * G)).is_nonzero or sp.simplify(lead / G).is_nonzero or lead.is_nonzero)

    pairs = list(itertools.product(range(max_deg + 1), repeat=2))
    return branch_I, branch_II, pairs


def build_pt_chain_graph(max_deg: int = 4, n_numeric: int = 300, search_iters: int = 120,
                         searches: bool = True) -> ProofGraph:
    """The graph above. ``searches=False`` leaves out the (slow) counterexample searches of L6 and C1."""
    import sympy as sp

    g = ProofGraph("No linear exceptional-point tuning in PT-symmetric tetramers")
    g.assume("A1", "The couplings b_k(lam) are polynomials in a real control parameter lam (complex coefficients).")
    g.assume("A2", "Target law: gamma_EP^2(lam) = l(lam) = J^2 + lam J G with J > 0 and G != 0 (a non-constant line).")
    g.assume("A3", "The middle bond does not vanish identically (B = |b2|^2 is not the zero polynomial).")
    g.define("D1", "gamma_EP(lam): the PT-breaking threshold, the smallest gamma > 0 at which the spectrum of H "
                   "stops being real (an exceptional point).")

    gam = sp.Symbol("gamma", real=True)
    E = sp.Symbol("E")
    b1, b2, b3 = sp.symbols("b1 b2 b3")
    H = sp.Matrix([[sp.I * gam, b1, 0, 0], [sp.conjugate(b1), 0, b2, 0], [0, sp.conjugate(b2), 0, b3],
                   [0, 0, sp.conjugate(b3), -sp.I * gam]])
    P = sp.Matrix(4, 4, lambda i, j: 1 if i + j == 3 else 0)
    R = P * H.conjugate() * P - H
    g.add("L1", "lemma", "PT symmetry P H* P = H holds iff b3 = b1 (b2 is free).", ["D1"],
          [symbolic_identity("residual vanishes when b3 = b1", sp.Matrix(R.subs(b3, b1)).norm() ** 2, 0),
           symbolic_identity("residual entry (0,1) is b3 - b1", R[0, 1], b3 - b1)])
    Hpt = H.subs(b3, b1)
    A_, B_ = b1 * sp.conjugate(b1), b2 * sp.conjugate(b2)
    cp = (E * sp.eye(4) - Hpt).det()
    g.add("L2", "lemma", "det(E - H) = E^4 + (gamma^2 - 2A - B) E^2 + (A^2 - B gamma^2), A = |b1|^2, B = |b2|^2: the "
                         "spectrum depends on the couplings only through A and B.", ["L1"],
          [symbolic_identity("characteristic polynomial", cp, E ** 4 + (gam ** 2 - 2 * A_ - B_) * E ** 2
                             + (A_ ** 2 - B_ * gam ** 2))])
    A, B, x = sp.symbols("A B x", positive=True)
    p_, q_ = gam ** 2 - 2 * A - B, A ** 2 - B * gam ** 2
    rng_pts = lambda r: np.array([r.uniform(0.05, 3.0), r.uniform(0.05, 3.0)])   # noqa: E731

    def threshold_on_branches(ab):
        a, bb = ab
        num = threshold_numeric([np.sqrt(a), np.sqrt(bb), np.sqrt(a)]) ** 2
        cands = tetramer_threshold_branches(a, bb)
        return any(abs(num - c) <= 1e-6 * max(1.0, c) for c in cands)
    g.add("L3", "lemma", "With x = E^2, the spectrum is real iff both roots of x^2 + p x + q (p = gamma^2 - 2A - B, "
                         "q = A^2 - B gamma^2) are real and >= 0, so the threshold lies on branch I (q = 0: "
                         "gamma^2 = A^2/B) or branch II (p^2 = 4q: (gamma^2 - 2A + B)^2 = 4A(A - 2B)).", ["L2", "A3"],
          [symbolic_identity("discriminant p^2 - 4q", p_ ** 2 - 4 * q_, (gam ** 2 - 2 * A + B) ** 2 - 4 * A * (A - 2 * B)),
           symbolic_identity("branch I: q = 0 at gamma^2 = A^2/B", q_.subs(gam ** 2, A ** 2 / B), 0),
           symbolic_identity("on p = 0 the discriminant is -4q (a p = 0 exit is a q = 0 exit)",
                             (sp.Symbol("pp") ** 2 - 4 * sp.Symbol("qq")).subs(sp.Symbol("pp"), 0), -4 * sp.Symbol("qq")),
           numeric("matrix threshold lies on a branch (random A, B)", threshold_on_branches, rng_pts, n=n_numeric)],
          argument="E = +/- sqrt(x) is real iff x is real and >= 0, i.e. iff p^2 - 4q >= 0, q >= 0 and p <= 0 "
                   "(Vieta). These are continuous in gamma, so the first gamma where one fails has one of them at "
                   "equality; p = 0 with p^2 - 4q >= 0 forces q = 0 (checked), leaving branch I or branch II. The "
                   "continuity step is written, not machine-checked.", covered=False)
    br_I, br_II, pairs = _poly_degree_checks(max_deg)
    from sympy.solvers.diophantine import diophantine

    pd, qd, nd = sp.symbols("p q n", integer=True, nonnegative=True)
    c_, lam_r = sp.Symbol("c", nonzero=True), sp.Symbol("lam", positive=True)

    def no_integer_solution(eq):
        def run():
            sols = diophantine(eq)
            return (len(sols) == 0, True, "no integer solution" if not sols else f"solutions {sols}")
        return run

    def leading_sign_branch_II():
        al, bl = sp.symbols("A_l B_l", positive=True)
        lead = sp.expand(bl ** 2 - (-4 * al * bl))                 # LHS lead minus RHS lead when p = q >= 1
        ok = bool(lead.is_positive)
        return (ok, True, f"leading coefficient {lead} is positive" if ok else f"sign of {lead} undecided")

    def equal_degree_forces_p_equal_q():
        sol = sp.solve(sp.Eq(4 * qd, 2 * pd + 2 * qd), pd)
        return (sol == [qd], True, f"4q = 2p + 2q gives p = {sol}")
    g.add("L4", "lemma", "Branch I cannot carry the target: A(lam)^2 = B(lam) l(lam) is impossible for polynomial "
                         "couplings, since deg A^2 = 4 deg b1 is even and deg (B l) = 2 deg b2 + 1 is odd.",
          ["L3", "A1", "A2", "A3"],
          [symbolic_identity("leading term of |b|^2 is |c|^2 lam^(2n) (degree 2 deg b, positive coefficient)",
                             c_ * sp.conjugate(c_) * lam_r ** (2 * nd), (c_ * lam_r ** nd) * sp.conjugate(c_ * lam_r ** nd)),
           custom("4p = 2q + 1 has no integer solution (even degree vs odd degree), all degrees",
                  no_integer_solution(4 * pd - 2 * qd - 1)),
           exhaustive(f"degree parity on generic polynomials, deg b1, deg b2 <= {max_deg}", br_I, pairs,
                      "generic coefficients, positive leading terms")],
          argument="deg A^2 = 4 deg b1 (A = |b1|^2 has degree 2 deg b1) and deg (B l) = 2 deg b2 + 1; the two "
                   "polynomials cannot be equal.")
    l_s = sp.Symbol("l", real=True)
    g.add("L5", "lemma", "Branch II cannot carry the target: gamma^2 = l on branch II squares to (B + l)^2 = 4A(l - B); "
                         "when the degrees of the two sides match their leading coefficients have opposite signs "
                         "(B_lead^2 > 0 against -4 A_lead B_lead < 0), otherwise one side dominates.",
          ["L3", "A1", "A2", "A3"],
          [symbolic_identity("squaring branch II at gamma^2 = l", sp.expand((l_s - 2 * A + B) ** 2 - 4 * A * (A - 2 * B)),
                             sp.expand((B + l_s) ** 2 - 4 * A * (l_s - B))),
           custom("q >= 1: equal degrees 4q = 2p + 2q force p = q, all degrees", equal_degree_forces_p_equal_q),
           custom("p = q >= 1: leading coefficient B_l^2 + 4 A_l B_l of LHS - RHS is positive", leading_sign_branch_II),
           custom("q = 0: 2 = 2p + 1 has no integer solution, all degrees", no_integer_solution(2 * pd + 1 - 2)),
           exhaustive(f"leading coefficient of (B + l)^2 - 4A(l - B) never vanishes, deg b1, deg b2 <= {max_deg}",
                      br_II, pairs, "generic coefficients, positive leading terms")],
          argument="deg A = 2p, deg B = 2q. If q >= 1, deg LHS = 4q and deg RHS = 2p + 2q: unequal degrees leave a "
                   "nonzero leading term, equal degrees force p = q and a positive leading coefficient. If q = 0, "
                   "deg LHS = 2 while deg RHS = 2p + 1 is odd.")

    def nonlinearity(c):
        lam = np.linspace(-1.0, 1.0, 9)
        b1v = c[0] + c[1] * lam + c[2] * lam ** 2 + 1j * (c[3] * lam)
        b2v = c[4] + c[5] * lam + c[6] * lam ** 2
        g2 = np.array([threshold_numeric([u, v, u]) ** 2 for u, v in zip(b1v, b2v, strict=True)])
        if not np.all(np.isfinite(g2)):
            return 10.0
        fit = np.polyfit(lam, g2, 1)
        resid = np.max(np.abs(g2 - np.polyval(fit, lam))) / max(np.max(np.abs(g2)), 1e-9)
        slope_pen = max(0.0, 0.05 - abs(fit[0]) / max(np.max(np.abs(g2)), 1e-9))
        return resid + 5 * slope_pen
    if not searches:
        g.add("T1", "theorem", "No PT-preserving polynomial modulation of the tetramer couplings realises exact "
                               "linear exceptional-point tuning gamma_EP^2(lam) = J^2 + lam J G (G != 0).",
              ["L1", "L3", "L4", "L5"])
        return g
    g.add("L6", "lemma", "Numerical cross-check: no quadratic modulation of (b1, b2) makes the matrix threshold "
                         "gamma_EP^2(lam) a non-constant straight line on [-1, 1].", ["L2"],
          [counterexample_search("search over quadratic couplings", lambda c: nonlinearity(c) - 1e-6,
                                 [(0.3, 1.5), (-0.6, 0.6), (-0.4, 0.4), (-0.4, 0.4), (0.3, 1.5), (-0.6, 0.6),
                                  (-0.4, 0.4)], tol=1e-9, maxiter=search_iters, restarts=2)])
    g.add("T1", "theorem", "No PT-preserving polynomial modulation of the tetramer couplings realises exact linear "
                           "exceptional-point tuning gamma_EP^2(lam) = J^2 + lam J G (G != 0).",
          ["L1", "L3", "L4", "L5"], supported_by=["L6"])

    def five_site_nonlinearity(c):
        lam = np.linspace(-1.0, 1.0, 7)
        b1v = c[0] + c[1] * lam + c[2] * lam ** 2
        b2v = c[3] + c[4] * lam + c[5] * lam ** 2
        g2 = np.array([threshold_numeric([u, v, v, u]) ** 2 for u, v in zip(b1v, b2v, strict=True)])
        if not np.all(np.isfinite(g2)):
            return 10.0
        fit = np.polyfit(lam, g2, 1)
        resid = np.max(np.abs(g2 - np.polyval(fit, lam))) / max(np.max(np.abs(g2)), 1e-9)
        return resid + 5 * max(0.0, 0.05 - abs(fit[0]) / max(np.max(np.abs(g2)), 1e-9))
    g.add("C1", "conjecture", "The same restriction holds for PT chains with N >= 5 sites (paper's Conjecture 2).",
          ["T1"],
          [counterexample_search("pentamer: search over quadratic couplings (b1, b2, b2, b1)",
                                 lambda c: five_site_nonlinearity(c) - 1e-6,
                                 [(0.3, 1.5), (-0.6, 0.6), (-0.4, 0.4), (0.3, 1.5), (-0.6, 0.6), (-0.4, 0.4)],
                                 tol=1e-9, maxiter=max(20, search_iters // 3), restarts=1)])
    return g
