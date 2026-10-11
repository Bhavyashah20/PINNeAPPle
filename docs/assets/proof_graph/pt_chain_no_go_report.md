# No linear exceptional-point tuning in PT-symmetric tetramers

Status: ASSUMED 4, PROVED 2, ARGUED 4, SUPPORTED 2

## [ASSUMED] assumption A1
The couplings b_k(lam) are polynomials in a real control parameter lam (complex coefficients).

## [ASSUMED] assumption A2
Target law: gamma_EP^2(lam) = l(lam) = J^2 + lam J G with J > 0 and G != 0 (a non-constant line).

## [ASSUMED] assumption A3
The middle bond does not vanish identically (B = |b2|^2 is not the zero polynomial).

## [ASSUMED] definition D1
gamma_EP(lam): the PT-breaking threshold, the smallest gamma > 0 at which the spectrum of H stops being real (an exceptional point).

## [PROVED] lemma L1
PT symmetry P H* P = H holds iff b3 = b1 (b2 is free).
- depends on: D1
- check (exact) residual vanishes when b3 = b1: passed, simplifies to 0 [0.00 s]
- check (exact) residual entry (0,1) is b3 - b1: passed, simplifies to 0 [0.00 s]
- status because: all checks exact and passed, dependencies proved

## [PROVED] lemma L2
det(E - H) = E^4 + (gamma^2 - 2A - B) E^2 + (A^2 - B gamma^2), A = |b1|^2, B = |b2|^2: the spectrum depends on the couplings only through A and B.
- depends on: L1
- check (exact) characteristic polynomial: passed, simplifies to 0 [0.00 s]
- status because: all checks exact and passed, dependencies proved

## [ARGUED] lemma L3
With x = E^2, the spectrum is real iff both roots of x^2 + p x + q (p = gamma^2 - 2A - B, q = A^2 - B gamma^2) are real and >= 0, so the threshold lies on branch I (q = 0: gamma^2 = A^2/B) or branch II (p^2 = 4q: (gamma^2 - 2A + B)^2 = 4A(A - 2B)).
- depends on: L2, A3
- check (exact) discriminant p^2 - 4q: passed, simplifies to 0 [0.00 s]
- check (exact) branch I: q = 0 at gamma^2 = A^2/B: passed, simplifies to 0 [0.00 s]
- check (exact) on p = 0 the discriminant is -4q (a p = 0 exit is a q = 0 exit): passed, simplifies to 0 [0.00 s]
- check (numerical) matrix threshold lies on a branch (random A, B): passed, holds on 300 random samples [0.57 s]
- argument (not machine-checked): E = +/- sqrt(x) is real iff x is real and >= 0, i.e. iff p^2 - 4q >= 0, q >= 0 and p <= 0 (Vieta). These are continuous in gamma, so the first gamma where one fails has one of them at equality; p = 0 with p^2 - 4q >= 0 forces q = 0 (checked), leaving branch I or branch II. The continuity step is written, not machine-checked.
- status because: checks passed; part of the argument is a written step not machine-checked

## [ARGUED] lemma L4
Branch I cannot carry the target: A(lam)^2 = B(lam) l(lam) is impossible for polynomial couplings, since deg A^2 = 4 deg b1 is even and deg (B l) = 2 deg b2 + 1 is odd.
- depends on: L3, A1, A2, A3
- check (exact) leading term of |b|^2 is |c|^2 lam^(2n) (degree 2 deg b, positive coefficient): passed, simplifies to 0 [0.00 s]
- check (exact) 4p = 2q + 1 has no integer solution (even degree vs odd degree), all degrees: passed, no integer solution [0.01 s]
- check (exact) degree parity on generic polynomials, deg b1, deg b2 <= 4: passed, holds for all 25 cases (generic coefficients, positive leading terms) [0.09 s]
- argument: deg A^2 = 4 deg b1 (A = |b1|^2 has degree 2 deg b1) and deg (B l) = 2 deg b2 + 1; the two polynomials cannot be equal.
- status because: a dependency is only argued

## [ARGUED] lemma L5
Branch II cannot carry the target: gamma^2 = l on branch II squares to (B + l)^2 = 4A(l - B); when the degrees of the two sides match their leading coefficients have opposite signs (B_lead^2 > 0 against -4 A_lead B_lead < 0), otherwise one side dominates.
- depends on: L3, A1, A2, A3
- check (exact) squaring branch II at gamma^2 = l: passed, simplifies to 0 [0.00 s]
- check (exact) q >= 1: equal degrees 4q = 2p + 2q force p = q, all degrees: passed, 4q = 2p + 2q gives p = [q] [0.00 s]
- check (exact) p = q >= 1: leading coefficient B_l^2 + 4 A_l B_l of LHS - RHS is positive: passed, leading coefficient 4*A_l*B_l + B_l**2 is positive [0.00 s]
- check (exact) q = 0: 2 = 2p + 1 has no integer solution, all degrees: passed, no integer solution [0.00 s]
- check (exact) leading coefficient of (B + l)^2 - 4A(l - B) never vanishes, deg b1, deg b2 <= 4: passed, holds for all 25 cases (generic coefficients, positive leading terms) [1.05 s]
- argument: deg A = 2p, deg B = 2q. If q >= 1, deg LHS = 4q and deg RHS = 2p + 2q: unequal degrees leave a nonzero leading term, equal degrees force p = q and a positive leading coefficient. If q = 0, deg LHS = 2 while deg RHS = 2p + 1 is odd.
- status because: a dependency is only argued

## [SUPPORTED] lemma L6
Numerical cross-check: no quadratic modulation of (b1, b2) makes the matrix threshold gamma_EP^2(lam) a non-constant straight line on [-1, 1].
- depends on: L2
- check (numerical) search over quadratic couplings: passed, no counterexample; smallest margin found 0.0001857 (2 searches) [176.43 s]
- status because: numerical evidence only

## [ARGUED] theorem T1
No PT-preserving polynomial modulation of the tetramer couplings realises exact linear exceptional-point tuning gamma_EP^2(lam) = J^2 + lam J G (G != 0).
- depends on: L1, L3, L4, L5
- corroborated by: L6 [SUPPORTED]
- status because: a dependency is only argued

## [SUPPORTED] conjecture C1
The same restriction holds for PT chains with N >= 5 sites (paper's Conjecture 2).
- depends on: T1
- check (numerical) pentamer: search over quadratic couplings (b1, b2, b2, b1): passed, no counterexample; smallest margin found 0.0007425 (1 searches) [39.50 s]
- status because: numerical evidence only
