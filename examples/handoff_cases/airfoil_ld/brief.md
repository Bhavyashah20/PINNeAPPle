# Brief: 2-D airfoil lift-to-drag optimisation

## Objective

Find the airfoil section with the **highest lift-to-drag ratio L/D = Cl / Cd at alpha = 4 deg, Re = 4 x 10^6
(based on chord), Mach 0** (incompressible), computed with the CFD model of this package: steady RANS, k-omega SST,
fully turbulent, resolved to the wall (y+ < 1).

Baseline: NACA 2412 (its 6-weight CST fit), **L/D = 65.3** on the campaign grid (Cl 0.6625, Cd 0.01014,
Cm -0.0473).

## Constraints

| # | Constraint | Value | Why |
|---|---|---|---|
| C1 | maximum thickness t/c | >= 0.12 | spar depth and fuel volume of the baseline |
| C2 | pitching moment about the quarter chord at the design point | abs(Cm c/4) <= 0.10 | trim drag of the tail |
| C3 | valid shape | upper and lower surfaces do not cross (thickness > 0 between the leading and trailing edges); no wavy surface (abs(d2y/dx2) <= 40 on the upper surface away from the leading edge) | manufacturable, meshable |
| C4 | trailing edge | blunt, 0.25 % chord (fixed by the parametrisation) | meshing with the O-grid |
| C5 | flow at the design point | attached and steady (the "converged" status below) | the steady RANS result must be meaningful |

All constraints are checked on the CFD result, not only on a surrogate.

## Design variables

Six CST weights (Kulfan class-shape transformation, Bernstein order 2, class function sqrt(x)(1-x)): thickness
weights T0..T2 and camber weights M0..M2; upper surface weights M + T, lower surface weights M - T. Definitions and
the formula: `geometry/cst_parameters.json` and `geometry/make_geometry.py`.

| Variable | Meaning | Lower | Upper | NACA 2412 |
|---|---|---|---|---|
| T0 | thickness, leading edge | 0.10 | 0.24 | 0.1696 |
| T1 | thickness, mid chord | 0.08 | 0.22 | 0.1450 |
| T2 | thickness, aft | 0.08 | 0.22 | 0.1314 |
| M0 | camber, front | -0.03 | 0.08 | 0.0249 |
| M1 | camber, mid chord | -0.02 | 0.14 | 0.0711 |
| M2 | camber, aft | -0.04 | 0.12 | 0.0590 |

These are the bounds of the 436-run training campaign and of the surrogate; outside them the surrogate is
extrapolating. A design outside the bounds is allowed, but it must then be verified in CFD like any other.

## What is already known (results/)

- The 436-run campaign (100 airfoils x 4 angles + 4 NACA polars x 9 angles, Re 4 x 10^6): `results/campaign/`.
- Surrogates trained on it (MLP ensemble: Cl MAE 0.003, Cd error 0.5 % mean on held-out airfoils):
  `results/surrogate/`.
- A surrogate optimum under C1-C5, checked in OpenFOAM: **L/D 82.4** (Cl 0.923, Cd 0.01120, Cm -0.098) on the
  campaign grid, surrogate 82.6; three of the six weights sit on their upper bound and C2 is active
  (`results/optimisation/`). Its grid study is in `results/README.md`.
- Earlier aircraft-level designs of the same family (11 % thick, so they miss C1) reach L/D 81-83 at 4 deg in CFD.

## What a good answer contains

1. The design: the 6 CST weights (or coordinates, if another parametrisation is used), t/c, a coordinates file and
   an STL or STEP of the section.
2. The CFD result at the design point on the campaign grid and on the fine grid: Cl, Cd, Cm, L/D, y+ max, the
   force history, convergence status.
3. The constraint check C1-C5 on the CFD result.
4. The improvement over the baseline on the same grid, and an honest uncertainty (grid, plus surrogate error if
   one was used to search).
5. Off-design: L/D at alpha 2, 3, 5 and 6 deg, so a design that is only good at exactly 4 deg is visible.

A good result beats the baseline by at least 22 % (L/D >= 80 on the campaign grid; for scale, the NACA 4412,
which meets C1 and C2 but lies just outside the CST bounds, gives 78.7) and stays within the grid-uncertainty band
of the package's own optimum (82.4), or explains why it is lower (for example a tighter moment). L/D above about 90 at these constraints would be unexpected and should be checked for a convergence or
constraint problem.

## Acceptance checks

| Check | Pass |
|---|---|
| A1 Mesh quality | `checkMesh` reports "Mesh OK"; y+ of the wall-adjacent cells < 1 everywhere on the airfoil |
| A2 Iterative convergence | over the last 400 iterations Cl changes by < 2 x 10^-3 and Cd by < 5 x 10^-5 (status "converged"); residuals have dropped at least 3 orders |
| A3 Grid convergence | the optimum on three grids of the same family (coarse 6 840, medium 20 064, fine 45 144 cells, `simulation/airfoil_case.py --grid`): GCI of Cd on the fine grid < 3 % and the L/D ranking against the baseline the same on all three grids |
| A4 Reference comparison | the baseline NACA 2412 run with the same set-up reproduces: Cl at 4 deg within 3 % and Cd within 5 % of XFOIL with transition tripped at 1 % chord (Cl 0.678, Cd 0.00971, `results/reference/`); zero-lift angle within 0.2 deg of thin-airfoil theory (-2.08 deg); lift slope within 5 % of XFOIL tripped (0.112 /deg). The package's own baseline passes all three (`results/baseline/baseline_summary.json`). |
| A5 Verification of the optimum | if the optimum was found with a surrogate, its CFD L/D is within 2 % of the surrogate value, and C1-C5 hold in CFD; otherwise the search is repeated with the CFD point added |
| A6 Physical sense | the Cp distribution has no unphysical spikes (outside the leading-edge suction peak and the trailing edge); Cd > 0 split into pressure and friction drag; L/D gain explained by the shape (camber, loading) |

## Known limitations of the CFD model

- **Fully turbulent.** No transition model: drag is that of a fully turbulent (rough or tripped) section. XFOIL
  with free transition gives about half the drag (Cd 0.0056 at 4 deg); a laminar-flow design would need a
  transition model (for example gamma-ReTheta) and is out of scope unless stated.
- **2-D, incompressible**, no wind-tunnel walls.
- **Drag on this grid family** is grid-dependent at about 2 % (fine-grid GCI of Cd for the baseline: 1.9 %); the
  same grid family is used for every design, so comparisons hold better than absolute values.
- **Surrogate**: MLP ensemble trained on the campaign; its uncertainty (ensemble spread) is given, but a design near
  or beyond the bounds must be checked in CFD (A5).
