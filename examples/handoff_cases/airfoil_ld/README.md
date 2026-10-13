# Case package: airfoil_ld (2-D airfoil lift-to-drag optimisation)

Everything needed to start: the task (`brief.md`), the geometry, a ready OpenFOAM case, the scripts that made every
number here, the existing CFD campaign and surrogate, a baseline run checked against references, an optimum checked
in CFD, and the references.

| Folder / file | Contents |
|---|---|
| `brief.md` | objective (max L/D at alpha 4 deg, Re 4 x 10^6), constraints, design variables and ranges, what a good answer looks like, acceptance checks |
| `geometry/` | CST parameter file (`cst_parameters.json`) for the NACA 2412 baseline and three variants; coordinates (`.dat`, Selig format) of each, of the exact NACA 2412 and of its CST fit; STEP and STL (section extruded one chord); `shapes.png`; `make_geometry.py` |
| `simulation/` | `baseline_case/` (ready OpenFOAM case, `Allrun`), the scripts (case writer, campaign runner, XFOIL reference, surrogate optimisation, post-processing), `README.md` with the exact set-up and how to run it |
| `results/` | baseline run on three grids with the grid study and the reference comparison; XFOIL polars; the 436-run campaign as CSV with three full fields as VTK; surrogate metrics; the optimised designs with their CFD verification; `results/README.md` summarises the numbers |
| `references.md` | NACA report data, XFOIL, OpenFOAM, k-omega SST (Menter 1994), CST (Kulfan 2008), GCI |

Units: lengths in chords (scale to any chord), coefficients per unit span. Software: OpenFOAM v1912, XFOIL 6.99,
Python 3 with numpy / scipy / matplotlib, the PINNeAPPle repository (`pip install -e .`), cadquery for STEP.

## Key numbers

NACA 2412 (CST fit, max deviation from the exact NACA ordinates 0.0022 c), alpha 4 deg, Re 4 x 10^6:

| Source | Cl | Cd | L/D |
|---|---|---|---|
| OpenFOAM k-omega SST, campaign grid (20 064 cells) | 0.6625 | 0.01014 | 65.3 |
| OpenFOAM k-omega SST, fine grid (45 144 cells) | 0.6624 | 0.00991 | 66.9 |
| OpenFOAM, Richardson extrapolation (3 grids) | 0.662 | 0.00976 | |
| XFOIL 6.99, exact NACA 2412, transition tripped at 1 % chord | 0.6778 | 0.00971 | 69.8 |
| XFOIL 6.99, exact NACA 2412, free transition (Ncrit 9) | 0.6812 | 0.00560 | 121.6 |
| Thin-airfoil theory: zero-lift angle | | | -2.08 deg (CFD -2.00, XFOIL tripped -2.04) |

The fully turbulent CFD matches the tripped XFOIL within 2.3 % in Cl and 4.5 % in Cd (2.0 % on the fine grid);
the free-transition XFOIL has much lower drag because of laminar flow, which the CFD model does not include.

Optimum found with the surrogate under the brief's constraints and run in OpenFOAM: L/D 82.4 (+26 % over the
baseline on the same grid), details and grid check in `results/README.md`.
