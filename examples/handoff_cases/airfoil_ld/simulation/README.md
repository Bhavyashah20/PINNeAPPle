# Simulation set-up: 2-D airfoil, OpenFOAM simpleFoam, k-omega SST

## Software used

| Tool | Version used here | Notes |
|---|---|---|
| OpenFOAM | v1912 (ESI/openfoam.com line), Ubuntu 24.04 package `openfoam` 1912.200626 | `apt install openfoam`; `source /usr/share/openfoam/etc/bashrc`. Any later openfoam.com release reads the same dictionaries. |
| Python | 3.13 with numpy, scipy, matplotlib | scripts in this folder |
| PINNeAPPle | this repository | `pip install -e .` from the repository root; the scripts import `pinneapple_design.aero` (geometry, mesh, case writer) |
| XFOIL | 6.99 | the reference polars. The Ubuntu package `xfoil` stops with SIGFPE (it is built with `-ffpe-trap=invalid,zero -finit-real=inf`); the Debian sources rebuilt with plain `-O2` were used (`apt-get source xfoil`, delete those flags in `bin/Makefile`, `plotlib/config.make` and `orrs/bin/Makefile_DP`, `make`). |
| cadquery | 2.8 | STEP / STL of the shapes only |

## The ready case: `baseline_case/`

NACA 2412 (its 6-weight CST fit), alpha = 4 deg, Re = 4 x 10^6, incompressible, fully turbulent.

```bash
cd baseline_case
./Allrun                 # checkMesh + simpleFoam, about 100 s on one core
python ../forces.py .    # Cl, Cd, Cm at every write -> forces.json (needs the PINNeAPPle repository)
./Allclean
```

The case is non-dimensional: chord c = 1, |U_inf| = 1, nu = 1/Re = 2.5 x 10^-7, kinematic pressure (rho = 1).

| Item | Setting |
|---|---|
| Mesh | structured O-grid written directly as `constant/polyMesh` by `pinneapple_design.aero.mesh`: 228 cells around the airfoil (112 per surface, 4 across the blunt trailing edge) x 88 cells wall-normal = 20 064 hexahedra, one cell (0.05 c) thick in z with `empty` front/back |
| Wall spacing | first cell 4 x 10^-6 c, geometric growth to the far field; y+ max 0.60 at alpha 4 (y+ < 1, no wall functions needed) |
| Far field | circle of radius 200 c centred at mid-chord (25 c gave +48 % drag at 10 deg) |
| Trailing edge | blunt, 0.25 % c thick, 4 cells across |
| Solver | `simpleFoam`, SIMPLEC (`consistent yes`), 1 non-orthogonal corrector, at most 2000 iterations (1200 in the 436-run campaign) |
| Turbulence | `kOmegaSST` (Menter 1994, OpenFOAM's default coefficients), fully turbulent (no transition model) |
| Schemes | steady; `linearUpwind` for U, `limitedLinear 1` for k and omega, cell-limited gradients for U, k, omega; `limited 0.5` laplacian / snGrad |
| Relaxation | U 0.8, k 0.6, omega 0.6, p 1 (SIMPLEC) |
| Residual targets | p 1e-6, U 1e-7, k/omega 1e-7 (rarely met; convergence is judged on the force history, below) |

Boundary conditions:

| Patch | U | p | k | omega | nut |
|---|---|---|---|---|---|
| `airfoil` (wall) | `noSlip` | `zeroGradient` | `fixedValue` 1e-14 | `omegaWallFunction` (low-Re switch, y+ < 1) | `nutLowReWallFunction` |
| `farfield` | `freestreamVelocity` (cos a, sin a, 0) | `freestreamPressure` 0 | `inletOutlet` 1.5e-6 | `inletOutlet` 6.0 | `calculated` |
| `frontAndBack` | `empty` | `empty` | `empty` | `empty` | `empty` |

Free-stream turbulence: intensity Tu = 0.1 % (k = 1.5 (Tu |U|)^2 = 1.5 x 10^-6) and nut/nu = 1 (omega = k / nu = 6.0).

**Forces.** Cl, Cd and Cm (about the quarter chord, nose-up positive, reference chord 1, span 0.05) come from the
pressure and viscous shear of the wall-adjacent cells (`pinneapple_design.aero.case.forces`), at every write (every
100 iterations), so the history is the convergence record. OpenFOAM function objects are not used: in this OpenFOAM
v1912 build any function object in `controlDict` aborts with `error in IOstream "sha1"`. With another build you can
add a `forceCoeffs` function object (patch `airfoil`, `CofR (0.25 0 0)`, `lRef 1`, `Aref 0.05`, `magUInf 1`,
`rhoInf 1`, lift and drag directions rotated by alpha) as an independent check.

**Convergence criterion** (the one used for the campaign): over the last five writes (400 iterations) Cl varies by
less than 2 x 10^-3 and Cd by less than 5 x 10^-5 ("converged"); less than 0.02 / 10^-3 is "steady-ish"; more is
"unsteady" (separated flow near stall, averaged over the last writes in the training set).

## Scripts

| Script | What it does |
|---|---|
| `airfoil_case.py` | writes (and with `--run` runs) one case from 6 CST weights or `nacaXXXX`, alpha, Re, grid (`coarse` 6 840, `medium` 20 064 = campaign, `fine` 45 144 cells) |
| `forces.py` | Cl, Cd, Cm history of a finished case |
| `run_campaign.py` | runs many variants in parallel: the shapes of `geometry/cst_parameters.json`, new Latin-hypercube shapes, or the original 436-run plan (`--original`) |
| `xfoil_reference.py` | XFOIL polars of the NACA 2412 (exact and CST fit, free and tripped transition) |
| `optimise_ld.py` | maximises L/D at alpha 4 with the trained surrogate (differential evolution) and checks the optimum in OpenFOAM |
| `summarise_baseline.py` | grid study (GCI), XFOIL and thin-airfoil comparison, figures in `results/baseline/` |
| `export_campaign.py`, `rerun_sample.py`, `plot_results.py` | the campaign CSV, re-runs of campaign runs with their fields as VTK, result figures |

Reproduce everything in `results/` (about 30 min on 4 cores):

```bash
W=work; mkdir -p $W
for g in coarse medium fine; do python airfoil_case.py $W/$g --shape naca2412 --alpha 4 --grid $g --run; done
python xfoil_reference.py ../results/reference --xfoil /path/to/xfoil
python optimise_ld.py $W/opt --cfd
python summarise_baseline.py $W ../results/baseline
python run_campaign.py $W/variants --variants ../geometry/cst_parameters.json --alphas 4 --clean
python export_campaign.py ../results/campaign/campaign_runs.csv
python rerun_sample.py $W/sample s012_a1 s047_a2 NACA4412_a+08
python plot_results.py ../results
```

The 436-run campaign itself and the surrogate training are in `apps/aero_optimizer` (`tools/generate_dataset.py`,
`tools/train_surrogate.py`, `tools/verify.py`); `run_campaign.py --original` repeats the same runs.
