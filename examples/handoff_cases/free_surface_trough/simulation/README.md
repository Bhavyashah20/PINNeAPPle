# Simulation set-up: liquid jet into an open trough, OpenFOAM interFoam (VOF)

## Software used

| Tool | Version used here | Notes |
|---|---|---|
| OpenFOAM | v1912 (openfoam.com line), Ubuntu 24.04 package `openfoam` 1912.200626 | `apt install openfoam`; `source /usr/share/openfoam/etc/bashrc` |
| Python | 3.13 with numpy, scipy, matplotlib | `trough_case.py` |
| PINNeAPPle | this repository | `pip install -e .` from the repository root; the case is written by `pp.cfd.FreeSurfaceFlow` (`pinneapple_simulation.numerical_solvers.free_surface`) |
| cadquery | 2.8 | `geometry/make_geometry.py` only |

## Run

```bash
python trough_case.py WORK/base --cell 0.005 --end 12 --run     # 5 mm cells, 12 s: about 1.5-3 h on one core
python trough_case.py WORK/coarse --cell 0.010 --end 12 --run   # 10 mm cells: about 20 min
python trough_case.py WORK/base --extend 16                       # continue a finished run to 16 s
python trough_case.py WORK/base --post-only                       # post-processing again
```

Other jets: `--speed` (m/s) and `--angle` (deg below the horizontal). Every other input is a field of the `Trough`
dataclass at the top of `trough_case.py` (geometry, bore, fluids, cell size, write interval, maximum Courant number).
`WORK/<case>/mesh` holds the blockMesh, `WORK/<case>/run` the interFoam case (`./Allrun`-free: `trough_case.py` runs
`blockMesh`, `checkMesh`, `setFields`, `interFoam`), `WORK/<case>/post` the post-processing.

`case_5mm/` in this folder is the baseline case as it was run (`0/` after `setFields`, `constant/` with the mesh,
`system/` with the `blockMeshDict` that made the mesh), so it can be run with OpenFOAM alone:

```bash
cd case_5mm
blockMesh > log.blockMesh 2>&1          # optional: constant/polyMesh is already there
setFields > log.setFields 2>&1          # liquid up to the sill crest (z < 0.08 m); 0/alpha.water already holds it
interFoam > log.interFoam 2>&1          # 12 s of physical time, 1.5-3 h on one core
```

## Model

**Domain.** The centre plane of the trough (x along it, z up), 2-D: one cell thick in y with `empty` front and back
faces. x = 0 is the inside face of the upstream end wall, x = L = 1.20 m the overflow sill; z = 0 is the floor and
the top of the domain, z = 0.40 m, is the height of the nozzle exit. The liquid leaves over the sill through the open
downstream boundary.

**The jet in 2-D.** The round nozzle (bore d = 20 mm) becomes a slot in the top boundary of width d / sin(theta)
= 23.1 mm at theta = 60 deg, centred under the nozzle exit (x = 0.30 m), with the full velocity vector imposed. The
jet is then a plane sheet 20 mm thick normal to its axis, with the same speed and angle as the round jet.
Inflow per metre of span: q = U d = 0.05 m^2/s at 2.5 m/s. A plane sheet carries more momentum per unit span than a
round jet of the same bore, so the 2-D case is a conservative screening model for impact and floor shear, not a
prediction of the 3-D values; the 3-D CAD (`geometry/`) is provided for the next step.

**Fluids** (20 °C, 1 atm):

| | density (kg/m^3) | kinematic viscosity (m^2/s) |
|---|---|---|
| water | 998.2 | 1.0034 x 10^-6 |
| air | 1.204 | 1.516 x 10^-5 |

Surface tension 0.0728 N/m. Gravity (0, 0, -9.81) m/s^2.

**Turbulence: k-omega SST (Menter 1994), RANS, with wall functions.** Why not laminar: the jet Reynolds number
U d / nu is 5 x 10^4 at the nozzle and the floor wall jet carries q / nu = 5 x 10^4 per unit width, both far above
transition, and the floor shear stress, one of the quantities of interest, is set by the turbulent wall jet; a
laminar model would under-predict it and the spreading of the impinging jet. Why SST: it is the k-omega model near
walls (the impingement and wall-jet region) and k-epsilon away from them, and it limits the eddy viscosity in
adverse pressure gradients such as stagnation regions. Known limits, stated in the brief: standard two-equation
models over-produce k near a free surface (Larsen and Fuhrman 2018); there is no buoyancy term in k; RANS does not
resolve the air entrained below the cell size. With 5 mm cells the floor y+ is in the wall-function range (see
`results/mesh_5mm/summary.json`, `floor_y_plus_of_first_cell_max`).

**Initial state.** Still liquid up to the sill crest (z < 0.08 m, `setFields`), air above, U = 0. hRef is at the
initial level. k and omega start uniform at the inlet values (k = 0.023 m^2/s^2) everywhere and decay in still
regions within the first second; the averages are taken over the second half of the run (6-12 s).

**Boundary conditions** (written by `FreeSurfaceFlow`):

| Patch | Type | U | p_rgh | alpha.water | k | omega | nut |
|---|---|---|---|---|---|---|---|
| `nozzle` (slot, top) | inlet | `fixedValue` (U cos theta, 0, -U sin theta) | `fixedFluxPressure` | `fixedValue` 1 | `turbulentIntensityKineticEnergyInlet` 5 % | `turbulentMixingLengthFrequencyInlet` 0.07 d = 1.4 mm | `calculated` |
| `outlet` (x = L above the sill) | open | `pressureInletOutletVelocity` | `prghTotalPressure` p0 = 0 | `inletOutlet` (0 if inflow) | `inletOutlet` | `inletOutlet` | `calculated` |
| `atmosphere` (top except slot; x = 0 above the wall) | open | `pressureInletOutletVelocity` | `prghTotalPressure` p0 = 0 | `inletOutlet` | `inletOutlet` | `inletOutlet` | `calculated` |
| `floor`, `upstreamWall`, `sill` | wall | `noSlip` | `fixedFluxPressure` | `zeroGradient` | `kqRWallFunction` | `omegaWallFunction` | `nutkWallFunction` |
| `frontAndBack` | 2-D | `empty` | `empty` | `empty` | `empty` | `empty` | `empty` |

**Mesh.** blockMesh, 9 blocks (columns: upstream of the slot, slot, downstream; rows: below the sill crest, up to
the wall top, above), uniform cells of about 5 mm (19 280 cells; 10 mm: 4 800 cells for the sensitivity run).

**Numerics.** interFoam (PIMPLE, 1 outer corrector, 3 pressure correctors, no momentum predictor), MULES with
`nAlphaCorr 2`, `cAlpha 1`, `vanLeer` for alpha and `linearUpwind` for U, first-order upwind for k and omega, Euler in
time. Adaptive time step with max Courant number 0.5 and max interface Courant number 0.5 (max time step 2 ms).
Fields written every 0.02 s.

## Post-processing (`trough_case.py`, writes `post/`)

| Output | How |
|---|---|
| Liquid volume balance (`mass_balance.csv`) | liquid volume from alpha x cell area at every write, against the time integral (trapezoidal) of the liquid fluxes alpha_f phi_f through `nozzle`, `outlet` and `atmosphere` read from the written fields; also compared with the solver's own "Phase-1 volume fraction" line at every step |
| alpha bounds, Courant numbers (`solver_log.csv`) | parsed from `log.interFoam` at every time step (every 10th step in the CSV) |
| Steadiness | the time-means of the peak floor shear and of k on the floor lines over the first and the second half of the averaging window |
| Impingement point | (a) where the floor pressure, minus the hydrostatic pressure of the local mean depth, peaks (time mean); (b) ballistic estimate: the jet axis from the nozzle exit to the mean free-surface level upstream of the jet |
| Floor wall shear stress (`floor.csv`) | tau_w = rho (nu + nut_w) u_P / y_P from the first cell above the floor and the wall-function nut on the floor, rho and nu of the local mixture; time mean, time mean of the magnitude, instantaneous maximum |
| Splash (`splash.csv`) | highest cell with alpha > 0.5 more than three slot widths away from the straight jet axis, at every write; liquid leaving through the open top |
| Time-mean k on lines (`k_lines_mean.json`) | nearest-cell values (no interpolation) on 3 horizontal lines 10, 30, 60 mm above the floor and 3 vertical lines at x = 0.45, 0.80, 1.10 m, sampled at every write by `FreeSurfaceFlow`'s line sampler, averaged over the second half of the run |
| Figures | free-surface snapshots, checks (balance, Courant, alpha bounds), floor shear and k, vertical k profiles |

VTK snapshots: `foamToVTK -time '1,4,8' -fields '(alpha.water U p_rgh k)' -noPointValues` in the run directory.

No OpenFOAM function objects are used: in this OpenFOAM v1912 build any function object in `controlDict` aborts with
`error in IOstream "sha1"`; everything above is computed from the written fields and the solver log.
