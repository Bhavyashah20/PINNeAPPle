# Case package: free_surface_trough (liquid jet into an open trough, VOF)

A generic free-surface problem: a water jet from an inclined nozzle falls into an open rectangular trough that
drains over a sill. Choose the jet speed and angle so the jet lands in the right place, does not scour the floor
too hard and does not splash out. Two-phase VOF (OpenFOAM interFoam) with k-omega SST.

| Folder / file | Contents |
|---|---|
| `brief.md` | the system, objective, inputs that may vary, quantities of interest, acceptance checks, baseline results, model limits |
| `geometry/` | parametric trough + nozzle: `dimensions.csv` (dimension table), STEP (metres) and ASCII STL of the trough, the nozzle, both together (`assembly.step`) and the fluid region; `side_view.png` (dimensioned); `make_geometry.py` |
| `simulation/` | `trough_case.py` (writes, runs and post-processes a case with `pp.cfd.FreeSurfaceFlow`), `case_5mm/` (the baseline case as run, OpenFOAM-only), `README.md` (model, boundary conditions, turbulence choice, numerics, how to run) |
| `results/` | the baseline run on two meshes (5 mm and 10 mm cells, 12 s each): free-surface snapshots, impingement, splash, floor shear, mean k on sampling lines, mass balance, alpha bounds, Courant numbers, figures, VTK snapshots; `results/README.md` summarises them |
| `references.md` | VOF and interFoam, k-omega SST, plunging and impinging jets, fluid properties |

Software: OpenFOAM v1912, Python 3 with numpy / scipy / matplotlib, the PINNeAPPle repository
(`pip install -e .`), cadquery for the CAD. Units: SI (m, s, Pa).

## Key numbers (baseline: U = 2.5 m/s, theta = 60 deg, 2-D centre plane)

| | 5 mm mesh | 10 mm mesh |
|---|---|---|
| Liquid volume balance error (end of run, % of inflow) | 0.25 % | 0.03 % |
| alpha bounds | [-1.7 x 10^-4, 1 + 8.8 x 10^-5] | [-1.7 x 10^-4, 1 + 5.1 x 10^-6] |
| Max Courant / interface Courant number | 0.57 / 0.56 | 0.63 / 0.56 |
| Impingement point on the floor | x = 0.519 m | x = 0.506 m |
| Peak time-mean floor wall shear | 35.4 Pa | 20.2 Pa |
| Highest splash away from the jet (wall top 0.25 m) | 0.223 m | 0.335 m |

The baseline fails the floor-shear requirement of the brief; finding a jet that meets all of them is the task.
