# Results

2-D centre-plane interFoam runs of `simulation/trough_case.py`, water/air, k-omega SST, 12 s of physical time each,
averages over 6-12 s. Volumes and fluxes are per metre of span (m^2, m^2/s).

| Folder | Jet | Mesh |
|---|---|---|
| `mesh_5mm/` | baseline: U = 2.5 m/s, theta = 60 deg | 5 mm cells, 19 280 cells |
| `mesh_10mm/` | baseline | 10 mm cells, 4 800 cells (mesh sensitivity) |
| `screening_10mm/u2.0_angle45/` | U = 2.0 m/s, theta = 45 deg | 10 mm |
| `screening_10mm/u2.0_angle75/` | U = 2.0 m/s, theta = 75 deg | 10 mm |

Each folder: `summary.json` (every number below), `mass_balance.csv`, `floor.csv` (floor shear and pressure
profiles), `splash.csv`, figures (`free_surface_snapshots.png`, `checks.png`, `floor_shear_and_k.png`,
`k_vertical_lines.png`), `trough.json` (all inputs). The baseline folders also hold `solver_log.csv` (every 10th
time step: dt, Courant numbers, liquid fraction, alpha bounds), `k_lines_mean.json` (time-mean k and alpha on the
six sampling lines), the first and last steps of the solver log, `log.checkMesh` and VTK snapshots (`VTK/`, ASCII:
alpha.water, U, p_rgh, k, nut; 5 mm at t = 2, 6, 12 s, 10 mm at 12 s; open the `.vtm` files in ParaView).

## Physical and numerical checks

| Check | 5 mm | 10 mm | Acceptance (brief) |
|---|---|---|---|
| Liquid volume balance, end of run (% of total inflow) | 0.25 % | 0.03 % | < 1 % |
| Liquid volume balance, worst write after 1 s | 0.25 % | 0.21 % | < 1 % |
| Liquid volume: solver log vs fields | 3 x 10^-8 | 3 x 10^-8 | (consistency) |
| alpha min / max - 1 | -1.7 x 10^-4 / 8.8 x 10^-5 | -1.7 x 10^-4 / 5.1 x 10^-6 | within +- 10^-3 |
| max Courant number / max interface Courant number | 0.57 / 0.56 | 0.63 / 0.56 | <= 1 / about 0.5 (target 0.5, adaptive overshoots) |
| time steps | 21 892 | 11 176 | |
| steadiness: first vs second half of 6-12 s, tau_max / mean k 10 mm above the floor | 8 % / 4 % | 19 % / 17 % | < 15 % |
| floor y+ (max, first cell) | 470 | 710 | about 30 - 500 for wall functions |

The 5 mm run passes every check. The 10 mm run passes the balance, boundedness and Courant checks but not the
steadiness check (the bath is still filling at 12 s) and its floor y+ is above the wall-function range.

## Quantities of interest (baseline, averages over 6-12 s)

| QoI | 5 mm | 10 mm |
|---|---|---|
| Impingement point x_imp (floor pressure peak) | 0.519 m | 0.506 m |
| Ballistic jet axis at the mean free surface | 0.425 m | 0.422 m |
| Mean liquid depth upstream of the jet | 0.133 m | 0.141 m |
| Peak time-mean floor shear abs(tau_w), at x | 35.4 Pa at 0.554 m | 20.2 Pa at 0.566 m |
| Peak instantaneous floor shear | 66 Pa | 31 Pa |
| Highest liquid away from the jet (whole run) | 0.223 m (t = 2.9 s) | 0.335 m (t = 9.6 s): over the 0.25 m wall |
| Liquid out through the open top | 3.5 x 10^-5 m^2 (0.006 % of inflow) | 1.9 x 10^-4 m^2 (0.03 %) |
| Liquid volume in the domain at 0 s / 12 s | 0.096 / 0.077 m^2 | 0.096 / 0.150 m^2 |
| Max time-mean k, lines 10 / 30 / 60 mm above the floor | 2.7 / 2.8 / 4.4 m^2/s^2 | 1.6 / 1.7 / 2.6 m^2/s^2 |
| Mean k in the liquid, same lines | 0.30 / 0.22 / 0.32 m^2/s^2 | 0.12 / 0.16 / 0.22 m^2/s^2 |

The two meshes settle into different flow states. On the 5 mm mesh the jet lands about 0.1 m beyond the ballistic
point, runs along the floor as a thin fast layer and sweeps the downstream pool over the sill (`free_surface_snapshots.png`
from 4 s): liquid volume falls below the initial one, the floor shear stays high (15-35 Pa) over 0.55-1.0 m. On the
10 mm mesh a hydraulic jump stays in the trough, the downstream pool builds up and splashes over the wall. The
impingement point agrees within 13 mm (about one coarse cell); floor shear and splash do not. The coarse mesh is too
coarse for the floor layer (y+ 700, the 20 mm sheet only two cells thick), so the 5 mm result is the baseline and
the difference is a measure of how sensitive this regime is. An answer should check its design on at least the 5 mm
mesh and a finer one.

## Screening of two other jets (10 mm mesh, U = 2.0 m/s)

| Jet | x_imp | peak time-mean tau_w | max liquid height away from the jet | steadiness (tau halves) |
|---|---|---|---|---|
| baseline 2.5 m/s, 60 deg | 0.506 m | 20.2 Pa | 0.335 m (over the wall) | 19 % |
| 2.0 m/s, 45 deg | 0.558 m | 14.4 Pa | 0.265 m at 1.3 s (start-up wave, over the wall) | 23 % |
| 2.0 m/s, 75 deg | 0.405 m | 16.6 Pa | 0.245 m | 4 % |

Mass balance within 0.3 % in both. On this mesh a slower jet lowers the floor shear; a shallow jet lands further
downstream and lowers it most but its start-up wave overtops the wall; a steep jet keeps the splash inside but
shears the floor more. These are coarse-mesh trends only (the baseline shows how much the 10 mm mesh can differ from
5 mm), a starting point for the search, not an answer.
