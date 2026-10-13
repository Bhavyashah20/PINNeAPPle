# Brief: liquid jet into an open trough (free-surface VOF)

## The system

A round jet of water leaves a nozzle above an open, rectangular trough and falls into it. The trough is
1.20 m long, 0.15 m wide inside, with 0.25 m high side walls and upstream end wall; at the downstream end a 0.08 m
high sill spans the full width, and the liquid leaves the trough by flowing over it. The nozzle (bore 20 mm) is
0.30 m from the upstream wall, 0.40 m above the floor, in the centre plane, and points downstream, inclined below
the horizontal. Before the jet starts the trough holds still water up to the sill crest. Geometry, dimension
table and CAD: `geometry/`.

The concern is the trough floor and the surroundings: where the jet lands, how hard the liquid scours the floor,
and whether liquid splashes over the walls.

## Objective

Choose the **jet speed U** and **jet angle theta** (below the horizontal) that **minimise the peak time-mean wall
shear stress on the trough floor**, while delivering at least the required flow and keeping the impingement zone
and the splash within limits:

| # | Requirement | Limit |
|---|---|---|
| R1 | delivered flow (round nozzle, bore d) | U pi d^2 / 4 >= 0.63 l/s, i.e. U >= 2.0 m/s at d = 20 mm |
| R2 | impingement zone on the floor (time-mean floor pressure peak) | 0.35 m <= x_imp <= 0.75 m from the upstream wall (clear of the upstream wall and of the sill) |
| R3 | time-mean floor wall shear stress, anywhere on the floor | max over x of the time mean of abs(tau_w) <= 15 Pa |
| R4 | splash | no liquid (alpha > 0.5) above the wall top (z = 0.25 m) away from the jet column, at any write time; liquid leaving through the open top < 0.1 % of the inflow |
| R5 | statistically steady operation | the outflow over the sill and the liquid volume fluctuate about a constant mean over the averaging window |

R3 and R4 are screening limits chosen for this exercise (a generic lining and a dry surrounding floor), not values
from a standard.

## Inputs that may vary

| Input | Baseline | Range | Notes |
|---|---|---|---|
| jet speed U | 2.5 m/s | 2.0 - 4.0 m/s | at the nozzle exit, uniform |
| jet angle theta | 60 deg | 30 - 90 deg | below the horizontal, in the vertical plane along the trough, pointing downstream |
| nozzle exit height z_n | 0.40 m | 0.30 - 0.50 m | optional third variable |
| nozzle position x_n | 0.30 m | fixed | |
| nozzle bore d | 20 mm | fixed | |
| trough, sill, liquid | as in `geometry/dimensions.csv` | fixed | water at 20 °C |

## Quantities of interest

| QoI | Definition (as computed by `simulation/trough_case.py`) |
|---|---|
| x_imp | x of the peak of the time-mean floor pressure minus the hydrostatic pressure of the local mean depth; with the ballistic estimate of where the jet axis meets the free surface |
| tau_max | max over x of the time-mean of abs(tau_w) on the floor; also the instantaneous maximum and the floor profile |
| splash height | highest liquid (alpha > 0.5) more than three slot widths from the jet axis, per write; liquid volume out through the open top |
| mean k on lines | time-mean turbulent kinetic energy on horizontal lines 10, 30, 60 mm above the floor and vertical lines at x = 0.45, 0.80, 1.10 m |
| free-surface shape | alpha = 0.5 contours at a sequence of times; mean depth along the trough |
| outflow | liquid flux over the sill against time |

## Baseline (U = 2.5 m/s, theta = 60 deg), 2-D centre-plane CFD

Full results: `results/README.md`. 12 s of physical time, averages over 6-12 s.

| | 5 mm mesh (reference) | 10 mm mesh | Requirement |
|---|---|---|---|
| R1 flow | 0.79 l/s (round nozzle) | | >= 0.63 l/s: met |
| R2 impingement x_imp | 0.519 m (ballistic estimate 0.425 m) | 0.506 m | 0.35-0.75 m: met |
| R3 peak time-mean floor shear | 35.4 Pa at x = 0.55 m | 20.2 Pa | <= 15 Pa: **not met** |
| R4 splash height / out through the top | 0.223 m / 0.006 % | 0.335 m / 0.03 % | < 0.25 m / < 0.1 %: met on 5 mm, not on 10 mm |
| R5 steady statistics (tau halves) | 8 % | 19 % | < 15 %: met on 5 mm |
| Mass balance / alpha bounds / max Co | 0.25 % / [-1.7e-4, 1 + 8.8e-5] / 0.57 | 0.03 % / [-1.7e-4, 1 + 5e-6] / 0.63 | A1-A3: met |

The baseline jet does not meet R3: a different speed and angle are needed. A coarse (10 mm) screening of two
other jets is in `results/screening_10mm/`: at 2.0 m/s, 45 deg lowers the floor shear to 14 Pa but its start-up
wave overtops the wall; 75 deg keeps the splash inside but gives 17 Pa. The two baseline meshes settle into
different flow states (pool swept over the sill on 5 mm, hydraulic jump held in the trough on 10 mm), so treat
10 mm numbers as trends only.

## Acceptance checks

| Check | Pass |
|---|---|
| A1 Mass balance | change of liquid volume vs time-integrated net liquid inflow (inlet minus outlet and open top) within 1 % of the total inflow, at the end and at every write after 1 s |
| A2 Boundedness | liquid fraction alpha within [-10^-3, 1 + 10^-3] at every time step |
| A3 Time step | Courant number <= 1 and interface Courant number <= 0.5 (nominal; brief adaptive overshoots of max Co up to about 0.65 are acceptable if A1 and A2 hold) |
| A4 Mesh sensitivity | the QoIs on two meshes (cell size h and 2h): the ranking of the candidate designs the same on both; x_imp within two cell sizes of the coarse value; tau_max reported on both meshes with its change |
| A5 Steady statistics | averaging window starts after the initial filling transient; the two halves of the window give time-means within 15 % of each other for tau_max and the mean k on the floor lines |
| A6 Physical sense | x_imp consistent with the ballistic jet trajectory to the free surface (within about 0.1 m, the jet is deflected in the pool); outflow over the sill matches the inflow on average once steady; floor y+ in the wall-function range (about 30 - 500) where tau_w is reported |
| A7 Design verification | the chosen (U, theta) is run with the full set-up, all requirements R1-R5 met in that run, not only in a surrogate or an interpolation |

## Known limitations of the model

- **2-D centre plane.** The round jet is a plane sheet of the same thickness (20 mm), speed and angle. A sheet carries
  more momentum per unit width than the round jet, and side-wall friction and three-dimensional spreading are
  absent, so floor shear and splash are conservative screening values. The 3-D geometry is in `geometry/` for the
  next step.
- **RANS (k-omega SST) with wall functions**: no buoyancy term in k, over-production of k near the free surface is
  known for two-equation models; air entrainment below the cell size is not modelled.
- **Coarse mesh** (5 mm, 10 mm for the sensitivity run) and short physical time (12 s): fine for ranking designs, not
  for absolute values.
