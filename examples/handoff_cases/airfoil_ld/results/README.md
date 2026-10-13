# Results

All runs: OpenFOAM v1912 simpleFoam, k-omega SST, fully turbulent, Re 4 x 10^6, the O-grid family of
`simulation/airfoil_case.py`. Coefficients per unit span, chord 1, moment about the quarter chord.

## `baseline/`: NACA 2412 at alpha 4 deg, three grids, reference comparison

| Grid | Cells | Cl | Cd | Cm | L/D | y+ max | status |
|---|---|---|---|---|---|---|---|
| coarse | 6 840 | 0.6614 | 0.011120 | -0.0481 | 59.5 | 0.59 | converged |
| medium (= campaign grid) | 20 064 | 0.6625 | 0.010143 | -0.0473 | 65.3 | 0.60 | converged |
| fine | 45 144 | 0.6624 | 0.009908 | -0.0469 | 66.9 | 0.60 | converged |
| Richardson extrapolation (Celik et al. 2008) | | 0.6624 | 0.00976 | | 68.0 | | |

Grid convergence index on the fine grid: Cd 1.9 % (observed order 2.4), L/D 2.1 %, Cl < 0.01 %.
Iterative convergence: residuals down 5 to 8 orders, Cl constant to 2 x 10^-5 and Cd to 10^-6 over the last 500 iterations
(`grid_study.csv`, `convergence.png`).

Reference comparison at alpha 4 deg (`baseline_summary.json`, `polar_vs_xfoil.png`, `cp_alpha4.png`):

| Reference | Cl | Cd | CFD medium vs reference | CFD fine vs reference |
|---|---|---|---|---|
| XFOIL 6.99, exact NACA 2412, tripped at 1 % chord | 0.6778 | 0.00971 | Cl -2.3 %, Cd +4.5 % | Cl -2.3 %, Cd +2.0 % |
| XFOIL 6.99, CST fit, tripped | 0.6766 | 0.00972 | Cl -2.1 %, Cd +4.4 % | |
| XFOIL 6.99, exact NACA 2412, free transition (Ncrit 9) | 0.6812 | 0.00560 | laminar runs: not comparable in drag | |

| Lift curve (fit over -2 ... 6 deg) | zero-lift angle | slope |
|---|---|---|
| OpenFOAM (campaign NACA 2412 polar) | -2.00 deg | 0.110 /deg |
| XFOIL tripped | -2.04 deg | 0.112 /deg |
| Thin-airfoil theory (NACA 24xx mean line) | -2.08 deg | 0.110 /deg (2 pi /rad) |

The surface pressure of the CFD and of XFOIL (tripped) agree except for a slightly higher suction peak in the CFD.
Fields of the medium-grid run (p, U, k, omega, nut, cell data) as VTK: `vtk_medium_alpha4/` (open the `.vtm` in
ParaView); solver and mesh logs: `log.*.medium`.

## `reference/`: XFOIL polars

`xfoil_<shape>_<transition>.pol`: XFOIL polar files, alpha -2 ... 14 deg (points where XFOIL did not converge are
absent); `xfoil_cp_alpha4_tripped.txt`: Cp(x) at 4 deg; `xfoil_summary.json`: all of it as JSON. Made by
`simulation/xfoil_reference.py`.

## `campaign/`: the existing 436-run OpenFOAM campaign

`campaign_runs.csv`: one row per run: id, set (`lhs` = 100 Latin-hypercube airfoils x 4 angles; `reference` = NACA
0012, 2412, 4412, 4415 x 9 angles), the 6 CST weights, thickness, alpha, Re, Cl, Cd, L/D and the convergence status
(330 converged, 74 steady-ish, 32 unsteady near stall, 0 failed). Same grid and solver as the baseline, 1200
iterations. `campaign_overview.png`: Cl-alpha, drag polar, L/D against thickness.

`sample_fields/`: three campaign runs repeated with their full fields kept as VTK (`s012_a1`: alpha 2.9 deg,
`s047_a2`: 9.2 deg, `NACA4412_a+08`: 8 deg). `rerun_check.json`: the re-runs reproduce the stored Cl and Cd to
2 x 10^-5 and 0.001 %.

## `surrogate/`

`surrogate.metrics.json`: test metrics of the two surrogates trained on the campaign (plus 54 verification runs):
MLP ensemble (5 networks; held-out airfoils: Cl MAE 0.003, Cd mean error 0.5 %, 90th percentile 1.0 %, Cm MAE
0.0005) and MeshGraphNet on the CFD grid (held-out: Cl MAE 0.011, Cd mean error 1.5 %; fields p, U, nut). The
trained model is `apps/aero_optimizer/model/surrogate.pt` in this repository (not copied here).

## `variants/`: the geometry variants at alpha 4 deg (campaign grid)

| Variant | t/c | Cl | Cd | Cm | L/D |
|---|---|---|---|---|---|
| baseline_naca2412 | 0.120 | 0.6625 | 0.010143 | -0.047 | 65.3 |
| variant_a_naca4412 | 0.120 | 0.8739 | 0.011097 | -0.093 | 78.7 |
| variant_b_aft_camber | 0.120 | 0.7348 | 0.010411 | -0.069 | 70.6 |
| variant_c_thick | 0.150 | 0.6523 | 0.010769 | -0.044 | 60.6 |

## `optimisation/`: optimised designs and their CFD verification

`optimised.json`, `optimised_shapes.png`: maximum L/D at alpha 4 deg found with the MLP-ensemble surrogate
(differential evolution inside the campaign bounds, `simulation/optimise_ld.py`), each optimum then run in OpenFOAM
on the campaign grid:

| Design | Constraints | t/c | Surrogate L/D | CFD Cl | CFD Cd | CFD Cm | CFD L/D | surrogate vs CFD |
|---|---|---|---|---|---|---|---|---|
| baseline_naca2412 | | 0.120 | 65.1 | 0.6625 | 0.010143 | -0.047 | 65.3 | -0.3 % |
| opt_A_brief | t/c >= 0.12, abs(Cm) <= 0.10 (the brief) | 0.120 | 82.6 | 0.9229 | 0.011204 | -0.098 | 82.4 | +0.3 % |
| opt_B_low_moment | t/c >= 0.12, abs(Cm) <= 0.06 | 0.120 | 76.7 | 0.8302 | 0.010932 | -0.059 | 75.9 | +1.0 % |
| opt_C_thick | t/c >= 0.15, abs(Cm) <= 0.10 | 0.150 | 77.6 | 0.9385 | 0.012009 | -0.090 | 78.2 | -0.8 % |

In every optimum some weights sit on their bounds (opt_A: M0, M1, M2 at the upper bound, T2 at the lower one) and the
moment constraint is active, so the bounds limit the result; going further means leaving the surrogate's range.

opt_A_brief on three grids (`baseline/baseline_summary.json`, `optimum_grid_study`):

| Grid | Cl | Cd | Cm | L/D | gain over the NACA 2412 on the same grid |
|---|---|---|---|---|---|
| coarse | 0.9247 | 0.012268 | -0.099 | 75.4 | +26.7 % |
| medium | 0.9229 | 0.011204 | -0.098 | 82.4 | +26.1 % |
| fine | 0.9239 | 0.011276 | -0.098 | 81.9 | +22.6 % |

The drag of the optimum converges non-monotonically (coarse to medium -8.7 %, medium to fine +0.6 %), so the GCI
formula does not apply cleanly; the medium-fine difference (0.6 % in Cd, 0.5 % in L/D) is the better estimate of
the grid error, and the gain over the baseline is +22 to +27 % on every grid.

`aircraft_search_verification.json`, `aircraft_pareto_picks_cfd.csv`: the six designs of the earlier aircraft-level
search of `apps/aero_optimizer` (light-aircraft objectives: top speed, CO2, stall speed; 11 % minimum thickness),
each run in OpenFOAM over -2 ... 14 deg; at 4 deg they reach L/D 81.4-82.7 in CFD (within 0.6 % of the surrogate),
at 11 % thickness, below this brief's 12 % limit.
