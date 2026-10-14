# Choosing a method: PINN, neural operator, graph network or the solver

PINNeAPPle covers the whole Physics-AI workflow: geometry, simulation campaigns, data-driven surrogates (neural
operators, graph networks, MLPs), physics-informed networks, optimisation on top, and verification of the result.
Which learned method to use depends on the problem, not on the library. This page gives the decision table, the
evidence behind it, and the entry point for each choice.

The evidence is an equal-budget benchmark: every method gets the same training compute, the same number of
hyper-parameter trials and the same hardware, and the cost of the simulation data is counted separately
([PINNeAPPle-Benchmark, experiment 09](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet)).
Its budgets are small (one CPU core per run, no GPU), so read its numbers as directions, not as the best each method
can do.

## Decision table

| Your situation | Use | Why (benchmark evidence below) | Entry point |
|---|---|---|---|
| **One forward problem**, a solver exists for it | The numerical solver | Faster than training any model for one instance (63 ms to 56 s here against 120-900 s of training) and exact to its discretisation | [Solvers and `pp.cfd`](#solvers-cfd-and-campaigns) |
| **No solver or mesh** for the equations or the geometry, one instance, a smooth solution | PINN | Needs only the equations; expect percent-level errors at best in modest budgets (16-25 % on Burgers and Navier-Stokes here, 87 % on Darcy) | [PINN factory](#pinn-factory) |
| **Coefficients with jumps, shocks, turbulence**, no data | The numerical solver | The PINNs barely learned or did not converge on these (Darcy rel. L2 0.87; the RANS airfoil 0.96, no better than predicting zero, at any budget up to 4 h) | [Solvers and `pp.cfd`](#solvers-cfd-and-campaigns) |
| **Simulation data available** (or affordable), many new inputs on a grid | Neural operator: FNO first, DeepONet second | FNO: 24-150x more accurate than the PINN at the same training compute, also at equal FLOPs and out of distribution; milliseconds per query; the data repays itself after 1-14 queries | [Neural operators](#neural-operators) |
| **Many geometries with a low-dimensional parametrisation** (CST, a few design variables), fields needed | POD + MLP on the parameters | 0.4 % velocity error on held-out airfoils (MeshGraphNet 17 %), 55 µs per design | [MLP surrogates](#mlp-surrogates) |
| **Many geometries without a shared parametrisation**, fields on a mesh | Graph network (MeshGraphNet; #293 for Transolver, GINO, point clouds) | Works on any mesh from one model; on a parametrised family it was beaten by POD + MLP | [MeshGraphNet](#meshgraphnet-and-geometry-surrogates) |
| **Many geometries**, only integrals needed (Cl, Cd, peak stress) | MLP on the shape parameters | Cd within 0.3 % in distribution, 0.7 % outside the design space; a scalar regressor beats field models on scalars | [MLP surrogates](#mlp-surrogates) |
| **Inverse problem** (a coefficient from sparse sensors), few cases | Calibration through the solver; a PINN only without a solver | Classical calibration: 0.9 % error in h; the PINN: 11 % in range, 39 % out of range | [Inverse problems](#inverse-problems) |
| **Inverse problem**, many cases inside a known range, answers in a fraction of a second | Surrogate + calibration | The classical calibration's accuracy (1 %) at 1-7 % of its cost per case; it held 2-4.5 % outside its training range | [Inverse problems](#inverse-problems) |
| **Fields** versus **integrals** | Fields: operator, POD + MLP or a graph network. Integrals only: MLP on the parameters | Integral targets are low-dimensional and easier | - |
| **Regime**: turbulent (RANS/LES), separated flow | Solver for the data, then a data-driven surrogate | The data-free RANS PINN collapsed to the zero-flow solution of its own equations; a closure without data cannot place the boundary layer | [Solvers and campaigns](#solvers-cfd-and-campaigns) |

Two costs decide between "train once on data" and "solve each instance":

- **Amortisation.** A surrogate pays for its data and training once, then answers each query in milliseconds. A
  per-instance PINN pays its training for every new instance. `pinneapple_analysis.cost.AmortizedCost` gives the
  cost per query and the break-even number of queries.
- **Extrapolation.** A surrogate is only as good as its training range; a PINN or the solver has no training range.
  Check the out-of-distribution rows below before trusting a surrogate outside its data.

## The evidence

Equal-budget benchmark, [PINNeAPPle-Benchmark experiment 09](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet),
run with `pp.benchmark` ([Running a benchmark](running_a_benchmark.md)). **CPU only**: one core per training run,
300 s (Burgers, Darcy, Navier-Stokes), 900 s (airfoil), 120 s (inverse heat); the same 4-trial search for every method;
3 seeds; n = instances in each cell.

| Problem (n test / n OOD) | Best data-driven, error in / out of distribution | PINN, same instances | Reference solver per instance |
|---|---|---|---|
| Burgers 1-D, rel. L2 (3 / 2) | FNO 0.00168 / 0.0393 | 0.246 / 0.592 | 1 s |
| Darcy 2-D, rel. L2 (3 / 2) | FNO 0.0167 / 0.0138 | 0.873 / 0.891 | 63 ms |
| Navier-Stokes 2-D, rel. L2 (3 / 2) | FNO 0.0068 / 0.0107 | 0.16 / 0.507 | 3 s |
| Airfoil, Cd rel. error (4 / 12; PINN 4 / 4) | MLP 0.295 % / 0.739 % | RANS PINN 89.8 % / 87.7 % | OpenFOAM 56 s |
| Airfoil, velocity rel. L2 (4 / 12; PINN 4 / 4) | POD + MLP 0.00383 / 0.0179 | potential flow 0.833 / 0.856; RANS 0.96 / 0.98 | OpenFOAM 56 s |
| Inverse heat, h rel. error (3 / 2) | surrogate + calibration 1.01 % / 2.06-4.48 % | 11.4 % / 39.3 % | FV + calibration 9 s: 0.946 % / 3.31 % |

What it does not show: GPU budgets (several methods were still improving), a wider search (the airfoil setup audit
shows the potential-flow PINN gains from other loss weights and a lower learning rate), PINNs with data, and a
Darcy OOD set that is really out of distribution (the library's data gate flagged it). The experiment's README lists
every limitation and every check the library ran.

## Entry points

### PINN factory

`pinneapple_physics.pinn_solver.factory.pinn_factory`: `PINNProblemSpec` (equations as strings, conditions, inverse
parameters, exogenous signals), `PINNFactory(spec).generate_loss_function()`, `PINN` and `NeuralNetwork`. Hard
initial conditions from gridded data: `pinneapple_simulation.numerical_solvers.operator_benchmarks.FourierInterpolant`.
See [PINN / Physics](../core_concepts/pinn.md).

### Neural operators

`pinneapple_neural.architectures.neural_operators`: `fno.FourierNeuralOperator` (1-D), `fno.FNO2d`, `fno.FNO3d`,
`deeponet.DeepONet`. Reference data for the standard problems: `operator_benchmarks.burgers_1d`,
`navier_stokes_2d_vorticity`, `darcy_2d`, `periodic_grf`.

### MeshGraphNet and geometry surrogates

`pinneapple_neural.architectures.graphnn.mesh_graph_net.MeshGraphNet`; the aero app's field-and-coefficient model
`pinneapple_design.aero.surrogate.AeroGNN`. Geometry-agnostic surrogates (Transolver, GINO, point-cloud baselines
behind one interface): issue #293.

### MLP surrogates

`pinneapple_design.aero.surrogate.MLPEnsemble` (shape parameters + angle → Cl, Cd, Cm, with an ensemble spread);
`pinneapple_neural.architectures.neural_operators.deeponet.mlp` for a plain MLP.

POD + MLP for fields over a parametrised family: `pinneapple_analysis.benchmark.builtin.airfoil_impl.train_mlp_fields`
(POD of the training fields, an MLP from the parameters to the POD coefficients), the benchmark's `mlp_fields` method.

### Inverse problems

`pinneapple_analysis.inverse_problems.heat_transfer`: `run_fin`, `run_plate` (PINN with h as an inverse parameter;
`run_plate(budget=...)` trains under a compute budget). Calibration through a solver or a surrogate:
`pinneapple_simulation.numerical_solvers.heat_fv_reference.plate_2d` plus any 1-D search.

### Solvers, `pp.cfd` and campaigns

`pp.cfd` (`pinneapple_simulation.numerical_solvers.external_flow`): OpenFOAM around any body, `FlowResult`. The
campaign runner `pinneapple_simulation.external_solvers.openfoam.campaign` runs one verified case many times and
returns results a surrogate can train on ([OpenFOAM parameter campaigns](../core_concepts/openfoam_campaigns.md)).
The 2-D airfoil campaign of the benchmark is `apps/aero_optimizer/tools/generate_dataset.py`.

### Comparing methods yourself

`pinneapple_analysis.cost`: `ComputeBudget`, `train_within_budget` (stops at the budget, budget-aware schedules),
`count_flops` (works around `torch.autograd.grad`), `equal_budget_search` (the same number of trials per method),
`AmortizedCost`, `relative_l2`, `time_per_query`, `hardware_record`.

```python
from pinneapple_analysis.cost import ComputeBudget, train_within_budget, cosine_lr, set_lr

budget = ComputeBudget(wall_s=300)            # or flops=..., steps=...

def step(progress):                           # progress in [0, 1]: the fraction of the budget used
    set_lr(opt, cosine_lr(progress, 1e-3))
    opt.zero_grad()
    loss = loss_fn(model, next_batch())
    loss.backward()
    opt.step()
    return loss.detach()

run = train_within_budget(step, budget)       # run.steps, run.wall_s, run.flops, run.history
```
