# Faster implicit pressure solves in OpenFOAM

`pinneapple_simulation.accelerated_foam` makes the pressure solve of an OpenFOAM run cheaper. It changes only
the **initial guess** of the linear solve. OpenFOAM's own solver (PCG, GAMG, ...) still iterates, with the same
tolerance and the same stopping test, so the converged fields pass the same convergence test as a plain run
(section 2 shows why an equal residual is not always an equal error, and compares at matched accuracy). Every
number on this page comes from a run made with the code in this package, and each one is compared with plain
OpenFOAM on the same case at the same settings.

## Where the time goes

In a pressure-based solver (PISO in `icoFoam`/`pisoFoam`, PIMPLE in `pimpleFoam`/`interFoam`, SIMPLE in
`simpleFoam`) every corrector assembles and solves a Poisson-type equation for the pressure (`p`, or `p_rgh`
with gravity): `fvm::laplacian(rAU, p) == fvc::div(phiHbyA)`. The matrix is symmetric and negative
definite, it is new at every corrector because `rAU` changes, and it is solved by PCG (DIC or FDIC
preconditioner) or GAMG. Momentum, turbulence and phase-fraction equations are much cheaper to solve. They are
diagonally dominant and need a few smoother sweeps. The pressure equation is elliptic, so its iteration count
grows with the mesh.

The logs of the runs below show this directly. On the 128 x 128 cavity with PCG-DIC, the pressure needed
34 239 iterations over 256 solves, while the velocity components needed 1 478 + 1 492 symmetric Gauss-Seidel
sweeps over 128 solves each. At 256 x 256 the figures were 74 261 against 2 540 + 2 579. Each pressure
iteration also costs more than a smoother sweep, because it adds a preconditioner application and two global
reductions. Profiling
studies by GPU-solver vendors also treat the pressure solve as the part worth offloading. One NVIDIA
presentation reports that the pressure solve is still "about 35% of total" *after* offloading it to AmgX
(OpenFOAM workshop 2020; we could not open the slides, so this figure comes from search excerpts). We found no
peer-reviewed CPU breakdown for PISO/PIMPLE that we could verify, so this page relies on our own logs.

## What exists, and what it costs to use

| Tool | What it does | Licence | Use here |
|---|---|---|---|
| **PETSc4FOAM** ([gitlab.com/petsc/petsc4foam](https://gitlab.com/petsc/petsc4foam), upstream develop.openfoam.com/modules/external-solver) | `ldu2csr` conversion and PETSc/Hypre solvers behind `solver petsc;` (BoomerAMG, GPU back-ends, matrix caching) | GPL-3.0-or-later (README and LICENSE) | Not used. It is a separate build against PETSc. |
| **AmgX** ([github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX)) | GPU algebraic multigrid and Krylov | BSD-3-Clause | Not used: there is no GPU here. |
| **amgx4Foam** ([github.com/SimFlowCFD/foamExternalSolvers](https://github.com/SimFlowCFD/foamExternalSolvers)) | OpenFOAM to AmgX bridge | No LICENSE file in that repository; the Cineca original was not reachable | Not used: the licence is unknown. |
| **Ginkgo** ([github.com/ginkgo-project/ginkgo](https://github.com/ginkgo-project/ginkgo)) | Portable sparse linear algebra (CUDA/HIP/SYCL/OpenMP), including mixed precision | BSD-3-Clause | Not used. |
| **OGL** ([github.com/hpsim/OGL](https://github.com/hpsim/OGL); Olenik et al., *Meccanica* 2024, [doi:10.1007/s11012-024-01806-1](https://doi.org/10.1007/s11012-024-01806-1)) | Ginkgo solvers as OpenFOAM `lduMatrix` solvers | GPL-3.0 | Not used. |
| **RapidCFD** ([github.com/Atizar/RapidCFD-dev](https://github.com/Atizar/RapidCFD-dev)) | OpenFOAM fork with everything on CUDA | GPLv3 | Not used. It is a fork, not a plugin. |
| **pyamg** | Smoothed-aggregation and classical AMG in Python | MIT | Optional, for the Python benchmarks |

None of these is a dependency of PINNeAPPle, and no code from them is copied into the repository. The
pinneWarmStart plugin is our own code. It is written against OpenFOAM's public `lduMatrix::solver` interface, the
same runtime-selection mechanism that PETSc4FOAM and OGL use. OpenFOAM is GPL-3.0-or-later: this repository
contains no OpenFOAM source, and the plugin binary you build links against your installed `libOpenFOAM.so`,
so that binary falls under OpenFOAM's licence if you distribute it.

**Ideas from the literature, and which ones were tried.** Learned multigrid and preconditioners: Greenfeld et
al. 2019 ([arXiv:1902.10248](https://arxiv.org/abs/1902.10248)), Luz et al. 2020
([arXiv:2003.05744](https://arxiv.org/abs/2003.05744)), Li et al. ICML 2023
([PMLR 202](https://proceedings.mlr.press/v202/li23e.html)), and Häusner et al. "Neural incomplete
factorization" ([arXiv:2305.16368](https://arxiv.org/abs/2305.16368)). Learned pressure predictions or initial
guesses: Tompson et al. 2017 ([arXiv:1607.03597](https://arxiv.org/abs/1607.03597)), CFD-GCN
([arXiv:2007.04439](https://arxiv.org/abs/2007.04439)), and a neural initial guess for pressure-velocity coupling
([arXiv:2409.07175](https://arxiv.org/abs/2409.07175)). Projection of a new right-hand side on earlier solutions:
Fischer 1998, *CMAME* 163:193-204, which PETSc implements as
[`KSPGUESSFISCHER`](https://petsc.org/main/manualpages/KSP/KSPGUESSFISCHER/). POD/ROM predictors as initial
guesses: Markovinović & Jansen 2006, *IJNME* 68:525-541. Mixed-precision iterative refinement: Carson & Higham
2018, *SISC* 40:A817. The arXiv and DOI pages could not be opened from the build environment, so these
references were checked through their abstracts and listings, not read in full.

We tried four of them, all measured below:

1. Fischer projection on past solutions, in C++ inside OpenFOAM.
2. POD of earlier runs.
3. A learned graph-network correction of the guess.
4. Mixed-precision CG.

We also tried AMG through pyamg, offline only.

## What this package does

```
OpenFOAM pressure corrector
   fvMatrix::solve -> lduMatrix::solver::New("pinneWarmStart")
      1. x_prev = incoming p (what plain OpenFOAM starts from)
      2. guess x0:  projection on the last k solutions  (C++, k mat-vecs)
                    or a Python guesser via named pipes (POD of earlier runs, learned network, ...)
      3. safeguard: keep x0 only if sum|b - A x0| < sum|b - A x_prev|
      4. inner OpenFOAM solver (PCG/DIC, GAMG, ...) from x0, same tolerance;
         relTol measured against the residual of x_prev, so a run stops where a plain run would
      5. store the solution; optionally write A, b, x_prev, x to disk
```

**`plugin/pinneWarmStart.C`** is the solver wrapper (about 500 lines). To use it, set this in `fvSolution`:

```
p
{
    solver          pinneWarmStart;
    innerSolver     PCG;            // or GAMG, with its usual entries
    preconditioner  DIC;
    tolerance       1e-6;
    relTol          0.05;
    guess           projection;     // none | projection | external
    nSnapshots      8;
    snapshots       all;            // all | converged: keep only solutions of relTol 0 solves
    applyTo         all;            // all | converged: change the guess of relTol 0 solves only
    dumpDir         "";             // write every system (lduMatrix arrays) to this directory
    pipeDir         "";             // guess external: request/reply FIFOs served by Python
}
```

Also load the library from `controlDict` with `libs ("<path>/libpinneWarmStart.so");`.

The **projection guess** is the Galerkin projection on the span of the last `k` solutions and the current
field: `x0 = X (X^T A X)^-1 X^T b`, built with the matrix and right-hand side of the *current* solve. Because
`x_prev` is in the span, the A-norm error of `x0` is never larger than that of `x_prev`. This is the default
guess that plain OpenFOAM uses. The cost is `k` matrix-vector products, `k(k+1)/2` dot products and a `k x k`
solve, which is a few inner iterations' worth. The C++ code does a symmetric elimination with diagonal
pivoting that drops linearly dependent snapshots. It runs in parallel, because the reductions use
`gSumProd`/`gSumMag` and the history is kept per processor.

**`relTolReference previous`** (the default) handles the case where OpenFOAM's `relTol` is relative to the
initial residual. With a better guess, the same `relTol` would ask for a smaller absolute residual. That makes
the solve more accurate but not cheaper, and the run would no longer be comparable with a plain run. The
wrapper therefore turns `relTol` into the absolute tolerance `max(tolerance, relTol * r(x_prev))`, the point
where a plain run stops.

**Building.** `build.build_plugin()` compiles the wrapper with the installed compiler against
`libOpenFOAM.so`:

- With a full OpenFOAM installation (`$FOAM_SRC/OpenFOAM/lnInclude`), it uses those headers. `wmake libso`
  also works with `plugin/Make`.
- With a binary-only package such as Ubuntu's `openfoam` (v1912), there are no headers. Use the source of
  the same version instead (`apt-get source openfoam` with a `deb-src` entry), passed as `openfoam_src` or
  `$PINNEAPPLE_OPENFOAM_SRC`. The headers are only read at build time.

**Python side:**

- **`ldu`**: reads the dumped systems and converts them to SciPy CSR or PyTorch sparse matrices. It also
  computes OpenFOAM's normalised residual (`normFactor`).
- **`solvers`**: PCG with Jacobi, DIC (the same algorithm as OpenFOAM's), or pyamg AMG preconditioners, plus
  a PyTorch CG. The PyTorch CG runs in float64, or in mixed precision with float32 inner CG and float64
  refinement. All of them use OpenFOAM's stopping test.
- **`warm_start`**: `ProjectionWarmStart` (the same algorithm as the C++ code) and `PODWarmStart` (a POD of
  earlier runs plus recent solutions).
- **`learned`**: `LearnedWarmStart`, the projection plus a graph-network correction. It is trained on dumped
  systems and safeguarded.
- **`guess_server`**: serves any of these guesses to a running solver through named pipes.
- **`case`**: writes the cavity case, runs and parses logs, and compares fields.
- **`benchmark`**: the comparisons on this page, including `tutorial_comparison` for an installed tutorial.

`FreeSurfaceFlow` (`numerical_solvers/free_surface.py`) accepts `pressure_solver=` and `libs=`, so its
`p_rgh` solve can be warm-started.

## Verification

**Extraction is exact.** A run with `guess none; dumpDir systems;` writes every pressure system as OpenFOAM
handed it to PCG. Re-solving those systems in Python with the DIC-PCG of `solvers.pcg`, starting from the
dumped `x_prev`, gives:

- the same iteration count as the OpenFOAM log, solve by solve;
- the same initial residual to about 1e-15 relative;
- a solution within about 1e-14 (relative L2) of OpenFOAM's.

`test_openfoam_dump_is_reproduced_by_python` checks this on a 16 x 16 cavity.

**Fields are unchanged within solver tolerance.** Every comparison runs three versions of the same case:

1. plain OpenFOAM;
2. the same inner solver wrapped by `pinneWarmStart`;
3. a reference whose pressure is solved to `tolerance 1e-11; relTol 0` (1e-12 for damBreak).

The tables report the relative L2 difference of the final fields between runs. "vs plain" is the warm-started
run against plain OpenFOAM. "vs ref" measures each run against the tightly converged reference. The plain
run's own "vs ref" value is the size of the error that the solver tolerance allows.

All runs used one core of the same machine (4-core VM, no GPU) with OpenFOAM v1912 from the Ubuntu 24.04
package. Wall times are the best of two runs of the solver process.

### 1. Lid-driven cavity, PISO (`icoFoam`), standard settings

Settings: Re = 100, Courant number 0.5, 128 time steps, 2 PISO correctors. Pressure tolerance 1e-6 with
`relTol 0.05`, and `relTol 0` for `pFinal`. `pinneWarmStart` uses projection on the last 8 solutions around the
same inner solver.

| case | inner | run | p solves | p iterations | wall [s] | dp vs plain | dU vs plain | dp vs ref | dU vs ref |
|---|---|---|---|---|---|---|---|---|---|
| cavity 128x128 | PCG-DIC | plain | 256 | 34 239 | 13.11 | - | - | 2.8e-4 | 1.2e-7 |
| cavity 128x128 | PCG-DIC | **warm** | 256 | **11 197** | **7.19** | 2.6e-4 | 1.8e-7 | 3.8e-5 | 1.5e-7 |
| cavity 128x128 | GAMG | plain | 256 | 1 590 | 5.02 | - | - | 9.9e-4 | 1.2e-6 |
| cavity 128x128 | GAMG | **warm** | 256 | **501** | **4.32** | 9.8e-4 | 1.7e-6 | 1.8e-5 | 5.5e-7 |
| cavity 256x256 | PCG-DIC | plain | 256 | 74 261 | 112.8 | - | - | 3.2e-3 | 1.5e-7 |
| cavity 256x256 | PCG-DIC | **warm** | 256 | **30 041** | **61.3** | 3.1e-3 | 4.1e-7 | 5.8e-5 | 3.6e-7 |
| cavity 256x256 | GAMG | plain | 256 | 2 955 | 32.8 | - | - | 4.5e-3 | 1.3e-6 |
| cavity 256x256 | GAMG | **warm** | 256 | **979** | **25.9** | 4.4e-3 | 2.3e-6 | 3.9e-5 | 1.1e-6 |

How the speed-ups break down:

- **Iterations:** 3.1x fewer (PCG, 128), 2.5x fewer (PCG, 256), 3.2x fewer (GAMG, 128) and 3.0x fewer
  (GAMG, 256).
- **Whole-run wall time:** 1.83x and 1.84x with PCG, 1.16x and 1.27x with GAMG.

The pressure difference to plain OpenFOAM (2.6e-4 to 4.4e-3) is the same size as the plain run's own
distance to the tightly converged reference (2.8e-4 to 4.5e-3). That distance comes from `relTol 0.05`, not
from the warm start.

The warm-started run is 7-115x closer to the reference in pressure. This is because it stops at the same
absolute residual as plain but starts from a much better point in the non-final correctors. The velocity
agrees with the reference to 1e-6 or better in every run.

### 2. The same cavity with `relTol 0` everywhere: matched accuracy

With `relTol 0`, both runs stop at the same absolute normalised residual. Even so, the warm-started solution
carries more *algebraic* error. The projection leaves error in smooth, low-eigenvalue modes, which the L1
residual under-weights.

In this closed cavity, most of that error is the pressure level (the near-null mode fixed only by `pRefCell`).
Re-solving the dumped 64 x 64 systems offline against exact solutions (`spsolve`) gives a relative error of
5e-7 for plain and 4e-6 for warm at the same residual, mostly in the mean level. The fair comparison is
therefore at **equal accuracy**: tighten the warm start's tolerance until its error against the reference is
no larger than plain's.

| case | run | p iterations | wall [s] | dp vs ref | dp vs ref (mean removed) | dU vs ref |
|---|---|---|---|---|---|---|
| cavity 64x64 | plain PCG-DIC, tol 1e-6 | 11 796 | 1.34 | 9.2e-7 | 9.2e-7 | 1.2e-8 |
| cavity 64x64 | warm, tol 1e-6 | 5 054 | 0.89 | 4.1e-5 | 8.2e-6 | 3.1e-8 |
| cavity 64x64 | warm, tol 1e-7 | 7 370 | 1.09 | 4.2e-6 | 5.2e-7 | 3.4e-9 |
| cavity 64x64 | warm, tol 1e-8 | 9 616 | 1.33 | 1.1e-6 | 9.8e-8 | 4.7e-10 |
| cavity 128x128 | plain PCG-DIC, tol 1e-6 | 46 844 | 17.52 | 1.5e-6 | 1.5e-6 | 5.3e-9 |
| cavity 128x128 | warm, tol 1e-6 | 11 435 | 7.92 | 4.6e-5 | 3.3e-6 | 1.1e-8 |
| cavity 128x128 | warm, tol 1e-7 | 19 782 | 10.35 | 2.2e-6 | 2.7e-7 | 9.9e-10 |
| cavity 128x128 | **warm, tol 1e-8** | **29 227** | **13.36** | **2.5e-7** | **1.2e-8** | **1.2e-10** |

At equal or better accuracy in every field, the warm start is 1.31x faster on 128 x 128 (13.4 s vs 17.5 s,
tolerance 1e-8 vs 1e-6). On 64 x 64 it gives nothing (1.33 s vs 1.34 s). At the *same tolerance* it is 2.2x
faster on 128 x 128 (1.5x on 64 x 64), but with a 30x larger pressure error (still at the 1e-5 level) and the same velocity to 3e-8.

**Recommendation:** pair `pinneWarmStart` with a tolerance about 10-100x tighter than the plain run's when the
pressure itself is a quantity of interest. Settings with `relTol > 0` in the non-final correctors, the usual
PISO/PIMPLE practice (table 1), do not need this.

### 3. Free surface: `damBreak` (`interFoam`, `p_rgh`), refined 2x

The case is the v1912 tutorial, refined 2x to 9 072 cells, run to t = 0.25 s. It uses PIMPLE with 3 correctors.
`p_rgh` uses PCG-DIC with tolerance 1e-7 and `relTol 0.05`, and `relTol 0` for `p_rghFinal`. The reference
uses tolerance 1e-12 with `relTol 0`.

| run | p_rgh iterations | wall [s] | dp_rgh vs plain | dp_rgh vs ref | dU vs ref | dalpha vs ref |
|---|---|---|---|---|---|---|
| plain PCG-DIC | 23 355 | 9.24-10.03 | - | 1.5e-2 | 2.3e-2 | 7.8e-3 |
| warm, all solves | 31 964 | 12.32 | 1.0e-2 | 6.9e-3 | 1.0e-2 | 3.8e-3 |
| warm, `snapshots converged` | 25 429 | 10.82 | 7.8e-3 | 8.5e-3 | 1.5e-2 | 4.3e-3 |
| warm, `snapshots converged; applyTo converged` | 23 346 | 9.41 | 2.6e-7 | 1.5e-2 | 2.3e-2 | 7.8e-3 |

**No gain here.** The interface moves through the mesh and changes the density in the matrix from step to
step, so old solutions are a poor basis. In the second corrector, the projected guess has about the same
residual as the field it replaces, but its error is smoother: PCG needed 58 iterations where plain needed 12
for the same reduction (read from the logs).

Restricting the projection to the final solves (`applyTo converged`) makes the wrapper neutral (23 346 vs
23 355 iterations, fields identical to 3e-7). The overhead of the wrapper is then 1.8% of wall time (9.41 s vs 9.24 s).

The differences to the reference are at the 1e-2 level for every run, including plain. This transient is
sensitive to the pressure tolerance itself.

### 4. Offline: Python solvers and guesses on the same systems

The systems are the 128 pressure systems of a 64 x 64 cavity (lid speed 1, `relTol 0` everywhere), dumped by
`guess none`. Each is re-solved to tolerance 1e-6 starting from each guess. Times are Python times, which
are only comparable *within* this table: the NumPy DIC uses SciPy triangular solves and is far slower per
iteration than OpenFOAM's C++.

The POD basis comes from the 256 systems of two other runs, at lid speeds 0.8 and 1.2 (a two-point design
of experiments), and keeps rank 9 for 99.9999% of the energy. The last 8 solutions of the current run are
added to it.

The graph network has 4 layers and a width of 32. It was trained for 150 epochs (390 s on the CPU) on 128
systems of those two runs. Its mean training residual ratio went from 1.0 to 0.53.

| guess | solver | iterations | solve [s] | guess [s] | mean initial residual |
|---|---|---|---|---|---|
| previous field (OpenFOAM) | PCG-DIC | 11 796 | 31.8 | 0 | 1.9e-2 |
| previous field | PCG-Jacobi | 39 762 | 3.29 | 0 | 1.9e-2 |
| previous field | PCG-AMG (pyamg) | 695 | 3.72 | 0 | 1.9e-2 |
| previous field | Jacobi-CG, PyTorch f64 | 39 762 | 22.1 | 0 | 1.9e-2 |
| previous field | Jacobi-CG, PyTorch f32 + f64 refinement | 48 508 | 6.49 | 0 | 1.9e-2 |
| projection, last 8 | PCG-DIC | 2 743 | 8.92 | 0.12 | 1.2e-2 |
| projection, last 8 | PCG-AMG (pyamg) | 282 | 3.31 | 0.12 | 1.2e-2 |
| POD of 2 earlier runs + last 8 | PCG-DIC | **2 128** | 6.88 | 0.34 | 3.8e-4 |
| POD of 2 earlier runs + last 8 | PCG-AMG (pyamg) | 253 | 3.42 | 0.34 | 3.8e-4 |
| projection + learned correction | PCG-DIC | 2 683 | 7.85 | 1.99 | 8.7e-3 |
| projection + learned correction | PCG-AMG (pyamg) | 273 | 3.27 | 1.90 | 8.7e-3 |

What the table shows:

- **Projection.** It cuts DIC-PCG iterations 4.3x and AMG-PCG iterations 2.5x on the same systems. Every
  solution in this section agrees with OpenFOAM's to 7e-5 or better.
- **POD of earlier runs.** It removes another 22% of iterations relative to the projection alone. Its initial
  residual is 30x smaller.
- **Learned correction.** It removes 2% of DIC iterations relative to the projection alone, and its guess
  costs more than it saves. A local graph network acts like a learned smoother: it reduces the residual, but
  the iteration count is set by global, low-frequency error that it cannot see. We report it as **no
  practical gain**.
- **AMG (pyamg).** It cuts iterations about 17x relative to DIC. In Python, its setup on every new matrix
  dominates the time.
- **Mixed-precision PyTorch CG.** On the CPU it needs more iterations than float64 and is not faster
  per useful iteration. Without a GPU we make no GPU claim.

### 5. Python guesses inside a running OpenFOAM (`guess external`)

The 64 x 64 cavity from section 4, run by `icoFoam` with the guesses served through the pipes:

| run | p iterations | wall [s] | Python guess time [s] | dp vs ref | dU vs ref |
|---|---|---|---|---|---|
| plain PCG-DIC | 11 796 | 1.20 | - | 9.2e-7 | 1.2e-8 |
| projection in C++ (`guess projection`) | 5 054 | 0.90 | - | 4.1e-5 | 3.1e-8 |
| Python projection (`guess external`) | 4 999 | 1.34 | 0.29 | 3.4e-5 | 6.9e-8 |
| Python POD of earlier runs + last 8 | **4 237** | 1.35 | 0.43 | 3.9e-5 | 2.6e-8 |
| Python projection + learned correction | 5 088 | 3.82 | 2.71 | 1.9e-5 | 5.2e-8 |

The Python projection reproduces the C++ one (4 999 vs 5 054 iterations; the small difference comes from
round-off in the elimination order). This confirms that the coupling loop works end to end. POD of earlier
runs saves the most iterations, but on a 4 096-cell case the pipe and Python overhead outweighs the saving.
These Python models belong inside the C++ wrapper, or on much larger meshes where an iteration costs more
than a round trip.

### Summary of measured speed-ups

| setting | pressure iterations | wall time (whole run) |
|---|---|---|
| Cavity, PISO with `relTol 0.05`, PCG-DIC (128², 256²) | 3.1x, 2.5x fewer | **1.83x, 1.84x faster** |
| Cavity, PISO with `relTol 0.05`, GAMG (128², 256²) | 3.2x, 3.0x fewer | 1.16x, 1.27x faster |
| Cavity, `relTol 0`, matched accuracy, PCG-DIC (128²) | 1.6x fewer | 1.31x faster |
| Cavity, `relTol 0`, matched accuracy, PCG-DIC (64²) | 1.2x fewer | none |
| damBreak, interFoam `p_rgh` | none (worse unless restricted to final solves) | none |

## Limits

- The gain comes from time correlation. The projection helps when consecutive pressure systems are alike, as
  in transient PISO/PIMPLE runs with small time steps. For steady SIMPLE with `relTol 0.1`-`0.01`, each solve
  is short and the right-hand side changes a lot, so expect little.
- Equal residual does not mean equal error. At the same tolerance, a warm-started solution can have a
  larger algebraic error in smooth modes (section 2). Tighten the tolerance 10-100x when the pressure itself
  matters, and compare at matched accuracy, as done above.
- Multiphase cases with a moving interface (damBreak) did not benefit. Use `applyTo converged` there, or
  leave the plain solver.
- It works with any symmetric inner solver. With GAMG, iterations drop, but a GAMG cycle costs much more
  than the projection, so the wall-time gain is smaller (see the tables).
- `guess external` blocks the solver until the Python server answers. Start the `GuessServer` before the solver.
  For production use, prefer the in-process C++ projection; the pipes are for experimenting with Python models.
- Systems with coupled interfaces (processor, cyclic) are handled by the C++ wrapper. In Python, `LduSystem.to_csr`
  refuses them because the interface coefficients are not in the dump.
- No GPU was available, so the GPU paths (`torch_pcg` on CUDA) were only run on the CPU. No GPU speed-up is
  claimed.

## Reproducing

```python
from pinneapple_simulation.accelerated_foam import build, benchmark
lib = build.build_plugin(openfoam_src="/path/to/openfoam-OpenFOAM-v1912.200626")   # or with FOAM_SRC set
rows = benchmark.openfoam_comparison("work", lib, n=128, end_time=0.5, repeats=2)
print(benchmark.markdown_table(rows))
```

The tests are in `tests/test_accelerated_foam.py`. The OpenFOAM tests skip when `icoFoam` is missing or the
plugin cannot be built.
