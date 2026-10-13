---
title: "Differentiable implicit linear solves with matrix construction by tracing (PhiFlow-style)"
labels: [enhancement, solvers, autodiff]
---

**Who has it:** PhiFlow (MIT): sparse matrices built automatically by tracing a linear function; preconditioned solvers with implicit (adjoint) gradients; dimension-agnostic code.

**What PINNeAPPle already has:** Implicit steps in `fdm.py`/`fvm.py` assemble matrices by hand; gradients through solves are not implicit.

**Proposal:** A `linear_solve(f, rhs)` helper: trace `f` to a sparse matrix once, solve with CG/BiCGStab, backward pass by solving the transposed system (implicit differentiation), no unrolled iterations.

**Acceptance (rule D1 — independent reference):** Gradient of a Poisson-solve objective matches finite differences to 1e-6 and costs one extra solve; traced matrix equals the hand-assembled 5-point Laplacian exactly.

**Origin:** mapping 2026-10-12: PhiFlow
