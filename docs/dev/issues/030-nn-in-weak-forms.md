---
title: "Neural networks inside variational forms (firedrake.ml-style) in the FEniCS bridge"
labels: [enhancement, hybrid, fem]
---

**Who has it:** Firedrake (LGPL-3.0, clean-room): `fem_operator` / `ml_operator` embed PyTorch/JAX models in UFL forms with adjoints.

**What PINNeAPPle already has:** `fenics_bridge`, hybrid surrogates; no NN term inside a weak form.

**Proposal:** External-operator term in a FEniCS form evaluated by a torch model, with the adjoint provided by torch autograd (method from the published paper, not Firedrake code).

**Acceptance (rule D1 — independent reference):** Poisson with a learned source equal to the exact source: FE solution matches the manufactured solution at the FE discretisation error; gradient check against finite differences.

**Origin:** mapping 2026-10-12: Firedrake
