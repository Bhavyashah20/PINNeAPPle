---
title: "JAX-Fluids as an optional external solver (GPL: subprocess bridge, no code reuse)"
labels: [enhancement, interop, solvers]
---

**Who has it:** JAX-Fluids 2.0 (GPL-3.0): fully differentiable compressible two-phase solver (level-set / diffuse interface), level-set immersed boundaries, multi-GPU.

**What PINNeAPPle already has:** `jax_cfd_solver.py` (incompressible, Apache); no compressible two-phase solver.

**Proposal:** Out-of-process bridge (case JSON in, HDF5 out) behind an optional extra, never imported in-process or vendored; clean-room policy for any method taken from it (ledger).

**Acceptance (rule D1 — independent reference):** Sod shock tube run through the bridge matches the exact Riemann solution (L1 density error decreasing with resolution).

**Origin:** mapping 2026-10-12: JAX-Fluids
