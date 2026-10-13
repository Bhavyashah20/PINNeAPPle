---
title: "NVIDIA Warp backend: warp.fem weak forms + zero-copy torch interop"
labels: [enhancement, gpu, solvers]
---

**Who has it:** NVIDIA Warp 1.18 (Apache-2.0): `@wp.kernel` with `wp.Tape` autodiff, `warp.fem` integrands on grid/tet/hex geometries, `wp.to_torch` via DLPack.

**What PINNeAPPle already has:** Warp only indirectly through the Newton bridge; ROADMAP §3 lists Warp as a candidate differentiable backend.

**Proposal:** `compute_backends.warp`: zero-copy tensors, a Poisson/elasticity example in warp.fem, differentiable particle step for `particle_dynamics`.

**Acceptance (rule D1 — independent reference):** warp.fem Poisson matches the manufactured solution with O(h^2) convergence; Warp vs torch gradients agree to 1e-6.

**Origin:** mapping 2026-10-12: NVIDIA Warp
