---
title: "Nektar++ bridge (spectral/hp element, MIT) for high-order reference data"
labels: [enhancement, interop, solvers]
---

**Who has it:** Nektar++ 5.9 (MIT): CG/DG/HDG/flux-reconstruction spectral/hp, incompressible and compressible N–S, linear stability and adjoint, curvilinear high-order meshes.

**What PINNeAPPle already has:** Bridges to OpenFOAM, FEniCS, CalculiX, FluidX3D, MuJoCo…; no high-order solver.

**Proposal:** Session-XML generation from a `ProblemSpec`, runner (subprocess/Docker), field reader into UPD — same pattern as the OpenFOAM bridge.

**Acceptance (rule D1 — independent reference):** Kovasznay flow: L2 error converges spectrally with polynomial order (≥ 1 decade per +2 orders) against the exact solution.

**Origin:** mapping 2026-10-12: Nektar++ (https://en.wikipedia.org/wiki/Nektar%2B%2B)
