---
title: "LBM D3Q27 cumulant collision + STL voxelization + FP16 storage"
labels: [enhancement, solvers, gpu]
---

**Who has it:** ultraFluidX, PowerFLOW, XFlow, M-Star, SimScale Pacefish, FluidX3D (FP16 storage).

**What PINNeAPPle already has:** `lbm.py`: D2Q9/D3Q19 BGK + Smagorinsky; `lbm_tunnel`.

**Proposal:** D3Q27 cumulant collision, STL voxelization (GPU via torch), FP16 storage with FP32 arithmetic (the idea only — FluidX3D is non-commercial, no code reuse).

**Acceptance (rule D1 — independent reference):** 3-D Poiseuille and Taylor–Green against exact solutions; sphere at Re = 100 with Cd within 3% of the Schiller–Naumann correlation.

**Origin:** §3
