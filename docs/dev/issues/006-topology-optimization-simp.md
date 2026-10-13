---
title: "SIMP topology optimization (2-D compliance) on top of solid_fem"
labels: [enhancement, design-optimization]
---

**Who has it:** Ansys Mechanical/Discovery, STAR-CCM+ adjoint topology optimization, SU2, OpenFOAM adjointOptimisation.

**What PINNeAPPle already has:** Absent.

**Proposal:** SIMP with density filter and OC/MMA updates, 2-D compliance, on `solid_fem.py`.

**Acceptance (rule D1 — independent reference):** Reproduce Sigmund's MBB beam ('99-line code', 2001): final compliance within 1% of the published value on the same mesh.

**Origin:** §2 item 6
