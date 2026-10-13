---
title: "Solution-adaptive mesh refinement (quadtree, wavelet/gradient criterion)"
labels: [enhancement, meshing, solvers]
---

**Who has it:** CONVERGE (AMR every step), Basilisk (`adapt_wavelet`), Luminary SDK.

**What PINNeAPPle already has:** Static refinement only (boundary layers, curvature, snappy levels).

**Proposal:** 2-D quadtree with a wavelet criterion for `fvm` and `shallow_water_fv`.

**Acceptance (rule D1 — independent reference):** Ritter dam break: same L1 error as the fine uniform grid with ≤ 40% of the cells.

**Origin:** §2 item 7
