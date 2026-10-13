---
title: "MultiAdam optimizer for multi-term PINN losses"
labels: [enhancement, training]
---

**Who has it:** PINNacle `src/optimizer/multiadam.py` (MIT).

**What PINNeAPPle already has:** NTK weighting and Adam → L-BFGS exist; MultiAdam does not.

**Proposal:** Separate Adam moment statistics per loss group.

**Acceptance (rule D1 — independent reference):** Poisson with badly balanced boundary terms: lower relative error than plain Adam over 3 seeds.

**Origin:** ledger: PINNacle
