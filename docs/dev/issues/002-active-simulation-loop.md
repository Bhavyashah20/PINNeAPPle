---
title: "Closed loop: the surrogate picks the next simulation (acquisition → campaign → retrain)"
labels: [enhancement, active-learning, orchestration]
---

**Who has it:** HEEDS adaptive sampling, PhysicsX (UQ → new simulation), Neural Concept.

**What PINNeAPPle already has:** `pinneapple_core/data.py::ActiveSampler`, EI in `design_optimizer`, `external_solvers/openfoam/campaign.py`, `pinneapple_registry`.

**Proposal:** An orchestrator wiring acquisition function → `campaign` (OpenFOAM/LBM/any registry solver) → dataset store → retrain → next acquisition, stopping on validation error or budget.

**Acceptance (rule D1 — independent reference):** With a cheap solver (analytic function + `stable_fluids`), the active loop reaches the same error as an LHS of equal final size with ≤ 60% of the samples (mean over 5 seeds).

**Origin:** §2 item 2
