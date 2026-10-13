---
title: "Multi-fidelity surrogate (Kennedy–O'Hagan co-kriging)"
labels: [enhancement, surrogate]
---

**Who has it:** optiSLang AI+, HEEDS data reuse, COMSOL.

**What PINNeAPPle already has:** Transfer/meta-learning exist; explicit multi-fidelity does not.

**Proposal:** Autoregressive co-kriging on the NumPy GP in `design_optimizer` (e.g. coarse LBM + fine OpenFOAM).

**Acceptance (rule D1 — independent reference):** Forrester functions (closed form): co-kriging with 4 HF + 11 LF points beats an HF-only GP in RMSE.

**Origin:** §2 item 8
