---
title: "Polynomial chaos expansion + Sobol indices for UQ and sensitivity"
labels: [enhancement, uq]
---

**Who has it:** optiSLang, COMSOL Uncertainty Quantification Module.

**What PINNeAPPle already has:** Conformal / ensemble / MC-dropout; no PCE.

**Proposal:** Regression PCE (Hermite/Legendre) with Sobol indices computed analytically from the coefficients.

**Acceptance (rule D1 — independent reference):** Ishigami function: first-order and total Sobol indices within 1% of the closed-form values.

**Origin:** §2 item 10
