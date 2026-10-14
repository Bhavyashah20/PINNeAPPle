---
title: "RBF / Kriging / EBF approximation models with training-set outlier detection"
labels: [enhancement, surrogate]
---

**Who has it:** Dassault Isight approximation models (RSM, Kriging, RBF, EBF), COMSOL surrogate training (GP, PCE, DNN), 3DEXPERIENCE outlier exploration.

**What PINNeAPPle already has:** NumPy GP in `design_optimizer`; no RBF surrogate; no outlier screening of training data.

**Proposal:** RBF interpolant (thin-plate/multiquadric with polynomial tail), ordinary Kriging, leave-one-out error, and an outlier score per training sample.

**Acceptance (rule D1 — independent reference):** RBF reproduces polynomials of the tail degree exactly; LOO error of Kriging on the Branin function within published ranges; an injected corrupted sample is ranked first by the outlier score.

**Origin:** mapping 2026-10-12: SIMULIA / COMSOL
