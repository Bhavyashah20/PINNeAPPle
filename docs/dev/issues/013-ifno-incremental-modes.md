---
title: "Incremental FNO (iFNO): grow Fourier modes during training"
labels: [enhancement, neural-operators]
---

**Who has it:** neuraloperator `IncrementalFNOTrainer` (MIT); George et al., TMLR 2024.

**What PINNeAPPle already has:** FNO in `architectures/neural_operators/fno.py` with fixed modes.

**Proposal:** Training callback (loss-gap and explained-variance rules) that increases the FNO `modes`.

**Acceptance (rule D1 — independent reference):** 1-D Burgers: same final accuracy as fixed-mode FNO with fewer training FLOPs (fixed seed).

**Origin:** ledger: neuraloperator
