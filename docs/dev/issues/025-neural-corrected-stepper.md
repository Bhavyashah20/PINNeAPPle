---
title: "Neural-corrected coarse solver trained through the rollout (solver-in-the-loop)"
labels: [enhancement, training, hybrid]
---

**Who has it:** APEBench `CorrectedStepper`; PhiFlow 'solver-in-the-loop' (Um et al., NeurIPS 2020).

**What PINNeAPPle already has:** Differentiable torch solvers (`fvm`, `lbm`, spectral), hybrid surrogates (`hybrid_surrogate_physics`).

**Proposal:** Wrap a coarse differentiable solver with a learned correction (sequential or parallel composition) trained by unrolling k steps.

**Acceptance (rule D1 — independent reference):** 1-D Burgers at 4× coarsening: corrected stepper error < 50% of the uncorrected coarse solver over a 100-step rollout (3 seeds).

**Origin:** mapping 2026-10-12: APEBench / PhiFlow
