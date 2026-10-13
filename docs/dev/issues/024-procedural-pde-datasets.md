---
title: "Procedural PDE datasets on the fly (APEBench-style) using the new ETDRK4"
labels: [enhancement, data, benchmark]
---

**Who has it:** APEBench (MIT): datasets are generated procedurally by a spectral ETDRK reference solver (Exponax) instead of downloaded; scenarios are parameterized by normalized 'difficulty' coefficients; autoregressive rollout metrics.

**What PINNeAPPle already has:** `numerical_solvers/etdrk4.py` (S27), PDE presets, `field_metrics`.

**Proposal:** `pinneapple_data.procedural`: periodic 1/2/3-D scenarios (advection, diffusion, Burgers, KdV, KS, Gray–Scott, Swift–Hohenberg) with seeded random ICs, generated on demand; difficulty parameterization γ_j = α_j N^j 2^(j-1) D with α_j = a_j Δt / L^j (Exponax `extract_normalized_coefficients_from_difficulty`); rollout nRMSE/VRMSE curves.

**Acceptance (rule D1 — independent reference):** Deterministic: same seed → identical dataset; linear scenarios match the exact Fourier solution to 1e-10; cross-check against Exponax trajectories when available.

**Origin:** mapping 2026-10-12: APEBench / Exponax
