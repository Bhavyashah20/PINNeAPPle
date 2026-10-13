---
title: "Return a 0–1 geometric similarity score with every surrogate prediction"
labels: [enhancement, surrogate, uq]
---

**Who has it:** Simcenter/Altair PhysicsAI return a *geometric similarity score*; Ansys SimAI returns a *confidence level* telling whether a new design lies inside the training domain.

**What PINNeAPPle already has:** `GeometryOODGuardrail`, `analysis/trust/trust_gate.py`, conformal/ensemble UQ in `pinneapple_analysis/uncertainty`.

**Proposal:** Geometry embedding (`geometry_features` descriptors + curvature from `geometry/ops/curvature.py`) → distance to the training set (kNN/Mahalanobis) → calibrated to [0, 1] on a hold-out split. Return it with every prediction and in the evidence report.

**Acceptance (rule D1 — independent reference):** Training designs score ≥ 0.9; a progressively deformed design gets a monotonically decreasing score; positive Spearman correlation between (1 − score) and the true error on a synthetic set.

**Origin:** docs/dev/ANALISE_FEATURES_COMERCIAIS.md §2 item 1
