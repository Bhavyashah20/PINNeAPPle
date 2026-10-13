---
title: "Geometry encoders (PointNet / DGCNN; B-rep UV-Net later)"
labels: [enhancement, geometry, surrogate]
---

**Who has it:** Neural Concept, Simcenter PhysicsAI (geometric deep learning).

**What PINNeAPPle already has:** `geometry_features`, `geometry_intelligence` (hand-crafted descriptors).

**Proposal:** PointNet/DGCNN encoders in `pinneapple_neural/architectures`, used by the similarity score and the raw-geometry surrogate.

**Acceptance (rule D1 — independent reference):** Permutation invariance tested; accuracy reported on a ModelNet10 subset.

**Origin:** §2 item 11
