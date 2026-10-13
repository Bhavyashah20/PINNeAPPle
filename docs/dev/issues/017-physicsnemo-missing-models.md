---
title: "PhysicsNeMo models not yet exposed: FIGConvNet, TopoDiff, DLWP-HEALPix"
labels: [enhancement, neural-operators]
---

**Who has it:** NVIDIA PhysicsNeMo (Apache-2.0).

**What PINNeAPPle already has:** `physicsnemo_bridge` exists; these models are not exposed.

**Proposal:** Expose through the bridge first (no reimplementation); reimplement only where justified.

**Acceptance (rule D1 — independent reference):** Forward smoke test + one small training case per model.

**Origin:** ledger: PhysicsNeMo
