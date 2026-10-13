---
title: "Viewer with hash-keyed inference cache and surrogate-vs-truth diff"
labels: [enhancement, frontend, twin3d]
---

**Who has it:** NVIDIA Omniverse digital-twins-for-fluid-simulation blueprint (SHA256-keyed .npz cache).

**What PINNeAPPle already has:** `twin3d` viewer, `pinneapple_app`.

**Proposal:** Re-infer on every parameter change, cache by input hash, diff tab against ground truth.

**Acceptance (rule D1 — independent reference):** Same input → same hash and a cache hit; zero diff when surrogate equals truth.

**Origin:** §2 item 15
