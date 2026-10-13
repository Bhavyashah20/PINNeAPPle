---
title: "Pipeline 'CFD cases → field surrogate from STL / point cloud' + a real DoMINO"
labels: [enhancement, surrogate]
---

**Who has it:** Ansys SimAI (non-parametric), Simcenter PhysicsAI, Luminary SHIFT (DoMINO/GeoTransolver), Emmi AB-UPT.

**What PINNeAPPle already has:** `Transolver`, MeshGraphNet, `physicsnemo_bridge`, `lpm.py`. **Note:** `pinneapple_physics/pinn_solver/domino.py` is a domain-decomposition PINN, not NVIDIA's DoMINO — the name is misleading.

**Proposal:** Ready-made pipeline: folder of OpenFOAM cases → dataset (surface + volume) → training (Transolver / DoMINO via the bridge) → fields + Cd/Cl → report. Rename or clearly document the current `domino.py`.

**Acceptance (rule D1 — independent reference):** Reproducible AhmedML benchmark (CC-BY-SA) reporting the Cd error and comparing it with a published number.

**Origin:** §2 item 3
