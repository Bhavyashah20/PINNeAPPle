---
title: "Onshape connector: sample configuration parameters and export geometry for DOE"
labels: [enhancement, cad, interop]
---

**Who has it:** Onshape REST API (configurations, encodeConfigurationMap, STL/STEP/Parasolid export); Luminary × Onshape partner app generating SHIFT-Wing variants.

**What PINNeAPPle already has:** STEP via gmsh, CadQuery; Onshape appears only in the resource catalog.

**Proposal:** `pinneapple_design.geometry.io.onshape`: read configuration parameters, LHS over them, export STL per variant, record document/version IDs as dataset lineage.

**Acceptance (rule D1 — independent reference):** Mocked API round-trip test; on a public demo document, two different configurations produce meshes with different volumes matching Onshape's mass properties.

**Origin:** mapping 2026-10-12: Onshape
