---
title: "ParaView Catalyst 2 in-situ adaptor for solvers and surrogates"
labels: [enhancement, visualization]
---

**Who has it:** ParaView 6.2 (BSD-3-Clause, Copyright.txt): Catalyst 2 with Conduit Blueprint; NekRS/FluidX3D/M-Star render while running.

**What PINNeAPPle already has:** PyVista/VTK export, trame, Rerun bridge; nothing streams fields in situ from the native solvers.

**Proposal:** Optional `catalyst` adaptor publishing mesh + fields as Conduit Blueprint every N steps from `fvm`, `lbm`, `stable_fluids` and surrogate rollouts.

**Acceptance (rule D1 — independent reference):** Blueprint verification (`conduit.blueprint.verify`) passes for each published step; field values round-trip exactly.

**Origin:** mapping 2026-10-12: ParaView
