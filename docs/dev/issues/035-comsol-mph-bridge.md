---
title: "COMSOL bridge through MPh (parameter sweeps → dataset)"
labels: [enhancement, interop]
---

**Who has it:** COMSOL 6.3/6.4 surrogate training, Model Manager; MPh (MIT, community) drives COMSOL's Java API from Python.

**What PINNeAPPle already has:** Nothing beyond mentions in the tool-recommendation tables.

**Proposal:** Optional bridge: open .mph, set parameters, solve, export scalars and fields to UPD/xarray; reuse the DOE samplers.

**Acceptance (rule D1 — independent reference):** Skipped without a COMSOL licence; with one, a parametric sweep of a shipped tutorial model reproduces COMSOL's own table to 1e-9.

**Origin:** mapping 2026-10-12: COMSOL
