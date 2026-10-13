---
title: "MAPDL .rst result reader and a PyFluent driver"
labels: [enhancement, interop]
---

**Who has it:** PyAnsys `pymapdl-reader` and `pyfluent` (MIT).

**What PINNeAPPle already has:** Readers for Abaqus, Exodus, CGNS, Fluent mesh, STAR-CCM+; Fluent driven by journal files.

**Proposal:** `.rst` reader (optional pymapdl-reader dependency) into UPD; a PyFluent path in `external_solvers/ansys`.

**Acceptance (rule D1 — independent reference):** pymapdl-reader's own example `.rst` file read with displacements equal to the reference library's.

**Origin:** ledger: PyAnsys
