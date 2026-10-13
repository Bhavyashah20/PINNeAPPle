---
title: "Adjoint: MMA optimizer and volumetric B-spline parameterization"
labels: [enhancement, design-optimization]
---

**Who has it:** OpenFOAM ESI adjointOptimisation, SU2 (RBF, Hicks–Henne).

**What PINNeAPPle already has:** `ContinuousAdjointSolver`, `ShapeParametrization`.

**Proposal:** MMA (Svanberg 1987) and volumetric B-spline morphing boxes.

**Acceptance (rule D1 — independent reference):** MMA reproduces the optimum of Svanberg's test problem; zero control-point displacement leaves the geometry unchanged.

**Origin:** §3
