---
title: "Adjoint: otimizador MMA e parametrização B-spline volumétrica"
labels: [enhancement, design-optimization]
---

**Quem tem:** OpenFOAM ESI adjointOptimisation, SU2 (RBF, Hicks-Henne).

**O que o PINNeAPPle já tem:** `ContinuousAdjointSolver`, `ShapeParametrization`.

**Proposta:** MMA (Svanberg 1987) e caixas B-spline volumétricas.

**Critério de aceite (regra D1 — referência independente):** MMA reproduz o ótimo do problema-teste de Svanberg; B-spline preserva volume em deformação nula.

**Origem:** §3
