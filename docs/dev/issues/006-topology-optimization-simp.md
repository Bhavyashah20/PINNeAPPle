---
title: "Otimização topológica SIMP (compliance 2-D) sobre o solid_fem"
labels: [enhancement, design-optimization]
---

**Quem tem:** Ansys Mechanical/Discovery, STAR-CCM+ adjoint topology, SU2, OpenFOAM adjointOptimisation.

**O que o PINNeAPPle já tem:** Ausente.

**Proposta:** SIMP com filtro de densidade e OC/MMA, 2-D compliance, sobre `solid_fem.py`.

**Critério de aceite (regra D1 — referência independente):** Reproduzir o MBB beam do Sigmund (2001, '99-line') — compliance final dentro de 1% do valor publicado para a mesma malha.

**Origem:** §2 item 6
