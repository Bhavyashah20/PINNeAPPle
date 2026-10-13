---
title: "AMR guiada pela solução (quadtree, critério wavelet/gradiente)"
labels: [enhancement, meshing, solvers]
---

**Quem tem:** CONVERGE (AMR a cada passo), Basilisk (adapt_wavelet), Luminary SDK.

**O que o PINNeAPPle já tem:** Só refinamento estático (camada limite, curvatura, níveis do snappy).

**Proposta:** Quadtree 2-D com critério wavelet para `fvm` e `shallow_water_fv`.

**Critério de aceite (regra D1 — referência independente):** Dam-break de Ritter: mesmo erro L1 da malha uniforme fina com ≤ 40% das células.

**Origem:** §2 item 7
