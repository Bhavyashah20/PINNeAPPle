---
title: "Modelos do PhysicsNeMo ausentes: FIGConvNet, TopoDiff, DLWP-HEALPix"
labels: [enhancement, neural-operators]
---

**Quem tem:** NVIDIA PhysicsNeMo (Apache-2.0).

**O que o PINNeAPPle já tem:** `physicsnemo_bridge` existe; esses modelos não.

**Proposta:** Expor via bridge primeiro (sem reimplementar); reimplementar só o que justificar.

**Critério de aceite (regra D1 — referência independente):** Smoke test de forward + um caso pequeno de treino por modelo.

**Origem:** ledger: PhysicsNeMo
