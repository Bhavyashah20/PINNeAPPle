---
title: "Loop fechado: o surrogate escolhe a próxima simulação (aquisição → campaign → re-treino)"
labels: [enhancement, active-learning, orchestration]
---

**Quem tem:** HEEDS adaptive sampling, PhysicsX (UQ → nova simulação), Neural Concept.

**O que o PINNeAPPle já tem:** `pinneapple_core/data.py::ActiveSampler`, EI em `design_optimizer`, `external_solvers/openfoam/campaign.py`, `pinneapple_registry`.

**Proposta:** Orquestrador que conecta função de aquisição → `campaign` (OpenFOAM/LBM/qualquer solver do registry) → dataset store → re-treino → nova aquisição, com parada por erro de validação ou orçamento.

**Critério de aceite (regra D1 — referência independente):** Teste com solver barato (função analítica + `stable_fluids`): o loop ativo atinge o mesmo erro com ≤ 60% das amostras de um LHS de mesmo tamanho final (média de 5 seeds).

**Origem:** §2 item 2
