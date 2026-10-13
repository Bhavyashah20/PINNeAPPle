---
title: "Score de similaridade geométrica (0–1) devolvido com cada predição de surrogate"
labels: [enhancement, surrogate, uq]
---

**Quem tem:** Simcenter/Altair PhysicsAI devolvem um *geometric similarity score* e o Ansys SimAI um *confidence level* que diz se o design novo está dentro do domínio de treino.

**O que o PINNeAPPle já tem:** `GeometryOODGuardrail`, `analysis/trust/trust_gate.py`, UQ conformal/ensemble em `pinneapple_analysis/uncertainty`.

**Proposta:** Embedding de geometria (descritores de `geometry_features` + curvatura de `geometry/ops/curvature.py`) → distância ao conjunto de treino (kNN/Mahalanobis) → calibração para [0,1] em hold-out. Devolver junto de toda predição e no relatório de evidência.

**Critério de aceite (regra D1 — referência independente):** Teste: designs do treino têm score ≥ 0.9; design deformado progressivamente tem score monotonicamente decrescente; correlação de Spearman positiva entre (1-score) e erro real num conjunto sintético.

**Origem:** docs/dev/ANALISE_FEATURES_COMERCIAIS.md §2 item 1
