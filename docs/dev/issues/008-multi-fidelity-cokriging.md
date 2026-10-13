---
title: "Surrogate multi-fidelidade (co-kriging Kennedy–O'Hagan)"
labels: [enhancement, surrogate]
---

**Quem tem:** optiSLang AI+, reuso de dados do HEEDS, COMSOL.

**O que o PINNeAPPle já tem:** Transfer/meta-learning existem; multi-fidelidade explícita não.

**Proposta:** Co-kriging autoregressivo sobre o GP numpy do `design_optimizer` (ex.: LBM grosso + OpenFOAM fino).

**Critério de aceite (regra D1 — referência independente):** Funções de Forrester (forma fechada): co-kriging com 4 pontos HF + 11 LF supera GP só-HF (RMSE).

**Origem:** §2 item 8
