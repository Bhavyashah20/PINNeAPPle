---
title: "Polynomial chaos + índices de Sobol para UQ e sensibilidade"
labels: [enhancement, uq]
---

**Quem tem:** optiSLang, COMSOL UQ Module.

**O que o PINNeAPPle já tem:** Conformal/ensemble/MC-dropout; nenhum PCE.

**Proposta:** PCE por regressão (Hermite/Legendre) com índices de Sobol analíticos dos coeficientes.

**Critério de aceite (regra D1 — referência independente):** Função de Ishigami: índices de Sobol de primeira ordem e totais dentro de 1% dos valores fechados.

**Origem:** §2 item 10
