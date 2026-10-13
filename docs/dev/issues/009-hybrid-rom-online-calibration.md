---
title: "ROM dinâmico híbrido (física + resíduo aprendido) calibrado online com sensores"
labels: [enhancement, digital-twin, rom]
---

**Quem tem:** Twin Builder (Neural ODE, TFT, Hammerstein LPV, fusion modeling), ROM Builder (LSTM/GRU).

**O que o PINNeAPPle já tem:** Neural ROM, Koopman, SINDy, Kalman, 4D-Var — sem a composição física+resíduo online.

**Proposta:** Modelo ROM físico + correção aprendida, atualizada por Kalman/EnKF no `digital_twin`.

**Critério de aceite (regra D1 — referência independente):** Sistema sintético com termo faltante conhecido: o resíduo aprendido converge para o termo (erro < 5%).

**Origem:** §2 item 9
