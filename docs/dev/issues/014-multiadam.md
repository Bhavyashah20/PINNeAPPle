---
title: "Otimizador MultiAdam para perdas PINN multi-termo"
labels: [enhancement, training]
---

**Quem tem:** PINNacle `src/optimizer/multiadam.py` (MIT).

**O que o PINNeAPPle já tem:** NTK weighting e Adam→L-BFGS existem; MultiAdam não.

**Proposta:** Estatísticas de Adam separadas por grupo de perda.

**Critério de aceite (regra D1 — referência independente):** Poisson com BCs desbalanceadas: converge onde Adam padrão estagna (erro relativo menor em 3 seeds).

**Origem:** ledger: PINNacle
