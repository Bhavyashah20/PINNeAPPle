---
title: "FNO incremental (iFNO): crescer modos de Fourier durante o treino"
labels: [enhancement, neural-operators]
---

**Quem tem:** neuraloperator `IncrementalFNOTrainer` (MIT), George et al. TMLR 2024.

**O que o PINNeAPPle já tem:** FNO em `architectures/neural_operators/fno.py`, sem crescimento de modos.

**Proposta:** Callback de treino (loss-gap e explained-variance) que aumenta `modes` do FNO.

**Critério de aceite (regra D1 — referência independente):** Burgers 1-D: mesma acurácia final com menos FLOPs de treino que FNO de modos fixos (seed fixa).

**Origem:** ledger: neuraloperator
