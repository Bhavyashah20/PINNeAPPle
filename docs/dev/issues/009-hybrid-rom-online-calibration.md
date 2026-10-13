---
title: "Hybrid dynamic ROM (physics + learned residual) calibrated online from sensors"
labels: [enhancement, digital-twin, rom]
---

**Who has it:** Twin Builder (Neural ODE, TFT, Hammerstein LPV, fusion modeling), ROM Builder (LSTM/GRU).

**What PINNeAPPle already has:** Neural ROM, Koopman, SINDy, Kalman, 4D-Var — but not the physics + residual composition updated online.

**Proposal:** Physics ROM + learned correction, updated by Kalman/EnKF inside `digital_twin`.

**Acceptance (rule D1 — independent reference):** Synthetic system with a known missing term: the learned residual converges to it (error < 5%).

**Origin:** §2 item 9
