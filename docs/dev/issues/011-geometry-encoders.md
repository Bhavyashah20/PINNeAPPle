---
title: "Encoders de geometria (PointNet/DGCNN; B-rep UV-Net depois)"
labels: [enhancement, geometry, surrogate]
---

**Quem tem:** Neural Concept, Simcenter PhysicsAI (geometric deep learning).

**O que o PINNeAPPle já tem:** `geometry_features`, `geometry_intelligence` (descritores manuais).

**Proposta:** Encoders PointNet/DGCNN em `pinneapple_neural/architectures`, usados pelos itens de similaridade e surrogate de geometria.

**Critério de aceite (regra D1 — referência independente):** Invariância a permutação testada; ModelNet10 subset com acurácia reportada.

**Origem:** §2 item 11
