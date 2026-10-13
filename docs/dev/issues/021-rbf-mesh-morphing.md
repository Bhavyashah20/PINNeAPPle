---
title: "Morphing paramétrico de malha tesselada (RBF) sem CAD"
labels: [enhancement, geometry, design-optimization]
---

**Quem tem:** STAR-CCM+ 2506 Parametric Morphing, SU2 RBF.

**O que o PINNeAPPle já tem:** Ausente (otimização depende de geometria paramétrica).

**Proposta:** Morphing RBF com pontos de controle e vetores de parâmetro sobre STL.

**Critério de aceite (regra D1 — referência independente):** Interpolação exata nos pontos de controle; deslocamento nulo longe do suporte; malha sem inversão (Jacobiano > 0).

**Origem:** §2 item 13
