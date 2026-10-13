---
title: "Pipeline 'casos CFD → surrogate de campo a partir de STL/point cloud' + DoMINO real"
labels: [enhancement, surrogate]
---

**Quem tem:** Ansys SimAI (non-parametric), Simcenter PhysicsAI, Luminary SHIFT (DoMINO/GeoTransolver), Emmi AB-UPT.

**O que o PINNeAPPle já tem:** `Transolver`, MeshGraphNet, `physicsnemo_bridge`, `lpm.py`. **Atenção:** `pinneapple_physics/pinn_solver/domino.py` é um PINN de decomposição de domínio, não o DoMINO da NVIDIA — o nome confunde.

**Proposta:** Pipeline pronto: pasta de casos OpenFOAM → dataset (superfície+volume) → treino (Transolver / DoMINO via bridge) → Cd/Cl + campos → relatório. Renomear ou documentar o `domino.py` atual.

**Critério de aceite (regra D1 — referência independente):** Benchmark reproduzível em AhmedML (CC-BY-SA) com erro de Cd reportado e comparado a um número publicado.

**Origem:** §2 item 3
