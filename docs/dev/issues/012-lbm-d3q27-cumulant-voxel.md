---
title: "LBM D3Q27 cumulant + voxelização de STL + armazenamento FP16"
labels: [enhancement, solvers, gpu]
---

**Quem tem:** ultraFluidX, PowerFLOW, XFlow, M-Star, SimScale Pacefish, FluidX3D (FP16 storage).

**O que o PINNeAPPle já tem:** `lbm.py`: D2Q9/D3Q19 BGK + Smagorinsky; `lbm_tunnel`.

**Proposta:** Colisão cumulant D3Q27, voxelização de STL (GPU via torch), armazenamento FP16 com aritmética FP32 (ideia, não código — FluidX3D é não comercial).

**Critério de aceite (regra D1 — referência independente):** Poiseuille 3-D e Taylor–Green 3-D contra solução exata; esfera Re=100 com Cd dentro de 3% da correlação de Schiller–Naumann.

**Origem:** §3
