---
title: "Viewer com cache de inferência por hash e diff surrogate × verdade"
labels: [enhancement, frontend, twin3d]
---

**Quem tem:** NVIDIA Omniverse digital-twins-for-fluid-simulation blueprint (cache .npz por SHA256).

**O que o PINNeAPPle já tem:** `twin3d` viewer, `pinneapple_app`.

**Proposta:** Re-inferência a cada mudança de parâmetro, cache por hash dos inputs, aba de diff.

**Critério de aceite (regra D1 — referência independente):** Mesma entrada → mesmo hash e acerto de cache; diff zero quando surrogate = verdade.

**Origem:** §2 item 15
