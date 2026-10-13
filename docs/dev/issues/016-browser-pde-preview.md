---
title: "Prévia de PDE no navegador (SymbolicPDE → GLSL), estilo VisualPDE"
labels: [enhancement, frontend]
---

**Quem tem:** VisualPDE (MIT) — localizado com REA: `setRDEquations`, url_codec, auto steps-per-frame.

**O que o PINNeAPPle já tem:** `pinneapple_app/frontend` (React/Vite); `SymbolicPDE`.

**Proposta:** Emitir GLSL de um `ProblemSpec` 2-D (reação-difusão) e rodar no frontend; estado compartilhável na URL.

**Critério de aceite (regra D1 — referência independente):** Gray–Scott: padrão estacionário igual ao do solver numpy (correlação > 0.99) em 5 presets.

**Origem:** ledger: VisualPDE
