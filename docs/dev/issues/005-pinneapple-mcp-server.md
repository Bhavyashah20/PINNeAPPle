---
title: "Servidor MCP do PINNeAPPle a partir do PhysicsToolRegistry"
labels: [enhancement, llm, agents]
---

**Quem tem:** PyAnsys (MCP para PyFluent/PyMechanical/PyMAPDL), Luminary SDK (MCP documentado).

**O que o PINNeAPPle já tem:** `PhysicsToolRegistry` já tem `input_schema`/`output_schema`; `agent_loop` usa o registry.

**Proposta:** Gerar ferramentas MCP a partir do registry (stdio), com allowlist e o `PhysicsGuardrail` na saída.

**Critério de aceite (regra D1 — referência independente):** Teste: `tools/list` expõe as tools registradas; uma chamada `reference_license_policy` e uma de solver retornam JSON válido.

**Origem:** §2 item 5
