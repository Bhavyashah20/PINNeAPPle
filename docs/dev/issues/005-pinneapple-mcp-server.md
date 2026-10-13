---
title: "PINNeAPPle MCP server generated from PhysicsToolRegistry"
labels: [enhancement, llm, agents]
---

**Who has it:** PyAnsys (MCP servers for PyFluent/PyMechanical/PyMAPDL), Luminary SDK (documented MCP server).

**What PINNeAPPle already has:** `PhysicsToolRegistry` already carries `input_schema`/`output_schema`; `agent_loop` uses the registry.

**Proposal:** Generate MCP tools from the registry (stdio), with an allowlist and `PhysicsGuardrail` on outputs.

**Acceptance (rule D1 — independent reference):** `tools/list` exposes the registered tools; a `reference_license_policy` call and one solver call return valid JSON.

**Origin:** §2 item 5
