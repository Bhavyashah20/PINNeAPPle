---
title: "REA: análise nativa de binários (FMUs, solvers) com Ghidra"
labels: [enhancement, rea]
---

**Quem tem:** REA `open_binary` + provider Ghidra/Hopper/IDA.

**O que o PINNeAPPle já tem:** `rea_bridge` cobre JS/proveniência; `fmu_inspect` cobre a tabela de símbolos ELF. Ghidra não pôde ser instalado no sandbox (download do GitHub Releases bloqueado).

**Proposta:** `rea setup` com Ghidra numa máquina com acesso; wrapper `ReaClient.open_binary/decompile` + summarizer de Evidence nativa.

**Critério de aceite (regra D1 — referência independente):** Decompilar `fmi2DoStep` de uma Reference-FMU e localizar a chamada ao integrador (Euler) com Evidence.

**Origem:** docs/dev/REA_INTEGRATION.md
