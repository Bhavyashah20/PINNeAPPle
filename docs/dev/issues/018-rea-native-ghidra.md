---
title: "REA: native binary analysis (FMUs, solvers) with a Ghidra provider"
labels: [enhancement, rea]
---

**Who has it:** REA `open_binary` + Ghidra/Hopper/IDA providers.

**What PINNeAPPle already has:** `rea_bridge` covers JavaScript and provenance; `fmu_inspect` covers the ELF symbol table. Ghidra could not be installed in the sandbox (GitHub Releases download blocked).

**Proposal:** `rea setup` with Ghidra on a machine with access; `ReaClient.open_binary/decompile` wrappers + a native-Evidence summarizer.

**Acceptance (rule D1 — independent reference):** Decompile `fmi2DoStep` from a Reference-FMU and locate the call into the integrator, with Evidence.

**Origin:** docs/dev/REA_INTEGRATION.md
