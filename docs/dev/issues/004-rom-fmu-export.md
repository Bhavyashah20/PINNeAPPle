---
title: "Exportar ROMs e surrogates como FMU (co-simulação)"
labels: [enhancement, rom, digital-twin, fmi]
---

**Quem tem:** Ansys Twin Builder, Simcenter ROM Builder (FMI + ONNX), Altair/Siemens romAI.

**O que o PINNeAPPle já tem:** ROMs (POD, DMD, OpInf, neural ROM), export ONNX/TorchScript; `modelica/fmi_runner.py` só importa; novo `fmu_inspect.py` valida FMUs.

**Proposta:** Empacotar ROM linear (POD/DMD/OpInf) como FMU 2.0/3.0 CS em C gerado (sem dependência) e ROM neural via ONNX Runtime.

**Critério de aceite (regra D1 — referência independente):** Teste: FMU gerada passa no `inspect_fmu` (exporta todas as funções), simula via fmpy e reproduz o ROM Python a 1e-10.

**Origem:** §2 item 4
