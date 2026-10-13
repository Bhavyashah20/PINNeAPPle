---
title: "Export ROMs and surrogates as FMUs (co-simulation)"
labels: [enhancement, rom, digital-twin, fmi]
---

**Who has it:** Ansys Twin Builder, Simcenter ROM Builder (FMI + ONNX), Altair/Siemens romAI.

**What PINNeAPPle already has:** ROMs (POD, DMD, OpInf, neural ROM), ONNX/TorchScript export; `modelica/fmi_runner.py` only imports FMUs; the new `fmu_inspect.py` validates them.

**Proposal:** Package linear ROMs (POD/DMD/OpInf) as FMI 2.0/3.0 co-simulation FMUs from generated C (no runtime dependency), and neural ROMs through ONNX Runtime.

**Acceptance (rule D1 — independent reference):** The generated FMU passes `inspect_fmu` (exports every required function), simulates with fmpy and reproduces the Python ROM to 1e-10.

**Origin:** §2 item 4
