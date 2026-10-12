# Modelica models as Physics AI data

| Script | What it does |
|---|---|
| `01_modelica_to_fmu.py` | `.mo` → FMI 2.0 Co-Simulation FMU with OpenModelica; FMU run checked against omc; run packed as a `PhysicalSample` |
| `02_ladle_surrogate.py` | 80-run Latin-hypercube sweep of a hot-metal ladle cooling model → MLP surrogate, tested on 16 unseen runs against a nearest-run baseline, with a no-warming physics check |
| `03_tank_inverse_pinn.py` | Discharge coefficient of a draining tank identified by a PINN from 21 noisy readings; extrapolation past the readings |

Models: `models/DrainingTank.mo`, `models/LadleCooling.mo`. OpenModelica is used from the PATH or from Docker
(`docker pull openmodelica/openmodelica:v1.24.4-minimal`). Results and discussion: `docs/core_concepts/modelica.md`.
