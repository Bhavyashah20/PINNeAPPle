"""From a Modelica file to PINNeAPPle data: compile with OpenModelica, simulate as an FMU, compare with omc.

Builds an FMI 2.0 Co-Simulation FMU of `models/LadleCooling.mo`, simulates it through fmpy, simulates the same
model with omc directly, and checks that both agree; then packs the FMU run as a PhysicalSample, the format the
rest of the library trains on.

Needs OpenModelica: `omc` on the PATH, or Docker with `openmodelica/openmodelica:v1.24.4-minimal`
(PINNEAPPLE_OMC_IMAGE picks another tag). About 20 seconds.
"""
import os

import numpy as np

from pinneapple_simulation.external_solvers.modelica import FMUSimConfig, fmu_to_upd, simulate_fmu
from pinneapple_simulation.external_solvers.modelica.omc import (
    build_fmu,
    omc_available,
    simulate_mo,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
model = os.path.join(HERE, "models", "LadleCooling.mo")
print("OpenModelica:", omc_available())

fmu = build_fmu(model, "LadleCooling", OUT)                       # FMI 2.0 Co-Simulation
cfg = FMUSimConfig(stop_time=4 * 3600, output_interval=60, parameters={"k_lin": 3.0},
                   output_variables=["T_melt", "T_shell"])
a = simulate_fmu(fmu, cfg)
b = simulate_mo(model, "LadleCooling", stop_time=4 * 3600, number_of_intervals=240, parameters={"k_lin": 3.0},
                variables=["time", "T_melt", "T_shell"], workdir=os.path.join(OUT, "omc_run"))
for k in ("T_melt", "T_shell"):
    diff = np.max(np.abs(a[k] - np.interp(a["time"], b["time"], b[k])))
    print(f"{k}: {a[k][0] - 273.15:.0f} °C -> {a[k][-1] - 273.15:.0f} °C after 4 h; FMU vs omc max diff {diff:.3f} K")
sample = fmu_to_upd(fmu, cfg)
print("PhysicalSample fields:", {k: tuple(v.shape) for k, v in sample.state.items()})
