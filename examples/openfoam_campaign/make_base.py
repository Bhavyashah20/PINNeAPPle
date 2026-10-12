"""Write the base case of the example campaign: a 2-D trough with a liquid bath fed by an inclined jet
(interFoam, k-omega SST). The campaign in plan.json changes the jet speed and elevation per run.

    python examples/openfoam_campaign/make_base.py
    python -m pinneapple_simulation.external_solvers.openfoam.campaign examples/openfoam_campaign/plan.json --status
"""
import os

from pinneapple_design.geometry.trough import TroughSection
from pinneapple_simulation.numerical_solvers.free_surface import FreeSurfaceFlow, Phase

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
sec = TroughSection(length=6.0, inlet_x=1.0, slope_deg=0.0, cell=0.04)
flow = FreeSurfaceFlow(mesh=sec.write_mesh(os.path.join(OUT, "mesh")), two_d=True,
                       roles={"inlet": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},
                       inlet_velocity={"inlet": (1.0, 0.0, -2.5)}, level=sec.level(), gravity=sec.gravity(9.81),
                       liquid=Phase("metal", rho=7200.0, nu=8.472e-6), sigma=1.8, end_time=10.0, sample_every=0.1)
flow.write(os.path.join(OUT, "base"), procs=2)
print("base case:", os.path.join(OUT, "base"))
